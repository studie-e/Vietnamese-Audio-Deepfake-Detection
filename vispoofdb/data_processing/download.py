"""
download.py
-----------
Tải audio từ HuggingFace Datasets Server API (không cần token).
Dataset: dolly-vn/dolly-audio-1000h-vietnamese

Cài thư viện:
    pip install requests tqdm

Cách chạy:
    python download.py                            # tải 7000 file mặc định
    python download.py --num_samples 1000         # tải 1000 file
    python download.py --num_samples 7000 --workers 16   # tải nhanh hơn
    python download.py --save_corpus              # kèm corpus.txt
    
Resume nếu bị gián đoạn:
    python download.py                            # tự động load checkpoint và resume

Lỗi 502 Bad Gateway:
    - Script sẽ retry tự động với exponential backoff (max 5 lần)
    - Skip batch lỗi và tiếp tục batch tiếp theo
    - Lưu checkpoint để resume lần sau nếu bị dừng
"""

import os
import time
import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# ─────────────────────────── CONFIG ─────────────────────────────
API_ROWS_URL  = "https://datasets-server.huggingface.co/rows"
DATASET       = "dolly-vn/dolly-audio-1000h-vietnamese"
CONFIG        = "default"
SPLIT         = "train"
BATCH_SIZE    = 100          # max rows per API request
DEFAULT_N     = 7000
DEFAULT_OUT   = "vispoofdb/data/raw/real"
DEFAULT_WORKERS = 10         # số luồng tải song song
MAX_RETRY     = 5             # retry nhiều hơn cho lỗi 502
RETRY_DELAY   = 1.0          # giây (sẽ tăng exponential)
MAX_BACKOFF   = 60            # giây tối đa khi backoff
CHECKPOINT_FILE = ".download_checkpoint.json"
# ────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_samples", type=int,   default=DEFAULT_N,
                   help=f"Số file cần tải (mặc định: {DEFAULT_N})")
    p.add_argument("--output_dir",  type=str,   default=DEFAULT_OUT,
                   help=f"Thư mục lưu (mặc định: {DEFAULT_OUT})")
    p.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS,
                   help=f"Số luồng song song (mặc định: {DEFAULT_WORKERS})")
    p.add_argument("--save_corpus", action="store_true",
                   help="Lưu corpus.txt (filename|text|voice_id)")
    p.add_argument("--offset",      type=int,   default=0,
                   help="Bắt đầu từ row thứ bao nhiêu (mặc định: 0)")
    return p.parse_args()


def fetch_batch(offset: int, length: int, session: requests.Session) -> list[dict]:
    """Gọi API lấy metadata một batch rows."""
    params = {
        "dataset": DATASET,
        "config":  CONFIG,
        "split":   SPLIT,
        "offset":  offset,
        "length":  length,
    }
    for attempt in range(MAX_RETRY):
        try:
            r = session.get(API_ROWS_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json().get("rows", [])
        except requests.exceptions.HTTPError as e:
            # Lỗi 502, 503, 429 → nên retry với exponential backoff
            if e.response.status_code in [502, 503, 429]:
                if attempt < MAX_RETRY - 1:
                    wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                    print(f"\n⚠️  HTTP {e.response.status_code} at offset={offset}. Retry trong {wait_time:.0f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n⚠️  Lỗi fetch batch offset={offset} (HTTP {e.response.status_code}) sau {MAX_RETRY} lần thử. Skip batch này.")
                    return []
            else:
                print(f"\n⚠️  Lỗi HTTP {e.response.status_code} at offset={offset}: {e}")
                return []
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRY - 1:
                wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                print(f"\n⚠️  Timeout at offset={offset}. Retry trong {wait_time:.0f}s...")
                time.sleep(wait_time)
            else:
                print(f"\n⚠️  Timeout fetch batch offset={offset} sau {MAX_RETRY} lần thử. Skip batch này.")
                return []
        except Exception as e:
            if attempt < MAX_RETRY - 1:
                wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                time.sleep(wait_time)
            else:
                print(f"\n⚠️  Lỗi fetch batch offset={offset}: {e}")
                return []


def download_one(task: dict, out_dir: Path, session: requests.Session) -> tuple[bool, str]:
    """Tải 1 file audio về disk. Trả về (success, filename)."""
    audio_url  = task["url"]
    filename   = task["filename"]
    out_path   = out_dir / filename

    # Bỏ qua nếu đã tồn tại
    if out_path.exists():
        return True, filename

    for attempt in range(MAX_RETRY):
        try:
            r = session.get(audio_url, timeout=60, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True, filename
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [502, 503, 429]:
                if attempt < MAX_RETRY - 1:
                    wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                    time.sleep(wait_time)
                else:
                    return False, filename
            else:
                return False, filename
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRY - 1:
                wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                time.sleep(wait_time)
            else:
                return False, filename
        except Exception:
            if attempt < MAX_RETRY - 1:
                wait_time = min(RETRY_DELAY * (2 ** attempt), MAX_BACKOFF)
                time.sleep(wait_time)
            else:
                return False, filename


def load_checkpoint(out_dir: Path) -> dict:
    """Load checkpoint nếu tồn tại."""
    cp_file = out_dir / CHECKPOINT_FILE
    if cp_file.exists():
        try:
            with open(cp_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {"offset": 0, "tasks_count": 0}

def save_checkpoint(out_dir: Path, offset: int, tasks_count: int):
    """Lưu checkpoint để resume sau."""
    cp_file = out_dir / CHECKPOINT_FILE
    with open(cp_file, "w") as f:
        json.dump({"offset": offset, "tasks_count": tasks_count}, f)

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Dataset : {DATASET}")
    print(f"📂 Output  : {out_dir.resolve()}")
    print(f"🎯 Số file : {args.num_samples}")
    print(f"⚡ Workers : {args.workers}")
    print(f"{'─'*50}")

    session = requests.Session()
    session.headers.update({"User-Agent": "vispoofdb-downloader/1.0"})

    # ── Load checkpoint (nếu resume) ────────────────────────────
    checkpoint = load_checkpoint(out_dir)
    if checkpoint["tasks_count"] > 0 and args.offset == 0:
        print(f"\n🔄 Resume từ checkpoint: offset={checkpoint['offset']}, tasks={checkpoint['tasks_count']}")
        args.offset = checkpoint["offset"]

    # ── Bước 1: Thu thập metadata theo batch ────────────────────
    print("\n📋 Đang lấy metadata...")
    tasks = []
    total_needed = args.num_samples
    offset = args.offset
    skipped_batches = 0

    with tqdm(total=total_needed, desc="Metadata", unit="row", ncols=80) as pbar:
        while len(tasks) < total_needed:
            batch_size = min(BATCH_SIZE, total_needed - len(tasks))
            rows = fetch_batch(offset, batch_size, session)
            
            if not rows:
                # Nếu batch này lỗi, skip và tiếp tục batch tiếp theo
                skipped_batches += 1
                if skipped_batches > 10:  # Nếu skip quá nhiều → dừng
                    print(f"\n⚠️  Đã skip {skipped_batches} batch liên tiếp. Dừng lại.")
                    break
                offset += BATCH_SIZE  # Move to next batch
                continue

            skipped_batches = 0  # Reset counter khi thành công
            for row in rows:
                r = row.get("row", {})
                audio_list = r.get("audio", [])
                if not audio_list:
                    continue
                audio_url = audio_list[0].get("src")
                if not audio_url:
                    continue
                tasks.append({
                    "url":      audio_url,
                    "filename": r.get("audio_filename", f"audio_{offset}.wav"),
                    "text":     r.get("text", ""),
                    "voice_id": r.get("voice_id", ""),
                })
                pbar.update(1)

            offset += len(rows)
            save_checkpoint(out_dir, offset, len(tasks))  # Lưu progress

    print(f"✅ Tổng metadata: {len(tasks)} rows")
    if skipped_batches > 0:
        print(f"⚠️  Đã skip {skipped_batches} batch do lỗi API")

    # ── Bước 2: Tải song song ────────────────────────────────────
    print(f"\n🚀 Bắt đầu tải {len(tasks)} file...\n")

    success_count = 0
    fail_count    = 0
    corpus_rows   = []
    lock          = Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, task, out_dir, session): task
            for task in tasks
        }
        with tqdm(total=len(tasks), desc="Downloading", unit="file", ncols=80) as pbar:
            for future in as_completed(futures):
                ok, fname = future.result()
                task = futures[future]
                with lock:
                    if ok:
                        success_count += 1
                        if args.save_corpus:
                            corpus_rows.append(
                                f"{task['filename']}|{task['text']}|{task['voice_id']}"
                            )
                    else:
                        fail_count += 1
                        tqdm.write(f"  ✗ Lỗi: {fname}")
                    pbar.update(1)
                    pbar.set_postfix(ok=success_count, fail=fail_count)

    # ── Bước 3: Lưu corpus.txt ───────────────────────────────────
    if args.save_corpus and corpus_rows:
        corpus_path = out_dir.parent.parent / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("filename|text|voice_id\n")
            f.write("\n".join(corpus_rows))
        print(f"\n📄 Corpus: {corpus_path}")

    # ── Tóm tắt ─────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"✅ Thành công : {success_count} file")
    if fail_count:
        print(f"❌ Thất bại  : {fail_count} file")
    print(f"📁 Thư mục   : {out_dir.resolve()}")
    print(f"{'─'*50}")
    
    # Xóa checkpoint khi tải xong
    cp_file = out_dir / CHECKPOINT_FILE
    if cp_file.exists():
        cp_file.unlink()
        print(f"\n🗑️  Xóa checkpoint (tải xong)")


if __name__ == "__main__":
    main()