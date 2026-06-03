"""
scripts_data_process.py
=======================
Script tổng hợp chạy toàn bộ pipeline xử lý dữ liệu cho vispoofdb.

Thứ tự thực hiện:
    1. vispoofdb_clean_data.py          — Chuẩn hóa âm thanh thô → clean_data/
    2. vispoofdb_generate_metadata.py   — Tạo metadata.csv (train/test_seen/test_unseen)
    3. vidb_extract_mfcc.py             — Trích xuất MFCC 3D (cho XGBoost)
    4. vidb_extract_processing.py       — Trích xuất Mel-Spectrogram

Bỏ qua:
    - download_vivos_all.py    (xem README để lấy dữ liệu thô)
    - vispoofdb_data.ipynb     (notebook phân tích, không nằm trong pipeline)

Yêu cầu:
    - Đã có thư mục vispoofdb/data/raw/ chứa file âm thanh gốc
    - Đã cài đầy đủ dependencies (pip install -r requirements.txt)

Cách chạy (từ thư mục gốc dự án):
    python vispoofdb/scripts/scripts_data_process.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[2]
DATA_PROC_DIR   = BASE_DIR / "vispoofdb" / "data_processing"
PYTHON          = sys.executable   # Dùng chính Python đang chạy script này

# Danh sách các bước theo thứ tự (tên file, mô tả ngắn)
PIPELINE = [
    (
        DATA_PROC_DIR / "vispoofdb_clean_data.py",
        "Bước 1/4 — Làm sạch và chuẩn hóa âm thanh thô → vispoofdb/data/clean_data/",
    ),
    (
        DATA_PROC_DIR / "vispoofdb_generate_metadata.py",
        "Bước 2/4 — Tạo metadata.csv (phân chia train/test_seen/test_unseen)",
    ),
    (
        DATA_PROC_DIR / "vidb_extract_mfcc.py",
        "Bước 3/4 — Trích xuất MFCC 3D → vispoofdb/data/features_mfcc/",
    ),
    (
        DATA_PROC_DIR / "vidb_extract_processing.py",
        "Bước 4/4 — Trích xuất Mel-Spectrogram → vispoofdb/data/features_mel/",
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def separator(char="=", width=65):
    print(char * width)

def run_step(script_path: Path, description: str) -> bool:
    """Chạy một bước và trả về True nếu thành công."""
    separator()
    print(f"\n{description}")
    print(f"Script: {script_path.relative_to(BASE_DIR)}\n")
    separator("-")

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(BASE_DIR),         # Luôn chạy từ thư mục gốc dự án
        text=True,
    )
    elapsed = time.time() - start

    separator("-")
    if result.returncode == 0:
        print(f"Hoàn thành trong {elapsed:.1f}s\n")
        return True
    else:
        print(f"LỖI (exit code {result.returncode}) sau {elapsed:.1f}s")
        print("    Kiểm tra output bên trên để biết chi tiết lỗi.")
        print("    Pipeline bị dừng lại.\n")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    separator()
    print("  VISPOOFDB — DATA PROCESSING PIPELINE")
    print(f"  Thư mục gốc: {BASE_DIR}")
    print(f"  Python:      {PYTHON}")
    separator()
    print()

    # Kiểm tra thư mục raw tồn tại
    raw_dir = BASE_DIR / "vispoofdb" / "data" / "raw"
    if not raw_dir.exists():
        print(f"[ERROR] Không tìm thấy thư mục dữ liệu thô: {raw_dir}")
        print("        Xem README.md để biết cách lấy dữ liệu âm thanh gốc trước.")
        sys.exit(1)

    total_start = time.time()
    completed   = 0

    for script_path, description in PIPELINE:
        if not script_path.exists():
            print(f"[WARN] Bỏ qua — không tìm thấy file: {script_path}")
            continue

        success = run_step(script_path, description)
        if not success:
            sys.exit(1)   # Dừng nếu có lỗi
        completed += 1

    # Tổng kết
    total_elapsed = time.time() - total_start
    separator()
    print(f"\n🎉  HOÀN THÀNH PIPELINE! ({completed}/{len(PIPELINE)} bước)")
    print(f"    Tổng thời gian: {total_elapsed/60:.1f} phút")
    print()
    print("    Các thư mục đầu ra:")
    print("      • vispoofdb/data/clean_data/   — Dữ liệu âm thanh đã chuẩn hóa")
    print("      • vispoofdb/data/features_mfcc/ — MFCC 3D (dùng cho XGBoost)")
    print("      • vispoofdb/data/features_mel/  — Mel-Spectrogram")
    print()
    print("    Bước tiếp theo:")
    print("      Chạy vispoofdb/scripts/scripts_feature_extract.py")
    print("      để trích xuất đặc trưng cho từng mô hình (LFCC, SVM, MLP, Wav2Vec2).")
    separator()


if __name__ == "__main__":
    main()
