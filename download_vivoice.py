"""
Download ~7k real voice samples từ HuggingFace dataset: capleaf/viVoice
Sử dụng HF_TOKEN environment variable (không hardcode token!)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import login

# ──────────────────────────────────────────────────────────────────────────────
# 1. Setup paths
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "vispoofdb" / "data" / "raw" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find max existing file number
existing_files = list(OUTPUT_DIR.glob("*.wav"))
max_num = 0
for f in existing_files:
    try:
        num = int(f.stem)
        max_num = max(max_num, num)
    except:
        pass

start_num = max_num + 1
print(f"[*] Thư mục đã có {len(existing_files)} files")
print(f"    Sẽ bắt đầu đánh số từ: {start_num}")

REPO_ID = "capleaf/viVoice"
REPO_TYPE = "dataset"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Authenticate with HuggingFace (from environment)
# ──────────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    print("[!] Hay set HF_TOKEN environment variable")
    print("    Windows: $env:HF_TOKEN = 'hf_...'")
    print("    Linux/Mac: export HF_TOKEN='hf_...'")
    sys.exit(1)

print(f"[*] Dang xac thuc voi HuggingFace...")
try:
    login(token=HF_TOKEN)
    print("[✓] Xac thuc thanh cong!")
except Exception as e:
    print(f"[!] Loi xac thuc: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Load Parquet files and extract audio
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Dang tai va extract audio tu Parquet files...")

from datasets import load_dataset
from tqdm import tqdm
import random
import soundfile as sf

try:
    # Load dataset from Parquet
    print("    Loading dataset (co the mat vai phut tuy internet speed)...")
    dataset = load_dataset(REPO_ID, split="train", token=HF_TOKEN)
    print(f"[✓] Da load {len(dataset)} samples")
    
    # Random subset (~7000)
    random.seed(42)
    n_samples = min(7000, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)
    print(f"    Lay {n_samples} samples ngau nhien")
    
    # Extract audio
    print(f"\n[*] Dang ghi file .wav...")
    success_count = 0
    failed_count = 0
    
    # Find audio column
    audio_col = None
    for col in dataset.column_names:
        if "audio" in col.lower():
            audio_col = col
            break
    
    if audio_col is None:
        print(f"[!] Khong tim thay audio column. Co san: {dataset.column_names}")
        sys.exit(1)
    
    print(f"    Audio column: '{audio_col}'")
    
    for idx, sample_idx in enumerate(tqdm(indices, desc="Extract audio")):
        try:
            sample = dataset[sample_idx]
            audio_data = sample[audio_col]
            
            # Extract waveform and sample rate
            if isinstance(audio_data, dict):
                waveform = audio_data.get("array", audio_data.get("bytes", None))
                sr = audio_data.get("sampling_rate", 16000)
            else:
                waveform = audio_data
                sr = 16000
            
            if waveform is None:
                failed_count += 1
                continue
            
            # Save to file — dùng start_num + idx
            file_num = start_num + idx
            output_file = OUTPUT_DIR / f"{file_num}.wav"
            sf.write(str(output_file), waveform, sr)
            success_count += 1
            
        except Exception as e:
            failed_count += 1
            if failed_count <= 5:
                print(f"    [Warn] Index {sample_idx}: {str(e)[:50]}")
    
    print(f"\n[✓] Extract hoan thanh!")
    print(f"    Thanh cong: {success_count} files")
    print(f"    That bai: {failed_count} files")
    print(f"    Duong dan: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"[!] Loi: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Create metadata
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Tao metadata cho cac file...")

import pandas as pd
import librosa

metadata_list = []
wav_files = list(OUTPUT_DIR.glob("*.wav"))

for i, wav_file in enumerate(sorted(wav_files)):
    try:
        y, sr = librosa.load(str(wav_file), sr=16000)
        duration = len(y) / sr
        
        metadata_list.append({
            "ID": i + 1,
            "Ten_File": wav_file.name,
            "Nhan_So": 0,  # 0 = real
            "Phan_Loai": "That",
            "Do_Dai (giay)": round(duration, 2)
        })
    except Exception as e:
        if i < 5:
            print(f"    [Warn] {wav_file.name}: {e}")

if metadata_list:
    metadata_df = pd.DataFrame(metadata_list)
    metadata_csv = OUTPUT_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    
    print(f"[✓] Metadata da luu: {metadata_csv}")
    print(f"    Tong: {len(metadata_df)} files")
    print(f"\nVi du:")
    print(metadata_df.head())
else:
    print(f"[!] Khong co file audio nao de tao metadata")
