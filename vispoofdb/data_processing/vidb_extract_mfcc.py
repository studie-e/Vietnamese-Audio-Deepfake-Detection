from pathlib import Path
import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- 1. CẤU HÌNH ---
BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / "vispoofdb" / "data" / "clean_data"
METADATA_PATH = CLEAN_DATA_DIR / "metadata.csv"
OUTPUT_DIR = BASE_DIR / "vispoofdb" / "data" / "features_mfcc"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cố định chiều rộng của MFCC (5 giây ở 16kHz, hop_length 512 thường ra 157 frames)
FIXED_WIDTH = 157

print("Bắt đầu trích xuất MFCC theo metadata.csv (split-aware)...")
print(f"Metadata: {METADATA_PATH}\n")

# --- 2. Load metadata ---
if not METADATA_PATH.exists():
    print(f"[ERROR] Không tìm thấy metadata.csv tại {METADATA_PATH}")
    print("Hãy chạy vispoofdb_generate_metadata.py trước!")
    exit(1)

df = pd.read_csv(METADATA_PATH)
print(f"Tổng số file trong metadata: {len(df)}")
print(df.groupby(['label', 'split']).size().to_string())
print()

# --- 3. Trích xuất ---
X_data, y_label, splits = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất MFCC"):
    file_path = CLEAN_DATA_DIR / row['file_path']
    label = 0 if row['label'] == 'real' else 1
    split = row['split']

    if not file_path.exists():
        print(f"[WARN] Không tìm thấy file: {file_path}")
        continue

    try:
        y, sr = librosa.load(str(file_path), sr=16000)

        # Trích xuất MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Ép đúng kích thước
        if mfccs.shape[1] < FIXED_WIDTH:
            mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_WIDTH - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :FIXED_WIDTH]

        X_data.append(mfccs)
        y_label.append(label)
        splits.append(split)

    except Exception as e:
        print(f"Lỗi file {file_path.name}: {e}")

# --- 4. ĐÓNG GÓI ---
print("\nĐang đóng gói dữ liệu...")
X_data = np.array(X_data)
y_label = np.array(y_label)
splits = np.array(splits)

print(f"Thành công! Kích thước X: {X_data.shape}, y: {y_label.shape}")
print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

# Lưu file
np.save(OUTPUT_DIR / 'X_data.npy', X_data)
np.save(OUTPUT_DIR / 'y_label.npy', y_label)
np.save(OUTPUT_DIR / 'splits_mfcc.npy', splits)

print(f"Đã lưu dữ liệu tại: {OUTPUT_DIR}")
print("  X_data.npy, y_label.npy, splits_mfcc.npy")