from pathlib import Path
import os
import sys
import librosa
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- 1. Cấu hình thư mục ---
BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'clean_data'
METADATA_PATH = CLEAN_DATA_DIR / 'metadata.csv'
SAVE_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_lfcc'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

from spafe.features.lfcc import lfcc

def extract_lfcc_features(file_path):
    try:
        y, sr = librosa.load(str(file_path), sr=16000)
        lfccs = lfcc(sig=y, fs=sr, num_ceps=40, nfilts=128)
        return np.mean(lfccs, axis=0)
    except Exception:
        return None

def process_data():
    print("Bắt đầu trích xuất LFCC theo metadata.csv (split-aware)...")
    print(f"Metadata: {METADATA_PATH}\n")

    if not METADATA_PATH.exists():
        print(f"[ERROR] Không tìm thấy metadata.csv tại {METADATA_PATH}")
        print("Hãy chạy vispoofdb_generate_metadata.py trước!")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Tổng số file trong metadata: {len(df)}")
    print(df.groupby(['label', 'split']).size().to_string())
    print()

    X, y, splits = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất LFCC"):
        file_path = CLEAN_DATA_DIR / row['file_path']
        label = 0 if row['label'] == 'real' else 1
        split = row['split']

        if not file_path.exists():
            print(f"[WARN] Không tìm thấy file: {file_path}")
            continue

        feat = extract_lfcc_features(file_path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            splits.append(split)

    X = np.array(X)
    y = np.array(y)
    splits = np.array(splits)

    print(f"\nHOÀN THÀNH! Tổng số tệp đã xử lý: {len(X)}")
    print(f"Kích thước ma trận đặc trưng: {X.shape} (Số file, 40 LFCC)")
    print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

    np.save(SAVE_DIR / 'X_lfcc.npy', X)
    np.save(SAVE_DIR / 'y_lfcc.npy', y)
    np.save(SAVE_DIR / 'splits_lfcc.npy', splits)
    print(f"Đã lưu tại: {SAVE_DIR}")
    print("  X_lfcc.npy, y_lfcc.npy, splits_lfcc.npy")

if __name__ == "__main__":
    process_data()