from pathlib import Path
import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- 1. Cấu hình thư mục ---
BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'clean_data'
METADATA_PATH = CLEAN_DATA_DIR / 'metadata.csv'
SAVE_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'svm'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(str(file_path), sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Lỗi file {Path(file_path).name}: {e}")
        return None

def process_all_data():
    print("Bắt đầu trích xuất MFCC (SVM) theo metadata.csv (split-aware)...")
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất MFCC SVM"):
        file_path = CLEAN_DATA_DIR / row['file_path']
        label = 0 if row['label'] == 'real' else 1
        split = row['split']

        if not file_path.exists():
            print(f"[WARN] Không tìm thấy file: {file_path}")
            continue

        feat = extract_mfcc(file_path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            splits.append(split)

    X = np.array(X)
    y = np.array(y)
    splits = np.array(splits)

    print(f"\nXong! Tổng cộng {len(X)} file đã được xử lý.")
    print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

    np.save(SAVE_DIR / 'X_all.npy', X)
    np.save(SAVE_DIR / 'y_all.npy', y)
    np.save(SAVE_DIR / 'splits_svm.npy', splits)
    print(f"Đã lưu tại: {SAVE_DIR}")
    print("  X_all.npy, y_all.npy, splits_svm.npy")

if __name__ == "__main__":
    process_all_data()