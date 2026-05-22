from pathlib import Path
import os
import sys
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- 1. Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'clean_data'
METADATA_PATH = CLEAN_DATA_DIR / 'metadata.csv'
SAVE_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'MLP'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=40):
    """Đọc file audio và trích xuất trung bình MFCC làm đặc trưng"""
    try:
        y, sr = librosa.load(str(file_path), sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {Path(file_path).name}: {e}")
        return None

def process_data():
    print("Bắt đầu trích xuất MFCC (MLP) theo metadata.csv (split-aware)...")
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất MFCC MLP"):
        file_path = CLEAN_DATA_DIR / row['file_path']
        label = 0 if row['label'] == 'real' else 1
        split = row['split']

        if not file_path.exists():
            print(f"[WARN] Không tìm thấy file: {file_path}")
            continue

        features = extract_mfcc(file_path)
        if features is not None:
            X.append(features)
            y.append(label)
            splits.append(split)

    X = np.array(X)
    y = np.array(y)
    splits = np.array(splits)

    print(f"\nShape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

    np.save(SAVE_DIR / 'X_mlp.npy', X)
    np.save(SAVE_DIR / 'y_mlp.npy', y)
    np.save(SAVE_DIR / 'splits_mlp.npy', splits)
    print(f"Features saved to {SAVE_DIR}")
    print("  X_mlp.npy, y_mlp.npy, splits_mlp.npy")

if __name__ == "__main__":
    process_data()
