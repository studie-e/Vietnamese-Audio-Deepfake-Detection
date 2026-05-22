from pathlib import Path
import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings

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
SAVE_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_wav2vec'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Tải Mô hình Wav2Vec2 từ Facebook ---
print("Đang tải 'não bộ' Wav2Vec2 (Lần đầu sẽ mất chút thời gian tải ~360MB)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Đã tải xong! Đang chạy trên thiết bị: {device.type.upper()}")

# --- 3. Hàm trích xuất ---
def extract_wav2vec_features(file_path):
    try:
        y, sr = librosa.load(str(file_path), sr=16000)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"Lỗi file {Path(file_path).name}: {e}")
        return None

# --- 4. Load metadata và trích xuất ---
def process_data():
    print(f"\nBắt đầu trích xuất Wav2Vec2 theo metadata.csv (split-aware)...")
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất Wav2Vec2"):
        file_path = CLEAN_DATA_DIR / row['file_path']
        label = 0 if row['label'] == 'real' else 1
        split = row['split']

        if not file_path.exists():
            print(f"[WARN] Không tìm thấy file: {file_path}")
            continue

        feat = extract_wav2vec_features(file_path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            splits.append(split)

    X = np.array(X)
    y = np.array(y)
    splits = np.array(splits)

    print(f"\nHOÀN THÀNH! Đã lưu ma trận {len(X)} x 768")
    print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

    np.save(SAVE_DIR / 'X_wav2vec.npy', X)
    np.save(SAVE_DIR / 'y_wav2vec.npy', y)
    np.save(SAVE_DIR / 'splits_wav2vec.npy', splits)
    print(f"Đã lưu tại: {SAVE_DIR}")
    print("  X_wav2vec.npy, y_wav2vec.npy, splits_wav2vec.npy")

if __name__ == "__main__":
    process_data()
