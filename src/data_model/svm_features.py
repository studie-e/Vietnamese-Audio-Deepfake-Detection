import os
import librosa
import numpy as np
from tqdm import tqdm

DATA_REAL = 'data/clean_data/real'
DATA_AI = 'data/clean_data/ai'
SAVE_DIR = os.path.join('data', 'features_model', 'svm')
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Lỗi file {file_path}: {e}")
        return None

def process_all_data():
    X, y = [], []
    
    # Xử lý Người thật (Nhãn 0)
    print("--- ĐANG TRÍCH XUẤT GIỌNG NGƯỜI THẬT ---")
    real_files = [os.path.join(DATA_REAL, f) for f in os.listdir(DATA_REAL) if f.endswith('.wav')]
    for f in tqdm(real_files):
        feat = extract_mfcc(f)
        if feat is not None:
            X.append(feat)
            y.append(0)

    # Xử lý Giọng AI (Nhãn 1)
    print("\n--- ĐANG TRÍCH XUẤT GIỌNG AI ---")
    ai_files = [os.path.join(DATA_AI, f) for f in os.listdir(DATA_AI) if f.endswith('.wav')]
    for f in tqdm(ai_files):
        feat = extract_mfcc(f)
        if feat is not None:
            X.append(feat)
            y.append(1)

    np.save(os.path.join(SAVE_DIR, 'X_all.npy'), np.array(X))
    np.save(os.path.join(SAVE_DIR, 'y_all.npy'), np.array(y))
    print(f"\n✅ Xong! Tổng cộng {len(X)} file đã được xử lý.")

if __name__ == "__main__":
    process_all_data()