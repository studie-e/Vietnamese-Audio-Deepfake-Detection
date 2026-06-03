# File: src/data_model/mlp_features.py
import os
import glob
import numpy as np
import librosa
from tqdm import tqdm # Để hiển thị thanh tiến trình

# Định nghĩa đường dẫn
DATA_DIR = "data/clean_data"
SAVE_DIR = "data/features_model/MLP"

# Đảm bảo thư mục lưu tồn tại
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=40):
    """Đọc file audio và trích xuất trung bình MFCC làm đặc trưng"""
    try:
        # Load audio, sr=None để giữ nguyên sample rate gốc (hoặc ép về 16000)
        y, sr = librosa.load(file_path, sr=16000)
        # Trích xuất MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Lấy giá trị trung bình của các frame theo từng hệ số MFCC để tạo mảng 1D
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_data():
    X = []
    y = []
    
    # Định nghĩa nhãn: real = 0, ai = 1
    classes = {"real": 0, "ai": 1}
    
    for class_name, label in classes.items():
        folder_path = os.path.join(DATA_DIR, class_name)
        # Tìm tất cả file .wav trong thư mục
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        print(f"Processing {len(wav_files)} files in '{class_name}' folder...")
        
        for file_path in tqdm(wav_files):
            features = extract_mfcc(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
                
    # Chuyển list sang numpy array
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nShape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    # Lưu vào file .npy
    np.save(os.path.join(SAVE_DIR, "X_mlp.npy"), X)
    np.save(os.path.join(SAVE_DIR, "y_mlp.npy"), y)
    print(f"Features saved to {SAVE_DIR}")

if __name__ == "__main__":
    process_data()