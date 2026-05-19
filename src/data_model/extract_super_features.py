import os
import librosa
import numpy as np
from tqdm import tqdm

DATA_DIR = r'D:\Projects\seminar\Vietnamese-Audio-Deepfake-Detection\data\clean_data'
OUTPUT_DIR = r'D:\Projects\seminar\Vietnamese-Audio-Deepfake-Detection\data\fetures_model\MLP'

CLASSES = {'real': 0, 'ai': 1}
X_data = []
y_label = []

def extract_super_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. Trích xuất MFCC (40 con số)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # 2. Trích xuất Chroma (12 con số - liên quan đến cao độ)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 3. Spectral Centroid (1 con số - độ sáng âm thanh)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid.T, axis=0)
        
        # 4. Spectral Bandwidth (1 con số)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth.T, axis=0)
        
        # 5. Spectral Rolloff (1 con số)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff.T, axis=0)
        
        # 6. Zero Crossing Rate - ZCR (1 con số)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr.T, axis=0)
        
        # GỘP TẤT CẢ LẠI THÀNH 1 VECTOR DUY NHẤT (Dài 56 con số cực kỳ chất lượng)
        super_vector = np.hstack([
            mfccs_mean, 
            chroma_mean, 
            centroid_mean, 
            bandwidth_mean, 
            rolloff_mean, 
            zcr_mean
        ])
        return super_vector
        
    except Exception as e:
        print(f"Lỗi ở file {file_path}: {e}")
        return None

print("🚀 Bắt đầu trích xuất SUPER FEATURES...")
for class_name, label in CLASSES.items():
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir): continue
        
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    for filename in tqdm(wav_files, desc=f"Xử lý {class_name}"):
        file_path = os.path.join(class_dir, filename)
        features = extract_super_features(file_path)
        
        if features is not None:
            X_data.append(features)
            y_label.append(label)

X_data = np.array(X_data)
y_label = np.array(y_label)

print(f"\n✅ Đã tạo xong! Kích thước X: {X_data.shape} (Mỗi file có {X_data.shape[1]} đặc trưng)")

# Lưu lại để mang lên Kaggle chạy mô hình AI
np.save(os.path.join(OUTPUT_DIR, 'X_super_data.npy'), X_data)
np.save(os.path.join(OUTPUT_DIR, 'y_super_label.npy'), y_label)