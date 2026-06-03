import os
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'data', 'clean_data') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'features_final')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CLASSES = {'real': 0, 'ai': 1}
X_data = []
y_label = []

# Cố định chiều rộng của MFCC (5 giây ở 16kHz, hop_length 512 thường ra 157 frames)
FIXED_WIDTH = 157 

print("🚀 Bắt đầu trích xuất MFCC (Bản V2 - Sửa lỗi lệch kích thước)...")

for class_name, label in CLASSES.items():
    class_dir = os.path.join(CLEAN_DATA_DIR, class_name)
    if not os.path.exists(class_dir): continue
        
    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    
    for filename in tqdm(wav_files, desc=f"Xử lý {class_name}"):
        file_path = os.path.join(class_dir, filename)
        try:
            y, sr = librosa.load(file_path, sr=16000)
            
            # Trích xuất MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            
            # --- BƯỚC QUAN TRỌNG: ÉP ĐÚNG KÍCH THƯỚC ---
            if mfccs.shape[1] < FIXED_WIDTH:
                # Nếu ngắn quá thì bù thêm số 0
                mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_WIDTH - mfccs.shape[1])), mode='constant')
            else:
                # Nếu dài quá thì cắt bớt
                mfccs = mfccs[:, :FIXED_WIDTH]
            
            X_data.append(mfccs)
            y_label.append(label)
            
        except Exception as e:
            print(f"Lỗi file {filename}: {e}")

# --- ĐÓNG GÓI ---
print("\n📦 Đang đóng gói dữ liệu...")
X_data = np.array(X_data) # Lúc này chắc chắn sẽ thành công 100%
y_label = np.array(y_label)

print(f"✅ Thành công! Kích thước X: {X_data.shape}, y: {y_label.shape}")

# Lưu file
np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X_data)
np.save(os.path.join(OUTPUT_DIR, 'y_label.npy'), y_label)

print(f"✨ Đã lưu dữ liệu tại: {OUTPUT_DIR}")