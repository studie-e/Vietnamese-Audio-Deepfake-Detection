import os
import librosa
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore') # Tắt cảnh báo đỏ cho đỡ rối mắt
from spafe.features.lfcc import lfcc

# --- 1. Cấu hình thư mục ---
DATA_REAL = 'data/clean_data/real'
DATA_AI = 'data/clean_data/ai'

SAVE_DIR = 'data/features_lfcc'
os.makedirs(SAVE_DIR, exist_ok=True)

def extract_lfcc_features(file_path):
    try:
        # Đọc âm thanh chuẩn hóa 16kHz
        y, sr = librosa.load(file_path, sr=16000)
        # Trích xuất 40 hệ số LFCC, dùng 128 bộ lọc (như bạn từng làm)
        lfccs = lfcc(sig=y, fs=sr, num_ceps=40, nfilts=128)
        # Lấy trung bình theo thời gian để nén thành 1 vector 40 chiều
        return np.mean(lfccs, axis=0)
    except Exception as e:
        return None

def process_data():
    X, y = [],[]
    
    # 1. Trích xuất nhóm NGƯỜI THẬT (Nhãn 0)
    print("--- ĐANG TRÍCH XUẤT LFCC: GIỌNG NGƯỜI THẬT ---")
    real_files =[f for f in os.listdir(DATA_REAL) if f.endswith('.wav')]
    for filename in tqdm(real_files):
        path = os.path.join(DATA_REAL, filename)
        feat = extract_lfcc_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)

    # 2. Trích xuất nhóm GIỌNG AI (Nhãn 1)
    print("\n--- ĐANG TRÍCH XUẤT LFCC: GIỌNG AI ---")
    ai_files = [f for f in os.listdir(DATA_AI) if f.endswith('.wav')]
    for filename in tqdm(ai_files):
        path = os.path.join(DATA_AI, filename)
        feat = extract_lfcc_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)

    # 3. Lưu mảng dữ liệu Numpy
    X = np.array(X)
    y = np.array(y)
    np.save(os.path.join(SAVE_DIR, 'X_lfcc.npy'), X)
    np.save(os.path.join(SAVE_DIR, 'y_lfcc.npy'), y)
    
    print(f"\n✅ HOÀN THÀNH! Tổng số tệp đã xử lý: {len(X)}")
    print(f"Kích thước ma trận đặc trưng: {X.shape} (Số file, 40 LFCC)")

if __name__ == "__main__":
    process_data()