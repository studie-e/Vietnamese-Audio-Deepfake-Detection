import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings
warnings.filterwarnings('ignore')

# --- 1. Cấu hình thư mục ---
DATA_REAL = 'data/clean_data/real'
DATA_AI = 'data/clean_data/ai'
SAVE_DIR = 'data/features_wav2vec'
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. Tải Mô hình Wav2Vec2 từ Facebook ---
print("Đang tải 'não bộ' Wav2Vec2 (Lần đầu sẽ mất chút thời gian tải ~360MB)...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Tự động đẩy mô hình lên Card đồ họa (GPU) nếu có để chạy nhanh hơn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Đã tải xong! Đang chạy trên thiết bị: {device.type.upper()}")

# --- 3. Hàm trích xuất ---
def extract_wav2vec_features(file_path):
    try:
        # Load audio chuẩn 16kHz
        y, sr = librosa.load(file_path, sr=16000)
        
        # Tiền xử lý âm thanh cho AI hiểu
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(device)

        # Chạy qua Mạng Nơ-ron (Tắt gradient để không bị tràn RAM)
        with torch.no_grad():
            outputs = model(**inputs)
            # Lấy layer sâu nhất, tính trung bình theo thời gian -> ra 768 con số
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        return features
    except Exception as e:
        print(f"Lỗi file {file_path}: {e}")
        return None

# --- 4. Quét toàn bộ Dữ liệu ---
def process_data():
    X, y = [],[]
    
    # Nhóm Người Thật (Nhãn 0)
    print("\n--- ĐANG TRÍCH XUẤT VÉC-TƠ 768 CHIỀU: GIỌNG NGƯỜI THẬT ---")
    real_files =[f for f in os.listdir(DATA_REAL) if f.endswith('.wav')]
    for filename in tqdm(real_files):
        path = os.path.join(DATA_REAL, filename)
        feat = extract_wav2vec_features(path)
        if feat is not None:
            X.append(feat)
            y.append(0)

    # Nhóm Giọng AI (Nhãn 1)
    print("\n--- ĐANG TRÍCH XUẤT VÉC-TƠ 768 CHIỀU: GIỌNG AI ---")
    ai_files =[f for f in os.listdir(DATA_AI) if f.endswith('.wav')]
    for filename in tqdm(ai_files):
        path = os.path.join(DATA_AI, filename)
        feat = extract_wav2vec_features(path)
        if feat is not None:
            X.append(feat)
            y.append(1)

    # Lưu kết quả
    np.save(os.path.join(SAVE_DIR, 'X_wav2vec.npy'), np.array(X))
    np.save(os.path.join(SAVE_DIR, 'y_wav2vec.npy'), np.array(y))
    print(f"\n✅ HOÀN THÀNH! Đã lưu ma trận {len(X)} x 768 vào thư mục {SAVE_DIR}/")

if __name__ == "__main__":
    process_data()