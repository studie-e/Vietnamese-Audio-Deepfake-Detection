import torch
import librosa
import torch.nn.functional as F
from models.baseline import Full_AASIST_Model
import warnings
import random
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_LENGTH = 80000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(file_path):
    """Hàm tải và cắt/ghép audio giống hệt lúc Train"""
    waveform, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    
    if waveform.shape[0] < TARGET_LENGTH:
        pad_size = TARGET_LENGTH - waveform.shape[0]
        waveform = F.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:TARGET_LENGTH]
        
    # Thêm chiều batch (batch_size=1)
    waveform = waveform.unsqueeze(0) 
    return waveform

def predict(audio_path, model_path="aasist_best_model.pth"):
    print(f"Đang phân tích file: {audio_path}")
    
    # 1. Khởi tạo model và tải trọng số (weights)
    model = Full_AASIST_Model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Chuyển sang chế độ test
    
    # 2. Xử lý âm thanh đầu vào
    audio_tensor = load_audio(audio_path).to(DEVICE)
    
    # 3. Dự đoán
    with torch.no_grad():
        output = model(audio_tensor)
        
        # Dùng Softmax để chuyển đổi đầu ra thành tỷ lệ phần trăm (0 đến 1)
        probabilities = F.softmax(output, dim=1).squeeze()
        
        # Lấy phần trăm của 2 class (0: Thật, 1: Giả)
        prob_real = probabilities[0].item() * 100
        prob_fake = probabilities[1].item() * 100
        
        # Quyết định
        if prob_fake > prob_real:
            print(f"=> KẾT QUẢ: GIỌNG TRÍ TUỆ NHÂN TẠO (AI) - Độ tin cậy: {prob_fake:.2f}%")
        else:
            print(f"=> KẾT QUẢ: GIỌNG NGƯỜI THẬT - Độ tin cậy: {prob_real:.2f}%")
            
    print("-" * 50)

if __name__ == "__main__":
    print("="*50)
    print("HỆ THỐNG KIỂM TRA GIỌNG NÓI AI (AASIST)")
    print("="*50)

    # Đường dẫn metadata từ AASIST/predict.py
    metadata_path = Path(__file__).parent.parent / "vispoofdb" / "data" / "clean_data" / "metadata.csv"
    
    if not metadata_path.exists():
        print(f"Lỗi: Không tìm thấy metadata: {metadata_path}")
        exit(1)
    
    # Đọc metadata và lọc test_unseen
    df = pd.read_csv(metadata_path)
    test_unseen_df = df[df["split"] == "test_unseen"]
    
    base_dir = metadata_path.parent
    
    # Tách real và fake từ test_unseen
    real_files = []
    fake_files = []
    
    for _, row in test_unseen_df.iterrows():
        file_path = base_dir / row["file_path"]
        if file_path.exists():
            if row["label"] == "real":
                real_files.append(str(file_path))
            else:
                fake_files.append(str(file_path))
    
    # Kiểm tra xem có file không
    if len(real_files) == 0 or len(fake_files) == 0:
        print("Lỗi: Không tìm thấy file âm thanh nào trong test_unseen!")
    else:
        # CHỌN NGẪU NHIÊN mỗi bên 1 file từ test_unseen
        random_real = random.choice(real_files)
        random_fake = random.choice(fake_files)
        
        # In ra tên file để bạn dễ theo dõi
        print(f"Đang bốc thăm ngẫu nhiên từ test_unseen...")
        print(f"File Thật được chọn: {Path(random_real).name}")
        print(f"File Giả được chọn:  {Path(random_fake).name}\n")
        
        try:
            # Tiến hành dự đoán
            predict(random_real)
            predict(random_fake)
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")