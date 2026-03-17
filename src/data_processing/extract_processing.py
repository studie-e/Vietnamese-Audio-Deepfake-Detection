import os
import librosa
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# (Giả sử bạn chạy file này từ thư mục gốc của project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DATA_DIR = os.path.join(BASE_DIR, 'data', 'clean_data')
FEATURE_DIR = os.path.join(BASE_DIR, 'data', 'features_mel')

# Thông số trích xuất Mel-Spectrogram
TARGET_SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Hai nhãn phân loại của chúng ta
CLASSES = ['real', 'fake']

def extract_mel_spectrogram(input_wav_path, output_npy_path):
    """Hàm xử lý cho 1 file âm thanh"""
    try:
        y, sr = librosa.load(input_wav_path, sr=TARGET_SR)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        np.save(output_npy_path, log_mel_spectrogram)
        return True
    except Exception as e:
        print(f"\n[LỖI] Không thể xử lý file {input_wav_path}: {e}")
        return False

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("Bắt đầu trích xuất đặc trưng Mel-Spectrogram...")
    
    # Lặp qua từng class (đầu tiên là 'real', xong đến 'fake')
    for class_name in CLASSES:
        input_class_dir = os.path.join(CLEAN_DATA_DIR, class_name)
        output_class_dir = os.path.join(FEATURE_DIR, class_name)
        
        # Nếu thư mục output chưa có thì tự động tạo
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            
        # Kiểm tra xem thư mục input có tồn tại không
        if not os.path.exists(input_class_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục {input_class_dir}. Bỏ qua...")
            continue
            
        # Lấy danh sách file .wav trong thư mục
        wav_files = [f for f in os.listdir(input_class_dir) if f.endswith('.wav')]
        
        print(f"\nĐang xử lý thư mục '{class_name}' ({len(wav_files)} files):")
        
        # Chạy và hiển thị thanh tiến trình
        for filename in tqdm(wav_files):
            input_path = os.path.join(input_class_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_class_dir, output_filename)
            
            # Chỉ trích xuất nếu file .npy chưa tồn tại (tránh chạy lại từ đầu nếu bị ngắt quãng)
            if not os.path.exists(output_path):
                extract_mel_spectrogram(input_path, output_path)
                
    print("\n🎉 HOÀN THÀNH TẤT CẢ! Dữ liệu đã sẵn sàng để train AI.")