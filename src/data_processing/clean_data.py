import os
import librosa
import soundfile as sf
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN CHUẨN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_BASE = os.path.join(BASE_DIR, 'data', 'process') 
OUTPUT_BASE = os.path.join(BASE_DIR, 'data', 'clean_data')

TARGET_SR = 16000
TARGET_SAMPLES = 16000 * 5 # 5 giây

# Cấu trúc thư mục chuẩn của nhóm
CLASSES = ['ai', 'real'] 

print("🚀 Bắt đầu làm sạch dữ liệu (chuẩn hóa 16kHz, 5s)...")

for folder in CLASSES:
    input_dir = os.path.join(INPUT_BASE, folder)
    output_dir = os.path.join(OUTPUT_BASE, folder)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(input_dir):
        print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {input_dir}")
        continue
        
    files =[f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for filename in tqdm(files, desc=f"Đang xử lý thư mục {folder}"):
        path_in = os.path.join(input_dir, filename)
        path_out = os.path.join(output_dir, filename)
        try:
            y, sr = librosa.load(path_in, sr=TARGET_SR)
            y_trim, _ = librosa.effects.trim(y, top_db=20)
            y_final = librosa.util.fix_length(y_trim, size=TARGET_SAMPLES)
            sf.write(path_out, y_final, TARGET_SR)
        except Exception as e:
            pass

print(f"✅ HOÀN THÀNH! Dữ liệu đã sạch, lưu tại: {OUTPUT_BASE}")