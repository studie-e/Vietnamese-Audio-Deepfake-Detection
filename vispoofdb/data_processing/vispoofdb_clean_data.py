import os
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_BASE = BASE_DIR / "vispoofdb" / "data"
OUTPUT_BASE = BASE_DIR / "vispoofdb" / "data" / "clean_data"

TARGET_SR = 16000
DURATION = 5 
TARGET_SAMPLES = TARGET_SR * DURATION
CLASSES = {'real': 0, 'fake': 1}


def clean_audio_data():
    """
    Làm sạch dữ liệu âm thanh từ thư mục data/raw
    - Chuẩn hóa sample rate: 16kHz
    - Chuẩn hóa độ dài: 5 giây
    - Lưu kết quả vào data/clean_data
    """
    print("🔄 Đang làm sạch dữ liệu từ data/raw...")
    print(f"   Input: {INPUT_BASE}")
    print(f"   Output: {OUTPUT_BASE}\n")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    total_processed = 0
    total_errors = 0

    for folder, label_num in CLASSES.items():
        input_dir = INPUT_BASE / folder
        output_dir = OUTPUT_BASE / folder
        os.makedirs(output_dir, exist_ok=True)
        
        if not input_dir.exists():
            print(f"⚠️  Thư mục không tồn tại: {input_dir}")
            continue
            
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
        
        if not files:
            print(f"⚠️  Thư mục trống: {folder}")
            continue
        
        print(f"📂 {folder.upper()} ({len(files)} files)")
        
        for filename in tqdm(files, desc=f"  Xử lý {folder}"):
            path_in = input_dir / filename
            path_out = output_dir / f"{os.path.splitext(filename)[0]}.wav"

            try:
                # 1. Đọc file âm thanh
                y, sr = librosa.load(str(path_in), sr=TARGET_SR, mono=True)
                
                # 2. Trim silence
                y_trim, _ = librosa.effects.trim(y, top_db=20)
                
                # 3. Chuẩn hóa độ dài (pad hoặc cắt)
                if len(y_trim) < TARGET_SAMPLES:
                    y_final = librosa.util.fix_length(y_trim, size=TARGET_SAMPLES)
                else:
                    y_final = y_trim[:TARGET_SAMPLES]
                
                # 4. Lưu file sạch
                sf.write(path_out, y_final, TARGET_SR)
                total_processed += 1
                
            except Exception as e:
                print(f"    ⚠️  Lỗi: {filename} - {str(e)[:50]}")
                total_errors += 1

    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH!")
    print(f"   Tổng file xử lý thành công: {total_processed}")
    print(f"   Lỗi: {total_errors}")
    print(f"   Output: {OUTPUT_BASE}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    clean_audio_data()