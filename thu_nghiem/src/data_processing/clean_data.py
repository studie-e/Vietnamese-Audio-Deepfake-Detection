import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_BASE = BASE_DIR / "data" / "raw"
OUTPUT_BASE = BASE_DIR / "data" / "clean_data"
CSV_OUTPUT = BASE_DIR / "data" / "metadata.csv" # Nơi lưu file CSV

TARGET_SR = 16000
DURATION = 5 
TARGET_SAMPLES = TARGET_SR * DURATION
CLASSES = {'real': 0, 'ai': 1}


def process_and_create_csv():
    print(" Đang làm sạch dữ liệu và lập chỉ mục CSV...")
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    metadata = []
    id_counter = 1

    for folder, label_num in CLASSES.items():
        input_dir = INPUT_BASE / folder
        output_dir = OUTPUT_BASE / folder
        os.makedirs(output_dir, exist_ok=True)
        
        if not input_dir.exists(): continue
            
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3'))]
        
        for filename in tqdm(files, desc=f"Đang xử lý {folder}"):
            path_in = input_dir / filename
            path_out = output_dir / f"{os.path.splitext(filename)[0]}.wav"

            try:
                # 1. Lấy thông tin độ dài gốc trước khi xử lý
                duration = librosa.get_duration(path=path_in)
                
                # 2. Xử lý âm thanh
                y, sr = librosa.load(path_in, sr=TARGET_SR, mono=True)
                y_trim, _ = librosa.effects.trim(y, top_db=20)
                y_final = librosa.util.fix_length(y_trim, size=TARGET_SAMPLES)
                
                # 3. Lưu file sạch
                sf.write(path_out, y_final, TARGET_SR)

                # 4. Thu thập dữ liệu cho CSV
                metadata.append({
                    'ID': id_counter,
                    'Tên_File': path_out.name,
                    'Nhãn_Số': label_num,
                    'Phân_Loại': 'Thật' if folder == 'real' else 'Giả mạo',
                    'Độ_Dài (giây)': round(duration, 2)
                })
                id_counter += 1
                
            except Exception:
                continue

    # --- XUẤT FILE CSV ---
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
    
    print(f"\n HOÀN THÀNH!")
    print(f"Đã xử lý: {len(df)} files.")
    print(f"File âm thanh sạch: {OUTPUT_BASE}")
    print(f"File CSV quản lý: {CSV_OUTPUT}")

if __name__ == "__main__":
    process_and_create_csv()