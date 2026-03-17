import os
import librosa
import soundfile as sf
import warnings
from tqdm import tqdm

# Tắt các cảnh báo lặt vặt của librosa khi đọc mp3
warnings.filterwarnings('ignore')

# --- CẤU HÌNH THÔNG SỐ ---
INPUT_DIR = 'raw_data'      # Thư mục chứa 1200 file lộn xộn ban đầu
OUTPUT_DIR = 'clean_data'   # Thư mục chứa file đã chuẩn hóa
TARGET_SR = 16000           # Tần số lấy mẫu (Yêu cầu 2: 16000 Hz)
TARGET_DURATION = 5         # Độ dài chuẩn (Yêu cầu 4: 5 giây)
TARGET_SAMPLES = TARGET_DURATION * TARGET_SR # Tổng số mẫu cho 5s (5 * 16000 = 80000 mẫu)

# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Lấy danh sách tất cả các file trong thư mục input
all_files = os.listdir(INPUT_DIR)

print(f"Bắt đầu xử lý {len(all_files)} files...")

# Dùng tqdm để tạo thanh tiến trình
for filename in tqdm(all_files, desc="Đang chuẩn hóa"):
    input_path = os.path.join(INPUT_DIR, filename)
    
    # Chỉ xử lý các file âm thanh (bỏ qua file ẩn như .DS_Store)
    if not os.path.isfile(input_path) or filename.startswith('.'):
        continue

    try:
        # YÊU CẦU 1 & 2: Load file và Resample về 16000 Hz. 
        # Hàm load của librosa tự động đọc mọi định dạng (mp3, m4a) và ép về SR mong muốn.
        y, sr = librosa.load(input_path, sr=TARGET_SR)

        # YÊU CẦU 3: Cắt khoảng lặng (Trim Silence) ở đầu và cuối.
        # top_db=20 nghĩa là coi bất cứ âm thanh nào nhỏ hơn mức max 20dB là khoảng lặng và cắt bỏ.
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # YÊU CẦU 4: Cắt/Đệm độ dài (Padding/Truncating) về đúng 5 giây
        # Hàm fix_length sẽ tự động cắt phần thừa nếu > 5s, hoặc đệm số 0 (im lặng) vào cuối nếu < 5s
        y_final = librosa.util.fix_length(y_trimmed, size=TARGET_SAMPLES)

        # Tạo tên file output mới: đổi đuôi thành .wav
        # Ví dụ: "giong_noi_1.mp3" -> "giong_noi_1.wav"
        name_without_ext = os.path.splitext(filename)[0]
        output_filename = f"{name_without_ext}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Lưu file mới ra thư mục clean_data
        sf.write(output_path, y_final, TARGET_SR)

    except Exception as e:
        print(f"\n Lỗi không thể xử lý file {filename}: {str(e)}")

print("\n🎉 HOÀN THÀNH! Toàn bộ dữ liệu đã được lưu trong thư mục 'clean_data'")