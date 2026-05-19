import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

INPUT_DIR = "raw_real"
OUTPUT_DIR = "dataset/real_new"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 5  # giây
TARGET_LEN = TARGET_SR * TARGET_DURATION


def process_audio(input_path, output_path):
    try:
        # load audio -> tự convert mono
        audio, sr = librosa.load(
            input_path,
            sr=TARGET_SR,
            mono=True
        )

        # chuẩn hóa amplitude
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        # cắt nếu quá dài
        if len(audio) > TARGET_LEN:
            audio = audio[:TARGET_LEN]

        # pad nếu quá ngắn
        elif len(audio) < TARGET_LEN:
            pad_len = TARGET_LEN - len(audio)
            audio = np.pad(audio, (0, pad_len))

        # lưu wav chuẩn
        sf.write(output_path, audio, TARGET_SR)

    except Exception as e:
        print(f"Lỗi {input_path}: {e}")


files = [f for f in os.listdir(INPUT_DIR)]

for file in tqdm(files):
    input_path = os.path.join(INPUT_DIR, file)

    output_name = os.path.splitext(file)[0] + ".wav"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    process_audio(input_path, output_path)

print("Hoàn thành preprocessing!")