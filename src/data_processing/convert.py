import os
import shutil
from pathlib import Path
import librosa
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_AI = BASE_DIR / "data/raw/ai"
RAW_REAL = BASE_DIR / "data/raw/real"

PROCESS_AI = BASE_DIR / "data/process/ai"
PROCESS_REAL = BASE_DIR / "data/process/real"


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file)

        # MP3 -> WAV
        if file.endswith(".mp3"):
            wav_name = os.path.splitext(file)[0] + ".wav"
            wav_path = os.path.join(output_folder, wav_name)

            try:
                audio, sr = librosa.load(input_path, sr=16000, mono=True)
                sf.write(wav_path, audio, sr)

                print(f"Converted: {file} -> {wav_name}")

            except Exception:
                print(f"Skip corrupted file: {file}")

        # WAV -> copy
        elif file.endswith(".wav"):
            output_path = os.path.join(output_folder, file)

            try:
                shutil.copy2(input_path, output_path)
                print(f"Copied wav: {file}")

            except Exception:
                print(f"Skip corrupted wav: {file}")


def main():
    print("Processing AI audio...")
    convert_folder(RAW_AI, PROCESS_AI)

    print("Processing REAL audio...")
    convert_folder(RAW_REAL, PROCESS_REAL)

    print("Done!")


if __name__ == "__main__":
    main()