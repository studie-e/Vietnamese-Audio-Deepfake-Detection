"""
File lấy dữ liệu từ VIVOS (7k files) với tên gốc, không đổi tên.
"""

import os
import sys
from pathlib import Path
import shutil
from collections import defaultdict
import pandas as pd
import librosa
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VIVOS_DIR = SCRIPT_DIR.parent.parent / "vivos"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "raw" / "real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILES = 7000  # Lấy 7k files

print(f"[*] VIVOS All Files Extractor (7k limit)")
print(f"    Output: {OUTPUT_DIR}")
print(f"    Target: {TARGET_FILES} files")
print(f"    Mode: Keep original filenames")

# ─────────────────────────────────────────────────────────────────────────────
# Scan VIVOS structure
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Scanning VIVOS folders...")

speaker_files = defaultdict(list)

for split in ["train", "test"]:
    split_dir = VIVOS_DIR / split / "waves"
    if not split_dir.exists():
        print(f"    [Warn] {split_dir} not found")
        continue
    
    # Train: VIVOSSPK*, Test: VIVOSDEV*
    pattern = "VIVOSSPK*" if split == "train" else "VIVOSDEV*"
    speaker_dirs = sorted(split_dir.glob(pattern))
    print(f"    {split}: {len(speaker_dirs)} speakers")
    
    for spk_dir in speaker_dirs:
        spk_name = spk_dir.name
        wav_files = list(spk_dir.glob("*.wav"))
        speaker_files[spk_name].extend(wav_files)

total_speakers = len(speaker_files)
total_available = sum(len(files) for files in speaker_files.values())

print(f"\n[✓] Found {total_speakers} speakers, {total_available} files total")

# ─────────────────────────────────────────────────────────────────────────────
# Collect all files (keep original names with speaker prefix)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Preparing files (keep original names, limit to {TARGET_FILES})...")

file_mapping = []  # (src_path, dst_name, speaker)

for spk_name in sorted(speaker_files.keys()):
    files = speaker_files[spk_name]
    for src_file in files:
        # Tên mới: speaker_name_original_name.wav
        dst_name = f"{spk_name}_{src_file.name}"
        file_mapping.append((src_file, dst_name, spk_name))

# Random sample if > TARGET_FILES
if len(file_mapping) > TARGET_FILES:
    import random
    random.seed(42)
    file_mapping = random.sample(file_mapping, TARGET_FILES)
    print(f"    Sampled {TARGET_FILES} from {len(file_mapping) + (len(file_mapping) if len(file_mapping) > TARGET_FILES else 0)} available files")

print(f"    Total files to copy: {len(file_mapping)}")

# ─────────────────────────────────────────────────────────────────────────────
# Copy files với original names
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Copying files to {OUTPUT_DIR}...")

success = 0
failed = 0
failed_files = []

for src_file, dst_name, spk_name in tqdm(file_mapping, desc="Copy progress"):
    try:
        dst_file = OUTPUT_DIR / dst_name
        shutil.copy2(src_file, dst_file)
        success += 1
    except Exception as e:
        failed += 1
        failed_files.append((dst_name, str(e)))
        if failed <= 5:
            print(f"    [Warn] {dst_name}: {e}")

print(f"\n[✓] Copy hoan thanh!")
print(f"    Success: {success}")
print(f"    Failed: {failed}")
