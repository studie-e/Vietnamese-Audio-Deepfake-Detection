#chia train/val/test

import os
import random
import shutil
from pathlib import Path

# =========================
# CONFIG
# =========================

random.seed(42)

REAL_SOURCE = "dataset/real_new"
FAKE_SOURCE = "dataset/fake"

OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# =========================
# CREATE FOLDERS
# =========================

splits = ["train", "val", "test"]
classes = ["bonafide", "spoof"]

for split in splits:
    for cls in classes:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, cls),
            exist_ok=True
        )


# =========================
# SPLIT FUNCTION
# =========================

def split_files(file_list):

    random.shuffle(file_list)

    total = len(file_list)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]

    return train_files, val_files, test_files


# =========================
# COPY FUNCTION
# =========================

def copy_files(files, split, cls_name):

    for file_path in files:

        destination = os.path.join(
            OUTPUT_DIR,
            split,
            cls_name,
            Path(file_path).name
        )

        shutil.copy(file_path, destination)


# =========================
# LOAD FILES
# =========================

real_files = list(Path(REAL_SOURCE).glob("*.wav"))
fake_files = list(Path(FAKE_SOURCE).glob("*.wav"))

print("Real files:", len(real_files))
print("Fake files:", len(fake_files))


# =========================
# SPLIT REAL
# =========================

real_train, real_val, real_test = split_files(real_files)

copy_files(real_train, "train", "bonafide")
copy_files(real_val, "val", "bonafide")
copy_files(real_test, "test", "bonafide")


# =========================
# SPLIT FAKE
# =========================

fake_train, fake_val, fake_test = split_files(fake_files)

copy_files(fake_train, "train", "spoof")
copy_files(fake_val, "val", "spoof")
copy_files(fake_test, "test", "spoof")


# =========================
# SUMMARY
# =========================

print("\nDONE!\n")

print("TRAIN")
print("  bonafide:", len(real_train))
print("  spoof:", len(fake_train))

print("\nVAL")
print("  bonafide:", len(real_val))
print("  spoof:", len(fake_val))

print("\nTEST")
print("  bonafide:", len(real_test))
print("  spoof:", len(fake_test))