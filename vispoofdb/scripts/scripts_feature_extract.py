"""
scripts_feature_extract.py
==========================
Script tổng hợp trích xuất đặc trưng (features) cho từng mô hình.

Yêu cầu trước:
    - Đã chạy scripts_data_process.py (có clean_data/ và metadata.csv)

Thứ tự thực hiện:
    1. lfcc_svm.py          — LFCC 40 chiều          → features_lfcc/
    2. svm_features.py      — MFCC mean 40 chiều      → features_model/svm/
    3. mlp_features.py      — MFCC mean 40 chiều      → features_model/MLP/
    4. xgboost_features.py  — MFCC + Delta 480 chiều  → features_model/xgb/
                              (phụ thuộc vidb_extract_mfcc.py đã chạy)
    5. wav2vec2.py          — Wav2Vec2 768 chiều       → features_wav2vec/
                              (mất ~1 giờ trên CPU)

Cách chạy (từ thư mục gốc dự án):
    python vispoofdb/scripts/scripts_feature_extract.py

    Bỏ Wav2Vec2 nếu không cần (tiết kiệm thời gian):
    python vispoofdb/scripts/scripts_feature_extract.py --skip-wav2vec
"""

import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[2]
DATA_MODEL_DIR = BASE_DIR / "vispoofdb" / "data_model"
PYTHON         = sys.executable

SKIP_WAV2VEC = "--skip-wav2vec" in sys.argv

PIPELINE = [
    (
        DATA_MODEL_DIR / "lfcc_svm.py",
        "Bước 1/5 — Trích xuất LFCC (40 chiều) → features_lfcc/",
        False,  # skip?
    ),
    (
        DATA_MODEL_DIR / "svm_features.py",
        "Bước 2/5 — Trích xuất MFCC mean (40 chiều, SVM) → features_model/svm/",
        False,
    ),
    (
        DATA_MODEL_DIR / "mlp_features.py",
        "Bước 3/5 — Trích xuất MFCC mean (40 chiều, MLP) → features_model/MLP/",
        False,
    ),
    (
        DATA_MODEL_DIR / "xgboost_features.py",
        "Bước 4/5 — Trích xuất MFCC + Delta (480 chiều, XGBoost) → features_model/xgb/",
        False,
    ),
    (
        DATA_MODEL_DIR / "wav2vec2.py",
        "Bước 5/5 — Trích xuất Wav2Vec2 (768 chiều) → features_wav2vec/ [~1 GIỜ]",
        SKIP_WAV2VEC,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def separator(char="=", width=65):
    print(char * width)

def run_step(script_path: Path, description: str) -> bool:
    separator()
    print(f"\n{description}")
    print(f"Script: {script_path.relative_to(BASE_DIR)}\n")
    separator("-")

    start  = time.time()
    result = subprocess.run([PYTHON, str(script_path)], cwd=str(BASE_DIR), text=True)
    elapsed = time.time() - start

    separator("-")
    if result.returncode == 0:
        print(f"Hoàn thành trong {elapsed:.1f}s ({elapsed/60:.1f} phút)\n")
        return True
    else:
        print(f"LỖI (exit code {result.returncode}) sau {elapsed:.1f}s")
        print("    Kiểm tra output bên trên để biết chi tiết lỗi.")
        print("    Pipeline bị dừng lại.\n")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    separator()
    print("  VISPOOFDB — FEATURE EXTRACTION PIPELINE")
    if SKIP_WAV2VEC:
        print("  [--skip-wav2vec] Bỏ qua Wav2Vec2 extraction")
    separator()
    print()

    # Kiểm tra metadata.csv tồn tại
    metadata = BASE_DIR / "vispoofdb" / "data" / "clean_data" / "metadata.csv"
    if not metadata.exists():
        print(f"[ERROR] Không tìm thấy metadata.csv tại {metadata}")
        print("        Hãy chạy scripts_data_process.py trước!")
        sys.exit(1)

    # Kiểm tra splits_mfcc.npy (cần cho xgboost_features.py)
    splits_mfcc = BASE_DIR / "vispoofdb" / "data" / "features_mfcc" / "splits_mfcc.npy"
    if not splits_mfcc.exists():
        print(f"[WARN] Chưa có splits_mfcc.npy — vidb_extract_mfcc.py sẽ cần chạy trước bước XGBoost.")
        print("        Nhưng scripts_data_process.py đã bao gồm bước này rồi.")
        print()

    total_start = time.time()
    completed   = 0
    skipped     = 0

    for script_path, description, skip in PIPELINE:
        if skip:
            separator()
            print(f"\n[BỎ QUA] {description}")
            print()
            skipped += 1
            continue

        if not script_path.exists():
            print(f"[WARN] Không tìm thấy file: {script_path}, bỏ qua.")
            continue

        success = run_step(script_path, description)
        if not success:
            sys.exit(1)
        completed += 1

    total_elapsed = time.time() - total_start
    separator()
    print(f"\nHOAN THANH! ({completed} bước thành công, {skipped} bị bỏ qua)")
    print(f"    Tổng thời gian: {total_elapsed/60:.1f} phút")
    print()
    print("    Bước tiếp theo:")
    print("      Chạy vispoofdb/scripts/scripts_train.py để huấn luyện các mô hình.")
    separator()


if __name__ == "__main__":
    main()
