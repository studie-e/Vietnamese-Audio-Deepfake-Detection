"""
scripts_train.py
================
Script tổng hợp huấn luyện tất cả các mô hình phân loại.

Yêu cầu trước:
    - Đã chạy scripts_feature_extract.py (có đủ các file features_*.npy)

Thứ tự thực hiện:
    1. train_lfcc_svm.py   — SVM trên LFCC         → svm_lfcc_model.pkl
    2. train_svm.py        — SVM trên MFCC          → svm_voice_model.pkl
    3. train_mlp.py        — MLP trên MFCC          → best_mlp.pkl
    4. train_xgboost.py    — XGBoost                → best_xgboost.pkl
    5. train_wav2vec.py    — MLP trên Wav2Vec2       → mlp_wav2vec_model.pkl
                             (bỏ qua nếu chưa trích xuất Wav2Vec2)

Đánh giá:
    Mỗi mô hình sẽ in kết quả trên 2 tập:
      - [TEST_SEEN]    — nguồn giọng AI đã thấy khi train
      - [TEST_UNSEEN]  — nguồn giọng AI hoàn toàn mới (gtts)

Cách chạy (từ thư mục gốc dự án):
    python vispoofdb/scripts/scripts_train.py

    Bỏ Wav2Vec2 nếu chưa trích xuất:
    python vispoofdb/scripts/scripts_train.py --skip-wav2vec
"""

import subprocess
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
MODELS_DIR  = BASE_DIR / "vispoofdb" / "models"
PYTHON      = sys.executable

SKIP_WAV2VEC = "--skip-wav2vec" in sys.argv

# Kiểm tra Wav2Vec2 features tồn tại để tự động bỏ qua nếu chưa có
wav2vec_features = BASE_DIR / "vispoofdb" / "data" / "features_wav2vec" / "X_wav2vec.npy"
if not wav2vec_features.exists() and not SKIP_WAV2VEC:
    print("[INFO] Chưa tìm thấy features_wav2vec/X_wav2vec.npy — tự động bỏ qua train_wav2vec.py")
    print("       (Chạy scripts_feature_extract.py trước để có Wav2Vec2 features)\n")
    SKIP_WAV2VEC = True

PIPELINE = [
    (
        MODELS_DIR / "train_lfcc_svm.py",
        "Bước 1/5 — SVM + LFCC",
        False,
    ),
    (
        MODELS_DIR / "train_svm.py",
        "Bước 2/5 — SVM + MFCC",
        False,
    ),
    (
        MODELS_DIR / "train_mlp.py",
        "Bước 3/5 — MLP + MFCC",
        False,
    ),
    (
        MODELS_DIR / "train_xgboost.py",
        "Bước 4/5 — XGBoost + MFCC-Delta",
        False,
    ),
    (
        MODELS_DIR / "train_wav2vec.py",
        "Bước 5/5 — MLP + Wav2Vec2 (768 chiều)",
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
    print("  VISPOOFDB — MODEL TRAINING PIPELINE")
    if SKIP_WAV2VEC:
        print("  [skip] train_wav2vec.py — Wav2Vec2 features chưa có")
    separator()
    print()

    # Kiểm tra feature files cơ bản
    required_checks = [
        (
            BASE_DIR / "vispoofdb" / "data" / "features_lfcc" / "splits_lfcc.npy",
            "features_lfcc/splits_lfcc.npy (cần lfcc_svm.py)",
        ),
        (
            BASE_DIR / "vispoofdb" / "data" / "features_model" / "svm" / "splits_svm.npy",
            "features_model/svm/splits_svm.npy (cần svm_features.py)",
        ),
        (
            BASE_DIR / "vispoofdb" / "data" / "features_model" / "MLP" / "splits_mlp.npy",
            "features_model/MLP/splits_mlp.npy (cần mlp_features.py)",
        ),
        (
            BASE_DIR / "vispoofdb" / "data" / "features_model" / "xgb" / "splits_xgb.npy",
            "features_model/xgb/splits_xgb.npy (cần xgboost_features.py)",
        ),
    ]

    missing = [label for path, label in required_checks if not path.exists()]
    if missing:
        print("[ERROR] Thiếu các file feature sau:")
        for m in missing:
            print(f"    - {m}")
        print("\n    Hãy chạy scripts_feature_extract.py trước!")
        sys.exit(1)

    total_start = time.time()
    completed   = 0
    skipped     = 0
    results     = []

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

        results.append(description)
        completed += 1

    total_elapsed = time.time() - total_start
    separator()
    print(f"\nHOAN THANH! ({completed} mô hình đã huấn luyện, {skipped} bị bỏ qua)")
    print(f"    Tổng thời gian: {total_elapsed/60:.1f} phút")
    print()
    print("    Các mô hình đã lưu tại: vispoofdb/models_saved/")
    separator()


if __name__ == "__main__":
    main()
