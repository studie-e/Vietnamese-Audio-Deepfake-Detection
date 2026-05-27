"""
scripts_train.py
================
Script tổng hợp huấn luyện tất cả các mô hình phân loại.

Yêu cầu trước:
    - Đã chạy scripts_feature_extract.py (có đủ các file features_*.npy)

Thứ tự thực hiện:
    1. train_lfcc_svm.py      — SVM trên LFCC              → svm_lfcc_model.pkl
    2. train_svm.py           — SVM trên MFCC               → svm_voice_model.pkl
    3. train_mlp.py           — MLP trên MFCC               → best_mlp.pkl
    4. train_xgboost.py       — XGBoost trên MFCC-Delta     → best_xgboost.pkl
    5. train_wav2vec.py       — MLP trên Wav2Vec2            → mlp_wav2vec_model.pkl
                                (bỏ qua nếu chưa trích xuất Wav2Vec2)
    6. train_tone_svm.py      — SVM trên Tone-Aware (24d)   → svm_tone_model.pkl
    7. train_tone_xgboost.py  — XGBoost trên Tone-Aware     → xgboost_tone_model.pkl
    8. train_tone_fusion.py   — SVM + MFCC+Tone Fusion (64d)→ svm_tone_fusion_model.pkl
                                (bỏ qua nếu chưa trích xuất Tone-Aware features)
    9. train_aasist.py        — AASIST (Deep Learning)      → aasist_best_model.pth

Đánh giá:
    Mỗi mô hình sẽ in kết quả trên 2 tập:
      - [TEST_SEEN]    — nguồn giọng AI đã thấy khi train
      - [TEST_UNSEEN]  — nguồn giọng AI hoàn toàn mới (gtts)

Cách chạy (từ thư mục gốc dự án):
    python vispoofdb/scripts/scripts_train.py

    Bỏ Wav2Vec2 nếu chưa trích xuất:
    python vispoofdb/scripts/scripts_train.py --skip-wav2vec

    Bỏ cả Wav2Vec2 lẫn Tone models:
    python vispoofdb/scripts/scripts_train.py --skip-wav2vec --skip-tone
"""

import subprocess
import sys
import time
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
MODELS_DIR  = BASE_DIR / "vispoofdb" / "models"
PYTHON      = sys.executable

# Đảm bảo UTF-8 encoding cho subprocess
os.environ['PYTHONIOENCODING'] = 'utf-8'

SKIP_WAV2VEC = "--skip-wav2vec" in sys.argv
SKIP_TONE    = "--skip-tone"    in sys.argv

# Kiểm tra Wav2Vec2 features tồn tại để tự động bỏ qua nếu chưa có
wav2vec_features = BASE_DIR / "vispoofdb" / "data" / "features_wav2vec" / "X_wav2vec.npy"
if not wav2vec_features.exists() and not SKIP_WAV2VEC:
    print("[INFO] Chưa tìm thấy features_wav2vec/X_wav2vec.npy — tự động bỏ qua train_wav2vec.py")
    print("       (Chạy scripts_feature_extract.py trước để có Wav2Vec2 features)\n")
    SKIP_WAV2VEC = True

# Kiểm tra Tone-Aware features tồn tại để tự động bỏ qua nếu chưa có
tone_features = BASE_DIR / "vispoofdb" / "data" / "features_model" / "tone" / "X_tone.npy"
if not tone_features.exists() and not SKIP_TONE:
    print("[INFO] Chưa tìm thấy features_model/tone/X_tone.npy — tự động bỏ qua các Tone models")
    print("       (Chạy: python vispoofdb/data_model/tone_features.py trước)\n")
    SKIP_TONE = True

PIPELINE = [
    (
        MODELS_DIR / "train_lfcc_svm.py",
        "Bước 1/9 — SVM + LFCC",
        False,
    ),
    (
        MODELS_DIR / "train_svm.py",
        "Bước 2/9 — SVM + MFCC",
        False,
    ),
    (
        MODELS_DIR / "train_mlp.py",
        "Bước 3/9 — MLP + MFCC",
        False,
    ),
    (
        MODELS_DIR / "train_xgboost.py",
        "Bước 4/9 — XGBoost + MFCC-Delta",
        False,
    ),
    (
        MODELS_DIR / "train_wav2vec.py",
        "Bước 5/9 — MLP + Wav2Vec2 (768 chiều)",
        SKIP_WAV2VEC,
    ),
    (
        MODELS_DIR / "train_tone_svm.py",
        "Bước 6/9 — SVM + Tone-Aware (24 chiều)",
        SKIP_TONE,
    ),
    (
        MODELS_DIR / "train_tone_xgboost.py",
        "Bước 7/9 — XGBoost + Tone-Aware (24 chiều)",
        SKIP_TONE,
    ),
    (
        MODELS_DIR / "train_tone_fusion.py",
        "Bước 8/9 — SVM + MFCC+Tone Fusion (64 chiều)",
        SKIP_TONE,
    ),
    (
        MODELS_DIR / "train_aasist.py",
        "Bước 9/9 — AASIST (Deep Learning Model)",
        False,
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
    # Prepare environment with UTF-8 encoding for subprocess
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([PYTHON, str(script_path)], cwd=str(BASE_DIR), text=True, env=env)
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
    if SKIP_TONE:
        print("  [skip] train_tone_*.py  — Tone-Aware features chưa có")
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
        (
            BASE_DIR / "vispoofdb" / "data" / "clean_data" / "metadata.csv",
            "clean_data/metadata.csv (cần cho AASIST)",
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
