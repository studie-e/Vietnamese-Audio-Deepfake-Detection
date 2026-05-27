"""
run_full_pipeline.py
====================
Master script chạy toàn bộ pipeline từ xử lý dữ liệu → trích đặc trưng → huấn luyện mô hình.

Thứ tự thực hiện:
    PHASE 1: Data Processing (4 bước)
        1. vispoofdb_clean_data.py
        2. vispoofdb_generate_metadata.py
        3. vidb_extract_mfcc.py
        4. vidb_extract_processing.py

    PHASE 2: Feature Extraction (6 bước)
        5. lfcc_svm.py
        6. svm_features.py
        7. mlp_features.py
        8. xgboost_features.py
        9. wav2vec2.py (tuỳ chọn)
        10. tone_features.py (tuỳ chọn)

    PHASE 3: Model Training (8 bước)
        11. train_lfcc_svm.py
        12. train_svm.py
        13. train_mlp.py
        14. train_xgboost.py
        15. train_wav2vec.py (tuỳ chọn)
        16. train_tone_svm.py (tuỳ chọn)
        17. train_tone_xgboost.py (tuỳ chọn)
        18. train_tone_fusion.py (tuỳ chọn)

Cách chạy:
    python run_full_pipeline.py                              # Chạy toàn bộ
    python run_full_pipeline.py --skip-wav2vec             # Bỏ Wav2Vec2
    python run_full_pipeline.py --skip-wav2vec --skip-tone # Bỏ Wav2Vec2 và Tone
    python run_full_pipeline.py --phase 1                  # Chỉ Phase 1 (Data Processing)
    python run_full_pipeline.py --phase 2                  # Chỉ Phase 2 (Feature Extraction)
    python run_full_pipeline.py --phase 3                  # Chỉ Phase 3 (Model Training)
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
DATA_PROC_DIR   = BASE_DIR / "vispoofdb" / "data_processing"
DATA_MODEL_DIR  = BASE_DIR / "vispoofdb" / "data_model"
MODELS_DIR      = BASE_DIR / "vispoofdb" / "models"
PYTHON          = sys.executable

# Parse arguments
SKIP_WAV2VEC    = "--skip-wav2vec" in sys.argv
SKIP_TONE       = "--skip-tone" in sys.argv
RUN_PHASE       = None

for arg in sys.argv[1:]:
    if arg.startswith("--phase"):
        try:
            RUN_PHASE = int(arg.split()[1] if " " in arg else sys.argv[sys.argv.index(arg) + 1])
        except (ValueError, IndexError):
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Định nghĩa Pipeline
# ─────────────────────────────────────────────────────────────────────────────

PIPELINES = {
    1: [
        (
            DATA_PROC_DIR / "vispoofdb_clean_data.py",
            "Làm sạch & chuẩn hóa âm thanh thô",
        ),
        (
            DATA_PROC_DIR / "vispoofdb_generate_metadata.py",
            "Tạo metadata.csv (train/test_seen/test_unseen)",
        ),
        (
            DATA_PROC_DIR / "vidb_extract_mfcc.py",
            "Trích xuất MFCC 3D",
        ),
        (
            DATA_PROC_DIR / "vidb_extract_processing.py",
            "Trích xuất Mel-Spectrogram",
        ),
    ],
    2: [
        (
            DATA_MODEL_DIR / "lfcc_svm.py",
            "Trích xuất LFCC (40 chiều)",
            False,
        ),
        (
            DATA_MODEL_DIR / "svm_features.py",
            "Trích xuất MFCC mean cho SVM (40 chiều)",
            False,
        ),
        (
            DATA_MODEL_DIR / "mlp_features.py",
            "Trích xuất MFCC mean cho MLP (40 chiều)",
            False,
        ),
        (
            DATA_MODEL_DIR / "xgboost_features.py",
            "Trích xuất MFCC + Delta cho XGBoost (480 chiều)",
            False,
        ),
        (
            DATA_MODEL_DIR / "wav2vec2.py",
            "Trích xuất Wav2Vec2 (768 chiều) [~1 GIỜ]",
            SKIP_WAV2VEC,
        ),
        (
            DATA_MODEL_DIR / "tone_features.py",
            "Trích xuất Tone-Aware (24 chiều)",
            SKIP_TONE,
        ),
    ],
    3: [
        (
            MODELS_DIR / "train_lfcc_svm.py",
            "SVM + LFCC",
            False,
        ),
        (
            MODELS_DIR / "train_svm.py",
            "SVM + MFCC",
            False,
        ),
        (
            MODELS_DIR / "train_mlp.py",
            "MLP + MFCC",
            False,
        ),
        (
            MODELS_DIR / "train_xgboost.py",
            "XGBoost + MFCC-Delta",
            False,
        ),
        (
            MODELS_DIR / "train_wav2vec.py",
            "MLP + Wav2Vec2 (768 chiều)",
            SKIP_WAV2VEC,
        ),
        (
            MODELS_DIR / "train_tone_svm.py",
            "SVM + Tone-Aware (24 chiều)",
            SKIP_TONE,
        ),
        (
            MODELS_DIR / "train_tone_xgboost.py",
            "XGBoost + Tone-Aware (24 chiều)",
            SKIP_TONE,
        ),
        (
            MODELS_DIR / "train_tone_fusion.py",
            "SVM + Tone + MFCC Fusion (64 chiều)",
            SKIP_TONE,
        ),
    ],
}

PHASE_NAMES = {
    1: "📊 DATA PROCESSING",
    2: "🔍 FEATURE EXTRACTION",
    3: "🤖 MODEL TRAINING",
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def separator(char="=", width=80):
    print(char * width)

def phase_separator(char="━", width=80):
    print(char * width)

def format_time(seconds):
    """Chuyển đổi giây thành định dạng dễ đọc."""
    return str(timedelta(seconds=int(seconds)))

def run_script(script_path: Path, description: str, step_num: int, total_steps: int) -> tuple[bool, float]:
    """
    Chạy một script và trả về (success, elapsed_time).
    """
    print(f"\n[{step_num}/{total_steps}] {description}")
    print(f"     📁 {script_path.relative_to(BASE_DIR)}")
    print()

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(BASE_DIR),
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"     ✅ Thành công trong {format_time(elapsed)}")
        return True, elapsed
    else:
        print(f"     ❌ FAILED (exit code {result.returncode})")
        return False, elapsed

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    separator()
    print("🚀 VIETNAMESE AUDIO DEEPFAKE DETECTION — FULL PIPELINE")
    print(f"📂 Base directory: {BASE_DIR}")
    print(f"🐍 Python: {PYTHON}")
    separator()
    print()

    # Xác định phases cần chạy
    if RUN_PHASE:
        phases_to_run = [RUN_PHASE]
    else:
        phases_to_run = [1, 2, 3]

    # Thống kê chung
    all_results = {}
    total_time = 0
    total_steps = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Chạy từng phase
    # ──────────────────────────────────────────────────────────────────────────
    for phase in phases_to_run:
        if phase not in PIPELINES:
            print(f"⚠️  Phase {phase} không tồn tại. Bỏ qua.\n")
            continue

        phase_start = time.time()
        pipeline = PIPELINES[phase]

        # Header phase
        phase_separator("━")
        print(f"\n{PHASE_NAMES[phase]}")
        phase_separator("━")
        print()

        # Chạy các bước trong phase
        phase_results = []
        phase_time = 0
        global_step = sum(
            len(PIPELINES[p]) for p in phases_to_run if p < phase
        ) + 1

        for step_idx, step_data in enumerate(pipeline, 1):
            script_path = step_data[0]
            description = step_data[1]
            skip = step_data[2] if len(step_data) > 2 else False

            if skip:
                print(f"\n⏭️  [{step_idx}/{len(pipeline)}] {description} (SKIPPED)")
                continue

            success, elapsed = run_script(script_path, description, global_step, 
                                        sum(len(PIPELINES[p]) for p in phases_to_run))
            phase_results.append({
                "step": step_idx,
                "description": description,
                "script": script_path.name,
                "success": success,
                "elapsed": elapsed,
            })
            phase_time += elapsed
            total_time += elapsed

            if not success:
                print(f"\n⚠️  ❌ PIPELINE BỊ DỪNG! Lỗi ở bước {step_idx}/{len(pipeline)}")
                print(f"    Hãy kiểm tra output bên trên để khắc phục.\n")
                return False

            global_step += 1

        # Tóm tắt phase
        all_results[phase] = {
            "results": phase_results,
            "total_time": phase_time,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # TỔNG KẾT CUỐI CÙNG
    # ──────────────────────────────────────────────────────────────────────────
    phase_separator("═")
    print("\n✨ TỔNG KẾT")
    phase_separator("═")
    print()

    # Liệt kê kết quả từng phase
    for phase in phases_to_run:
        if phase not in all_results:
            continue

        results = all_results[phase]["results"]
        phase_time = all_results[phase]["total_time"]

        print(f"\n{PHASE_NAMES[phase]}")
        for res in results:
            status = "✅" if res["success"] else "❌"
            print(f"  {status} [{res['step']}] {res['description']:<50} ({format_time(res['elapsed'])})")

        print(f"  ├─ Tổng thời gian: {format_time(phase_time)}")

    # Tổng kết toàn bộ
    print(f"\n{'─' * 80}")
    print(f"📈 THỐNG KÊ CHUNG")
    print(f"{'─' * 80}")
    total_completed = sum(
        len(all_results[p]["results"]) for p in all_results
    )
    print(f"  ✅ Tổng bước hoàn thành: {total_completed}")
    print(f"  ⏱️  Tổng thời gian: {format_time(total_time)}")
    print()

    separator()
    print("🎉 PIPELINE HOÀN THÀNH THÀNH CÔNG!")
    separator()

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline bị dừng bởi người dùng.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
