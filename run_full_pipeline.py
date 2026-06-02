import subprocess
import sys
import time
import os
import shutil
import argparse
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

PIPELINE = [
    "vispoofdb/scripts/scripts_data_process.py",
    "vispoofdb/scripts/scripts_feature_extract.py",
    "vispoofdb/scripts/scripts_train.py",
    "vispoofdb/scripts/experiment_fusion.py",
    "vispoofdb/scripts/plot_results.py",
    "vispoofdb/scripts/eval_noise_augmentation.py",
    "vispoofdb/scripts/quantize.py",
]

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def format_time(sec):
    return str(timedelta(seconds=int(sec)))


def run_script(script):
    name = Path(script).stem
    log_file = LOG_DIR / f"{name}.log"

    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    print("=" * 80)

    start = time.time()

    # --- BẢN VÁ LỖI TIẾNG VIỆT ---
    # Ép môi trường Windows phải dùng UTF-8 khi ghi file log
    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"

    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [PYTHON, script],
            cwd=BASE_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8", 
            env=custom_env 
        )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n❌ FAILED: {name}")
        print(f"Log file: {log_file}")
        return False, elapsed

    print(f"✅ SUCCESS ({format_time(elapsed)})")
    return True, elapsed


def save_results():
    print("\n" + "=" * 80)
    print("📊 SAVING RESULTS")
    print("=" * 80)

    runs_dir = BASE_DIR / "vispoofdb/experiments/training_runs"

    if not runs_dir.exists():
        print("⚠️ No training_runs found")
        return

    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if not runs:
        print("⚠️ No run directories found")
        return

    latest = runs[-1]
    src = latest / "model_results.csv"
    dst = runs_dir / "latest_model_results.csv"

    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ Saved latest model results to: {dst}")
    else:
        print(f"⚠️ model_results.csv not found in {latest.name}")


def main():
    total_start = time.time()

    for idx, script in enumerate(PIPELINE, start=1):
        print(f"\n[{idx}/{len(PIPELINE)}]")
        ok, _ = run_script(script)

        if not ok:
            print("\n❌ PIPELINE STOPPED")
            return 1

    save_results()

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print(f"TOTAL TIME: {format_time(total_elapsed)}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shutdown", action="store_true", help="Tắt máy tính sau khi chạy xong")
    args = parser.parse_args()

    code = main()

    if args.shutdown:
        print("\n💤 Tự động tắt máy trong 60 giây nữa...")
        os.system("shutdown /s /t 60")

    sys.exit(code)