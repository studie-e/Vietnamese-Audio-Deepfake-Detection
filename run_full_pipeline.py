import subprocess
import sys
import time
import os
import shutil
import argparse
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent
PYTHON   = sys.executable

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def format_time(sec):
    return str(timedelta(seconds=int(sec)))


def run_script(script, extra_args=None):
    """
    Chay script va vua in ra terminal vua ghi vao log file (tee-style).
    Dung Popen de stream output realtime thay vi cho den khi ket thuc.
    """
    name     = Path(script).stem
    log_file = LOG_DIR / f"{name}.log"
    cmd      = [PYTHON, script] + (extra_args or [])

    print("\n" + "=" * 80)
    print(f"RUNNING: {name}")
    if extra_args:
        print(f"Args   : {' '.join(extra_args)}")
    print("=" * 80)

    custom_env = os.environ.copy()
    custom_env["PYTHONIOENCODING"] = "utf-8"

    start = time.time()

    with open(log_file, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=custom_env,
        )

        # Stream tung dong: vua in ra terminal vua ghi vao log
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)

        proc.wait()

    elapsed = time.time() - start

    if proc.returncode != 0:
        print(f"\nFAILED: {name} (exit code {proc.returncode})")
        print(f"Log: {log_file}")
        return False, elapsed

    print(f"\nSUCCESS: {name} ({format_time(elapsed)})")
    print(f"Log: {log_file}")
    return True, elapsed


def save_results():
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    runs_dir = BASE_DIR / "vispoofdb" / "experiments" / "training_runs"

    if not runs_dir.exists():
        print(f"[WARN] Thu muc khong ton tai: {runs_dir}")
        return False

    runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])

    if not runs:
        print("[WARN] Khong tim thay thu muc run nao trong training_runs/")
        return False

    latest = runs[-1]
    src    = latest / "model_results.csv"
    dst    = runs_dir / "latest_model_results.csv"

    if src.exists():
        shutil.copy(src, dst)
        print(f"[OK] Sao chep ket qua moi nhat toi: {dst}")
        return True
    else:
        print(f"[WARN] Khong tim thay model_results.csv trong: {latest.name}")
        return False


def main(args):
    total_start = time.time()

    # Cac buoc trong pipeline voi extra_args tuong ung
    pipeline = [
        ("vispoofdb/scripts/scripts_data_process.py",   []),
        ("vispoofdb/scripts/scripts_feature_extract.py",[]),
        ("vispoofdb/scripts/scripts_train.py",          _build_train_args(args)),
        ("vispoofdb/scripts/experiment_fusion.py",      []),
        ("vispoofdb/scripts/plot_results.py",           []),
        ("vispoofdb/scripts/eval_noise_augmentation.py",[]),
        ("vispoofdb/scripts/quantize.py",               []),
    ]

    results = {}

    for idx, (script, extra_args) in enumerate(pipeline, start=1):
        print(f"\n[{idx}/{len(pipeline)}]")

        script_path = BASE_DIR / script
        if not script_path.exists():
            print(f"[WARN] Khong tim thay file: {script}, bo qua.")
            results[script] = "SKIPPED"
            continue

        ok, elapsed = run_script(str(script_path), extra_args)
        results[script] = f"OK ({format_time(elapsed)})" if ok else "FAILED"

        if not ok:
            print("\nPIPELINE STOPPED")
            _print_summary(results, time.time() - total_start)
            return 1

    save_results()

    total_elapsed = time.time() - total_start
    _print_summary(results, total_elapsed)
    print("\nPIPELINE COMPLETED SUCCESSFULLY")
    return 0


def _build_train_args(args):
    """Chuyen args thanh extra_args cho scripts_train.py."""
    extra = []
    if getattr(args, "skip_wav2vec", False):
        extra.append("--skip-wav2vec")
    if getattr(args, "skip_tone", False):
        extra.append("--skip-tone")
    return extra


def _print_summary(results, total_elapsed):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for script, status in results.items():
        name = Path(script).stem
        print(f"  {name:<40} {status}")
    print(f"\nTong thoi gian: {format_time(total_elapsed)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chay toan bo pipeline ViSpoofDB")
    parser.add_argument("--shutdown",      action="store_true", help="Tat may sau khi chay xong")
    parser.add_argument("--skip-wav2vec", action="store_true", help="Bo qua train_wav2vec.py")
    parser.add_argument("--skip-tone",    action="store_true", help="Bo qua cac Tone models")
    args = parser.parse_args()

    code = main(args)

    if args.shutdown:
        print("\nTu dong tat may trong 60 giay...")
        os.system("shutdown /s /t 60")

    sys.exit(code)