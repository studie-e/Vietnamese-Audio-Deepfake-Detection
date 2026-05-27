"""
train_aasist.py
===============
Gọi AASIST training từ thư mục AASIST/, lưu kết quả vào vispoofdb/models_saved/

Cách chạy:
    python vispoofdb/models/train_aasist.py
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Setup paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
AASIST_DIR = BASE_DIR / 'AASIST'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
AASIST_TRAIN_SCRIPT = AASIST_DIR / 'train.py'

SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("AASIST Training Wrapper")
print("=" * 65)
print(f"AASIST folder: {AASIST_DIR}")
print(f"Save directory: {SAVE_MODEL_DIR}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Run AASIST training from AASIST directory
# ─────────────────────────────────────────────────────────────────────────────
if not AASIST_TRAIN_SCRIPT.exists():
    print(f"❌ Error: {AASIST_TRAIN_SCRIPT} not found!")
    sys.exit(1)

print(f"📍 Running: {AASIST_TRAIN_SCRIPT}")
print("=" * 65)

# Set environment for UTF-8
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

# Run training from AASIST directory
result = subprocess.run(
    [sys.executable, str(AASIST_TRAIN_SCRIPT)],
    cwd=str(AASIST_DIR),
    env=env,
    text=True
)

if result.returncode != 0:
    print(f"\n❌ Training failed with exit code {result.returncode}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Copy model to vispoofdb/models_saved/
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Moving model to vispoofdb/models_saved/")
print("=" * 65)

aasist_model = AASIST_DIR / 'aasist_best_model.pth'
target_model = SAVE_MODEL_DIR / 'aasist_best_model.pth'

if aasist_model.exists():
    print(f"📍 Moving: {aasist_model}")
    print(f"      to: {target_model}")
    shutil.copy(aasist_model, target_model)
    print(f"✅ Model saved to: {target_model}")
else:
    print(f"⚠️  Warning: Model file not found at {aasist_model}")

print("=" * 65)
print("✅ Training completed!")
print("=" * 65)
