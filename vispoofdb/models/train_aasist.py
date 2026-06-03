"""
train_aasist.py
===============
Goi AASIST training tu vispoofdb/models/aasist/.
Ket qua luu tai: vispoofdb/models_saved/aasist_best_model.pth

Cach chay:
    python vispoofdb/models/train_aasist.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR     = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = BASE_DIR / 'vispoofdb' / 'models' / 'aasist' / 'train_aasist_model.py'

print("=" * 65)
print("  AASIST Training Wrapper")
print(f"  Script: {TRAIN_SCRIPT.relative_to(BASE_DIR)}")
print("=" * 65)

if not TRAIN_SCRIPT.exists():
    print(f"[ERROR] Khong tim thay script: {TRAIN_SCRIPT}")
    sys.exit(1)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(
    [sys.executable, str(TRAIN_SCRIPT)],
    cwd=str(BASE_DIR),
    env=env,
    text=True,
)

if result.returncode != 0:
    print(f"\n[ERROR] Training that bai (exit code {result.returncode})")
    sys.exit(1)

print("\n[OK] Training hoan tat!")
print("     Model da luu tai: vispoofdb/models_saved/aasist_best_model.pth")
print("=" * 65)
