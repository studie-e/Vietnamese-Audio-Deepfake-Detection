import os
import sys
import librosa
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- 1. Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
FEATURES_MFCC_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_mfcc'
SAVE_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'xgb'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra tồn tại splits_mfcc.npy
splits_path = FEATURES_MFCC_DIR / 'splits_mfcc.npy'
if not splits_path.exists():
    print("[ERROR] Không tìm thấy splits_mfcc.npy!")
    print("Hãy chạy lại vidb_extract_mfcc.py (đã được cập nhật) trước!")
    exit(1)

print("Đang tải dữ liệu thô MFCC...")
X_base = np.load(FEATURES_MFCC_DIR / 'X_data.npy')    # (N, 40, 157)
y      = np.load(FEATURES_MFCC_DIR / 'y_label.npy')    # (N,)
splits = np.load(splits_path, allow_pickle=True)         # (N,) strings

print(f"Dữ liệu thô: {X_base.shape}")
print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

# --- 2. Feature Engineering ---
print("\nĐang tính toán Vận tốc (Delta) và Gia tốc (Delta-Delta) âm thanh...")
X_delta  = librosa.feature.delta(X_base)
X_delta2 = librosa.feature.delta(X_base, order=2)

# Ghép 3 ma trận: 40 MFCC + 40 Delta + 40 Delta2 = 120 theo trục feature (axis=1)
X_combined = np.concatenate((X_base, X_delta, X_delta2), axis=1)
print(f"Đã sinh thêm feature! Kích thước ma trận mới: {X_combined.shape}")

# --- 3. Tạo đặc trưng cho XGBoost / Random Forest ---
print("\nĐang chế biến Features cho XGBoost...")
X_mean = np.mean(X_combined, axis=2)    # 120 features
X_std  = np.std(X_combined, axis=2)     # 120 features
X_max  = np.max(X_combined, axis=2)     # 120 features
X_min  = np.min(X_combined, axis=2)     # 120 features

X_xgb = np.hstack((X_mean, X_std, X_max, X_min))  # (N, 480)

# --- 4. Lưu kết quả ---
np.save(SAVE_DIR / 'X_xgb.npy', X_xgb)
np.save(SAVE_DIR / 'y_xgb.npy', y)
np.save(SAVE_DIR / 'splits_xgb.npy', splits)

print(f"Đã lưu X_xgb.npy với kích thước: {X_xgb.shape}")
print(f"Đã lưu y_xgb.npy, splits_xgb.npy tại: {SAVE_DIR}")
