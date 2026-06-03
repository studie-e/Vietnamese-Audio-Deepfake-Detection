import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Định nghĩa đường dẫn
BASE_X_PATH = 'data/features_final/X_data.npy'
BASE_Y_PATH = 'data/features_final/y_label.npy'

# Tạo thư mục con nếu chưa có
os.makedirs('data/features_model/xgb', exist_ok=True)

print("Đang tải dữ liệu thô...")
X_base = np.load(BASE_X_PATH) # Kích thước: (2398, 40, 157)
y = np.load(BASE_Y_PATH)
print(f"Dữ liệu thô: {X_base.shape}")

# CHIẾN THUẬT 1: SINH THÊM FEATURE (FEATURE ENGINEERING)
# Tính Đạo hàm bậc 1 (Vận tốc) và Bậc 2 (Gia tốc) của MFCC
# ==========================================
print("\nĐang tính toán Vận tốc (Delta) và Gia tốc (Delta-Delta) âm thanh...")
# librosa.feature.delta hỗ trợ tính toán trực tiếp trên ma trận 3D
X_delta = librosa.feature.delta(X_base)
X_delta2 = librosa.feature.delta(X_base, order=2)

# Ghép 3 ma trận lại với nhau theo trục đặc trưng (axis=1)
# 40 MFCC + 40 Delta + 40 Delta2 = 120 Đặc trưng cho mỗi khung thời gian
X_combined = np.concatenate((X_base, X_delta, X_delta2), axis=1) 
print(f"Đã sinh thêm feature! Kích thước ma trận mới: {X_combined.shape}")

# ==========================================
# TẠO ĐẶC TRƯNG CHO XGBOOST / RANDOM FOREST
# Chiến thuật: Lấy Thống kê (Mean, Std, Max, Min) theo trục thời gian (axis=2)
# ==========================================
print("\nĐang chế biến Features cho XGBoost...")
X_mean = np.mean(X_combined, axis=2) # 120 features
X_std = np.std(X_combined, axis=2)   # 120 features
X_max = np.max(X_combined, axis=2)   # 120 features
X_min = np.min(X_combined, axis=2)   # 120 features

# Ghép 4 mảng này lại theo chiều ngang. Kích thước mới: (2398, 160)
X_xgb = np.hstack((X_mean, X_std, X_max, X_min)) 
np.save('data/features_model/xgb/X_xgb.npy', X_xgb)
print(f" Đã lưu X_xgb.npy với kích thước: {X_xgb.shape} (Vũ khí tối thượng cho Cây quyết định)")
