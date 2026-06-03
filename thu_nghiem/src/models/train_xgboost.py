import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib

# 1. TẢI DỮ LIỆU
X = np.load('data/features_model/xgb/X_xgb.npy')
y = np.load('data/features_final/y_label.npy')

# Split 1: 80% Train_full, 20% Test (Untouched)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split 2: Từ Train_full tách ra Train và Val (để dùng cho Early Stopping)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ==============================================================================
# BƯỚC 1: SEARCH HYPERPARAMETERS (KHÔNG EARLY STOPPING)
# ==============================================================================
print("Bước 1: Tìm kiếm bộ tham số khung (Hyperparameter Tuning)...")

param_dist = {
    'max_depth': [3, 4, 5],            # Đã tinh gọn theo gợi ý
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200],             # Cố định để so sánh công bằng giữa các fold
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

# n_jobs=1 bên trong model để tránh xung đột với n_jobs=-1 của RandomizedSearchCV
search_model = xgb.XGBClassifier(
    random_state=42, 
    eval_metric='logloss', 
    tree_method='hist',
    n_jobs=1 
)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=search_model,
    param_distributions=param_dist,
    n_iter=60, 
    scoring='accuracy',
    cv=cv_strategy,
    n_jobs=-1, # Dùng tối đa CPU cho các fold chạy song song
    verbose=1,
    random_state=42
)

random_search.fit(X_train_full, y_train_full)
best_params = random_search.best_params_
print(f"Best Params: {best_params}")

# ==============================================================================
# BƯỚC 2: TÌM BEST_ITERATION VỚI EARLY STOPPING
# ==============================================================================
print("\nBước 2: Tìm số lượng cây tối ưu (Early Stopping)...")

final_params = best_params.copy()
final_params['n_estimators'] = 2000 # Tăng cao để early stopping tự quyết

temp_model = xgb.XGBClassifier(
    **final_params,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
    n_jobs=-1 # Bây giờ mới dùng full CPU cho 1 model duy nhất
)

temp_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

best_n_trees = temp_model.best_iteration
print(f"Số cây tối ưu tìm được: {best_n_trees}")

# ==============================================================================
# BƯỚC 3: REFIT TRÊN TOÀN BỘ TRAIN_FULL (MAX PING) - ĐÃ SỬA LỖI
# ==============================================================================
print(f"\nBước 3: Huấn luyện model cuối cùng trên 100% dữ liệu Train với {best_n_trees} cây...")

# Xóa n_estimators cũ (200) khỏi dictionary để tránh xung đột
final_config = best_params.copy()
final_config.pop('n_estimators', None) 

final_model = xgb.XGBClassifier(
    **final_config,           # Truyền các tham số tối ưu khác
    n_estimators=best_n_trees, # Dùng con số 400 mới tìm được
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

final_model.fit(X_train_full, y_train_full)
joblib.dump(final_model, 'models_saved/best_xgboost.pkl')
print("Đã lưu mô hình XGBoost tốt nhất vào 'models_saved/best_xgboost.pkl'")
# ==============================================================================
# 4. ĐÁNH GIÁ CUỐI CÙNG (SO SÁNH TRAIN VS TEST)
# ==============================================================================
# Dự đoán trên tập Train Full (toàn bộ dữ liệu dùng để học)
y_train_full_pred = final_model.predict(X_train_full)
train_acc = accuracy_score(y_train_full, y_train_full_pred)

# Dự đoán trên tập Test (dữ liệu chưa từng thấy)
y_test_pred = final_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n" + "="*50)
print(f"SO SÁNH HIỆU NĂNG HỆ THỐNG:")
print(f"Độ chính xác trên tập TRAIN: {train_acc * 100:.2f}%")
print(f"Độ chính xác trên tập TEST : {test_acc * 100:.2f}%")
print(f"Khoảng cách (Gap)         : {(train_acc - test_acc) * 100:.2f}%")
print("="*50)

if (train_acc - test_acc) > 0.10:
    print("Cảnh báo: Mô hình vẫn còn dấu hiệu Overfitting (Gap > 10%).")
else:
    print("Chúc mừng: Mô hình có tính tổng quát hóa tốt (Generalization).")

# Tính thêm các metrics: Precision, Recall, F1-Score và EER
y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Lấy probability của class 1 (Fake)

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Tính EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = fpr[eer_idx]

print("\n" + "="*50)
print(f"PRECISION: {precision * 100:.2f}%")
print(f"RECALL   : {recall * 100:.2f}%")
print(f"F1-SCORE : {f1 * 100:.2f}%")
print(f"EER      : {eer * 100:.2f}%")
print("="*50)

# Lưu ma trận nhầm lẫn để phân tích sâu hơn
ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test, display_labels=['Real', 'Fake'])
plt.title("Confusion Matrix - XGBoost")
plt.savefig('figures/confusion_matrix_xgboost.png', dpi=300)
plt.show()