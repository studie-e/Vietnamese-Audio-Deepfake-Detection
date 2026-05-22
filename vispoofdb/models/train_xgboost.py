# File: vispoofdb/models/train_xgboost.py
import os
import sys
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, accuracy_score,
                             ConfusionMatrixDisplay, roc_curve,
                             precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import joblib

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- Đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
LOAD_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'xgb'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Tải dữ liệu ---
print("Đang tải đặc trưng XGBoost...")
X      = np.load(LOAD_DIR / 'X_xgb.npy')
y      = np.load(LOAD_DIR / 'y_xgb.npy')
splits = np.load(LOAD_DIR / 'splits_xgb.npy', allow_pickle=True)

print(f"Kích thước X: {X.shape}, y: {y.shape}")
print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

# --- 2. Filter theo splits ---
mask_train   = splits == 'train'
mask_seen    = splits == 'test_seen'
mask_unseen  = splits == 'test_unseen'

X_train,  y_train  = X[mask_train],  y[mask_train]
X_seen,   y_seen   = X[mask_seen],   y[mask_seen]
X_unseen, y_unseen = X[mask_unseen], y[mask_unseen]

print(f"\nTập Train:        {len(X_train)} mẫu")
print(f"Tập Test_seen:    {len(X_seen)}  mẫu")
print(f"Tập Test_unseen:  {len(X_unseen)} mẫu")

# Tách val từ train để dùng cho Early Stopping (20% của train)
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ==============================================================================
# BƯỚC 1: SEARCH HYPERPARAMETERS
# ==============================================================================
print("\nBước 1: Tìm kiếm bộ tham số (Hyperparameter Tuning)...")

param_dist = {
    'max_depth':         [3, 4, 5],
    'learning_rate':     [0.01, 0.05, 0.1],
    'n_estimators':      [200],
    'subsample':         [0.7, 0.8, 0.9],
    'colsample_bytree':  [0.7, 0.8, 0.9],
    'min_child_weight':  [1, 3, 5],
    'gamma':             [0, 0.1, 0.5],
    'reg_alpha':         [0, 0.1, 1],
    'reg_lambda':        [1, 2, 5]
}

search_model = xgb.XGBClassifier(
    random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=1
)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=search_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Best Params: {best_params}")

# ==============================================================================
# BƯỚC 2: TÌM BEST_ITERATION VỚI EARLY STOPPING
# ==============================================================================
print("\nBước 2: Tìm số lượng cây tối ưu (Early Stopping)...")
final_params = best_params.copy()
final_params['n_estimators'] = 2000
temp_model = xgb.XGBClassifier(
    **final_params, random_state=42, eval_metric='logloss',
    early_stopping_rounds=50, n_jobs=-1
)
temp_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_n_trees = temp_model.best_iteration
print(f"Số cây tối ưu tìm được: {best_n_trees}")

# ==============================================================================
# BƯỚC 3: REFIT TRÊN TOÀN BỘ TRAIN
# ==============================================================================
print(f"\nBước 3: Huấn luyện model cuối cùng với {best_n_trees} cây...")
final_config = best_params.copy()
final_config.pop('n_estimators', None)
final_model = xgb.XGBClassifier(
    **final_config, n_estimators=best_n_trees, random_state=42,
    eval_metric='logloss', n_jobs=-1
)
final_model.fit(X_train, y_train)
joblib.dump(final_model, SAVE_MODEL_DIR / 'best_xgboost.pkl')
print("Đã lưu mô hình XGBoost tốt nhất")

# ==============================================================================
# BƯỚC 4: ĐÁNH GIÁ
# ==============================================================================
def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return fpr[idx]

def evaluate(name, X_t, y_t):
    y_pred = final_model.predict(X_t)
    y_prob = final_model.predict_proba(X_t)[:, 1]
    acc    = accuracy_score(y_t, y_pred)
    prec   = precision_score(y_t, y_pred)
    rec    = recall_score(y_t, y_pred)
    f1     = f1_score(y_t, y_pred)
    eer    = compute_eer(y_t, y_prob)
    print(f"\n{'='*55}")
    print(f"  [{name.upper()}]")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  F1-Score:  {f1 * 100:.2f}%")
    print(f"  EER:       {eer * 100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_t, y_pred, target_names=['Real', 'Fake']))
    return acc, eer

evaluate('test_seen',   X_seen,   y_seen)
evaluate('test_unseen', X_unseen, y_unseen)

# Lưu confusion matrix
ConfusionMatrixDisplay.from_estimator(
    final_model, X_seen, y_seen, display_labels=['Real', 'Fake']
)
plt.title("Confusion Matrix - XGBoost (test_seen)")
plt.savefig(str(SAVE_MODEL_DIR / 'confusion_matrix_xgboost.png'), dpi=300)
plt.close()
print(f"Đã lưu confusion matrix tại {SAVE_MODEL_DIR}/")
