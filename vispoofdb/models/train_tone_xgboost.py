# File: vispoofdb/models/train_tone_xgboost.py
"""
Huấn luyện XGBoost trên bộ đặc trưng Tone-Aware (24 chiều).
So sánh trực tiếp với XGBoost+MFCC-Delta (train_xgboost.py — 480 chiều).

Mục tiêu: kiểm tra xem 24 chiều Tone-Aware có cạnh tranh được với
480 chiều MFCC hay không, đặc biệt trên test_unseen (generalization).

Lưu mô hình: models_saved/xgboost_tone_model.pkl
"""
import sys
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_curve,
    precision_score, recall_score, f1_score,
)

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[2]
LOAD_DIR       = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'tone'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  XGBOOST + TONE-AWARE FEATURES (24 chiều)")
print("  Hyperparameter tuning + Early Stopping")
print("=" * 65)

for f in ['X_tone.npy', 'y_tone.npy', 'splits_tone.npy']:
    if not (LOAD_DIR / f).exists():
        print(f"\n[ERROR] Không tìm thấy {f} tại {LOAD_DIR}")
        print("Hãy chạy tone_features.py trước!")
        sys.exit(1)

print("\nĐang tải dữ liệu Tone-Aware...")
X      = np.load(LOAD_DIR / 'X_tone.npy')
y      = np.load(LOAD_DIR / 'y_tone.npy')
splits = np.load(LOAD_DIR / 'splits_tone.npy', allow_pickle=True)

print(f"Kích thước X: {X.shape}  (N x {X.shape[1]} đặc trưng Tone-Aware)")
print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Phân chia theo splits
# ─────────────────────────────────────────────────────────────────────────────
mask_train  = splits == 'train'
mask_seen   = splits == 'test_seen'
mask_unseen = splits == 'test_unseen'

X_train,  y_train  = X[mask_train],  y[mask_train]
X_seen,   y_seen   = X[mask_seen],   y[mask_seen]
X_unseen, y_unseen = X[mask_unseen], y[mask_unseen]

print(f"\nTập Train:       {len(X_train)} mẫu")
print(f"Tập Test_seen:   {len(X_seen)}  mẫu")
print(f"Tập Test_unseen: {len(X_unseen)} mẫu")

# Tách val từ train để dùng cho Early Stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Tìm kiếm Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
print("\nBước 1: Tìm kiếm bộ tham số (Hyperparameter Tuning)...")

param_dist = {
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'n_estimators':     [200],
    'subsample':        [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.3],
    'reg_alpha':        [0, 0.1, 0.5],
    'reg_lambda':       [1, 2, 5],
}

search_model = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=1,
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
    random_state=42,
)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Best Params: {best_params}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Early Stopping để tìm số cây tối ưu
# ─────────────────────────────────────────────────────────────────────────────
print("\nBước 2: Tìm số lượng cây tối ưu (Early Stopping)...")
final_params = best_params.copy()
final_params['n_estimators'] = 2000

temp_model = xgb.XGBClassifier(
    **final_params,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
    n_jobs=-1,
)
temp_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_n_trees = temp_model.best_iteration
print(f"Số cây tối ưu tìm được: {best_n_trees}")

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: Refit trên toàn bộ train
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBước 3: Huấn luyện model cuối cùng với {best_n_trees} cây...")
final_config = best_params.copy()
final_config.pop('n_estimators', None)

final_model = xgb.XGBClassifier(
    **final_config,
    n_estimators=best_n_trees,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1,
)
final_model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Đánh giá
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(fpr[idx])


def evaluate(name: str, X_t: np.ndarray, y_t: np.ndarray) -> tuple[float, float]:
    y_pred = final_model.predict(X_t)
    y_prob = final_model.predict_proba(X_t)[:, 1]
    acc    = accuracy_score(y_t, y_pred)
    prec   = precision_score(y_t, y_pred, zero_division=0)
    rec    = recall_score(y_t, y_pred, zero_division=0)
    f1     = f1_score(y_t, y_pred, zero_division=0)
    eer    = compute_eer(y_t, y_prob)

    print(f"\n{'='*65}")
    print(f"  [{name.upper()}]  —  XGBoost + Tone-Aware Features")
    print(f"{'='*65}")
    print(f"  Accuracy:  {acc  * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec  * 100:.2f}%")
    print(f"  F1-Score:  {f1   * 100:.2f}%")
    print(f"  EER:       {eer  * 100:.2f}%  (thấp hơn = tốt hơn)")
    print(f"{'='*65}")
    print(classification_report(y_t, y_pred, target_names=['Real', 'Fake']))
    return acc, eer


evaluate('test_seen',   X_seen,   y_seen)
evaluate('test_unseen', X_unseen, y_unseen)

# Feature importance
print("\nTop-10 Tone-Aware Features quan trọng nhất (XGBoost):")
FEATURE_NAMES = [
    'f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 'f0_range',
    'f0_linear_slope', 'voiced_rate', 'f0_mean_abs_delta',
    'local_jitter', 'rap_jitter', 'local_shimmer', 'db_shimmer',
    'delta_f0_mean', 'delta_f0_std', 'delta2_f0_mean', 'delta2_f0_std',
    'hnr_mean',
    'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
    'mfcc1_mean', 'mfcc1_std',
]
importances = final_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx[:10], 1):
    feat_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'feat_{idx}'
    print(f"  {rank:2d}. {feat_name:<20s}  importance = {importances[idx]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Lưu mô hình
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(final_model, SAVE_MODEL_DIR / 'xgboost_tone_model.pkl')
print(f"\nMô hình đã lưu tại {SAVE_MODEL_DIR}/")
print("  xgboost_tone_model.pkl")
