# File: vispoofdb/models/train_tone_svm.py
"""
Huấn luyện SVM trên bộ đặc trưng Tone-Aware (24 chiều).
So sánh trực tiếp với SVM+MFCC (train_svm.py) và SVM+LFCC (train_lfcc_svm.py).

Đặc trưng:
  F0 statistics, F0 contour, Jitter/Shimmer, Delta F0, HNR, ZCR, Energy, MFCC-1
  → 24 chiều, gọn, có khả năng giải thích cao

Lưu mô hình: models_saved/svm_tone_model.pkl
"""
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve,
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
print("  SVM + TONE-AWARE FEATURES (24 chiều)")
print("  Đặc trưng: F0/Pitch Contour, Jitter, Shimmer, HNR, ...")
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
print(f"Kích thước y: {y.shape}")
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
print(f"Tập Test_seen:   {len(X_seen)}  mẫu  (nguồn đã thấy khi train)")
print(f"Tập Test_unseen: {len(X_unseen)} mẫu  (nguồn hoàn toàn mới — quan trọng nhất!)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Chuẩn hóa
# ─────────────────────────────────────────────────────────────────────────────
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_seen_sc   = scaler.transform(X_seen)
X_unseen_sc = scaler.transform(X_unseen)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Huấn luyện SVM
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- ĐANG HUẤN LUYỆN SVM (TONE-AWARE) ---")
model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    probability=True,
    random_state=42,
)
model.fit(X_train_sc, y_train)
print("Hoàn thành huấn luyện SVM!")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Đánh giá
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(fpr[idx])


def evaluate(name: str, X_t: np.ndarray, y_t: np.ndarray) -> tuple[float, float]:
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    acc    = accuracy_score(y_t, y_pred)
    prec   = precision_score(y_t, y_pred, zero_division=0)
    rec    = recall_score(y_t, y_pred, zero_division=0)
    f1     = f1_score(y_t, y_pred, zero_division=0)
    eer    = compute_eer(y_t, y_prob)

    print(f"\n{'='*65}")
    print(f"  [{name.upper()}]  —  SVM + Tone-Aware Features")
    print(f"{'='*65}")
    print(f"  Accuracy:  {acc  * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec  * 100:.2f}%")
    print(f"  F1-Score:  {f1   * 100:.2f}%")
    print(f"  EER:       {eer  * 100:.2f}%  (thấp hơn = tốt hơn)")
    print(f"{'='*65}")
    print(classification_report(y_t, y_pred, target_names=['Người Thật', 'Giọng AI']))
    print("MA TRẬN NHẦM LẪN:")
    print(confusion_matrix(y_t, y_pred))
    return acc, eer


evaluate('test_seen',   X_seen_sc,   y_seen)
evaluate('test_unseen', X_unseen_sc, y_unseen)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Lưu mô hình
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(model,  SAVE_MODEL_DIR / 'svm_tone_model.pkl')
joblib.dump(scaler, SAVE_MODEL_DIR / 'scaler_tone.pkl')
print(f"\nMô hình đã lưu tại {SAVE_MODEL_DIR}/")
print("  svm_tone_model.pkl")
print("  scaler_tone.pkl")
