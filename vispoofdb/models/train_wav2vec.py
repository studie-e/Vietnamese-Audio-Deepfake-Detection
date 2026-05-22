# File: vispoofdb/models/train_wav2vec.py
import numpy as np
import os
import sys
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- Đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
LOAD_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_wav2vec'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return fpr[idx]

# --- 1. Load Dữ liệu 768 chiều ---
print("Đang tải ma trận dữ liệu 768 chiều...")
X_path      = LOAD_DIR / 'X_wav2vec.npy'
y_path      = LOAD_DIR / 'y_wav2vec.npy'
splits_path = LOAD_DIR / 'splits_wav2vec.npy'

if not X_path.exists() or not y_path.exists() or not splits_path.exists():
    print("[ERROR] Feature files not found. Run wav2vec2.py first.")
    exit(1)

X      = np.load(X_path)
y      = np.load(y_path)
splits = np.load(splits_path, allow_pickle=True)

print(f"Tổng số mẫu: {len(X)}")
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

# --- 3. Chuẩn hóa ---
scaler = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_seen_sc   = scaler.transform(X_seen)
X_unseen_sc = scaler.transform(X_unseen)

# --- 4. Cấu hình Mạng Nơ-ron (MLP) ---
print("\n--- ĐANG HUẤN LUYỆN MẠNG NƠ-RON (MLP) TRÊN ĐẶC TRƯNG WAV2VEC ---")
model = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)
model.fit(X_train_sc, y_train)

# --- 5. Đánh giá ---
def evaluate(name, X_t, y_t):
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    acc    = accuracy_score(y_t, y_pred)
    eer    = compute_eer(y_t, y_prob)
    print(f"\n{'='*55}")
    print(f"  [{name.upper()}]")
    print(f"  ĐỘ CHÍNH XÁC (Wav2Vec2 + MLP): {acc * 100:.2f}%")
    print(f"  EER (Equal Error Rate):         {eer * 100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_t, y_pred, target_names=['Người Thật', 'Giọng AI']))
    print("MA TRẬN NHẦM LẪN:")
    print(confusion_matrix(y_t, y_pred))
    return acc, eer

evaluate('test_seen',   X_seen_sc,   y_seen)
evaluate('test_unseen', X_unseen_sc, y_unseen)

# --- 6. Lưu Mô hình ---
joblib.dump(model,  SAVE_MODEL_DIR / 'mlp_wav2vec_model.pkl')
joblib.dump(scaler, SAVE_MODEL_DIR / 'scaler_wav2vec.pkl')
print(f"\nĐã lưu mô hình Mạng Nơ-ron tại {SAVE_MODEL_DIR}/")
