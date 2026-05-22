# File: vispoofdb/models/train_mlp.py
import os
import sys
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# --- Đường dẫn ---
BASE_DIR = Path(__file__).resolve().parents[2]
LOAD_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'MLP'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return fpr[idx]

def train_model():
    print("Loading features...")

    X_path      = LOAD_DIR / 'X_mlp.npy'
    y_path      = LOAD_DIR / 'y_mlp.npy'
    splits_path = LOAD_DIR / 'splits_mlp.npy'

    if not X_path.exists() or not y_path.exists() or not splits_path.exists():
        print("Error: Feature files not found. Run mlp_features.py first.")
        return

    X      = np.load(X_path)
    y      = np.load(y_path)
    splits = np.load(splits_path, allow_pickle=True)

    print(f"Tổng số mẫu: {len(X)}")
    print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

    # --- Filter theo splits ---
    mask_train   = splits == 'train'
    mask_seen    = splits == 'test_seen'
    mask_unseen  = splits == 'test_unseen'

    X_train,  y_train  = X[mask_train],  y[mask_train]
    X_seen,   y_seen   = X[mask_seen],   y[mask_seen]
    X_unseen, y_unseen = X[mask_unseen], y[mask_unseen]

    print(f"\nTập Train:        {len(X_train)} mẫu")
    print(f"Tập Test_seen:    {len(X_seen)}  mẫu")
    print(f"Tập Test_unseen:  {len(X_unseen)} mẫu")

    # --- Chuẩn hóa ---
    scaler = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_seen_sc   = scaler.transform(X_seen)
    X_unseen_sc = scaler.transform(X_unseen)

    # --- Huấn luyện MLP ---
    print("\nTraining MLP model...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        alpha=0.2,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
        verbose=True
    )
    mlp_model.fit(X_train_sc, y_train)

    # --- Hàm đánh giá ---
    def evaluate(name, X_t, y_t):
        y_pred = mlp_model.predict(X_t)
        y_prob = mlp_model.predict_proba(X_t)[:, 1]
        acc    = accuracy_score(y_t, y_pred)
        eer    = compute_eer(y_t, y_prob)
        print(f"\n{'='*55}")
        print(f"  [{name.upper()}]")
        print(f"  Accuracy on Test Set: {acc * 100:.2f}%")
        print(f"  EER (Equal Error Rate): {eer * 100:.2f}%")
        print(f"{'='*55}")
        print(classification_report(y_t, y_pred, target_names=["Real (0)", "AI (1)"]))
        return acc, eer

    evaluate('test_seen',   X_seen_sc,   y_seen)
    evaluate('test_unseen', X_unseen_sc, y_unseen)

    # --- Lưu model ---
    joblib.dump(mlp_model, SAVE_MODEL_DIR / 'best_mlp.pkl')
    joblib.dump(scaler,    SAVE_MODEL_DIR / 'scaler_mlp.pkl')
    print(f"\nModel successfully saved to: {SAVE_MODEL_DIR}")

if __name__ == "__main__":
    train_model()
