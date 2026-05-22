# File: vispoofdb/models/train_tone_fusion.py
"""
Fusion: MFCC (40 chiều) + Tone-Aware (24 chiều) → 64 chiều → SVM.

Đây là thực nghiệm chính để đo đóng góp thực sự của Tone-Aware features.
Kết quả sẽ trả lời câu hỏi:
  "Thêm đặc trưng F0/pitch vào MFCC có cải thiện phát hiện deepfake tiếng Việt không?"

Bảng so sánh (sẽ in sau khi huấn luyện):
  ┌──────────────────────────────┬──────────┬────────────────────┐
  │ Mô hình                      │ Accuracy │ EER (test_unseen)  │
  ├──────────────────────────────┼──────────┼────────────────────┤
  │ SVM + MFCC (40d)             │  ??.??%  │     ??.??%         │
  │ SVM + Tone-Aware (24d)       │  ??.??%  │     ??.??%         │
  │ SVM + MFCC + Tone (64d) [☆] │  ??.??%  │     ??.??%         │
  └──────────────────────────────┴──────────┴────────────────────┘

Lưu mô hình: models_saved/svm_tone_fusion_model.pkl
             models_saved/scaler_tone_fusion.pkl
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
MFCC_DIR       = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'svm'
TONE_DIR       = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'tone'
SAVE_MODEL_DIR = BASE_DIR / 'vispoofdb' / 'models_saved'
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  LATE FUSION: MFCC (40d) + TONE-AWARE (24d) → 64d")
print("  Mô hình: SVM với kernel RBF")
print("=" * 65)

# Kiểm tra files
required = [
    (MFCC_DIR / 'X_all.npy',       'MFCC features (cần svm_features.py)'),
    (MFCC_DIR / 'y_all.npy',       'MFCC labels'),
    (MFCC_DIR / 'splits_svm.npy',  'MFCC splits'),
    (MFCC_DIR / 'paths_svm.npy',   'MFCC paths'),
    (TONE_DIR / 'X_tone.npy',      'Tone-Aware features (cần tone_features.py)'),
    (TONE_DIR / 'y_tone.npy',      'Tone-Aware labels'),
    (TONE_DIR / 'splits_tone.npy', 'Tone-Aware splits'),
    (TONE_DIR / 'paths_tone.npy',  'Tone-Aware paths'),
]
missing = [(path, desc) for path, desc in required if not path.exists()]
if missing:
    print("\n[ERROR] Thiếu các file sau:")
    for path, desc in missing:
        print(f"  - {path.name}  ({desc})")
    print("\nHãy chạy svm_features.py và tone_features.py trước!")
    sys.exit(1)

print("\nĐang tải dữ liệu MFCC và Tone-Aware...")

X_mfcc        = np.load(MFCC_DIR / 'X_all.npy')
y_mfcc        = np.load(MFCC_DIR / 'y_all.npy')
splits_mfcc   = np.load(MFCC_DIR / 'splits_svm.npy',  allow_pickle=True)
paths_mfcc    = np.load(MFCC_DIR / 'paths_svm.npy',   allow_pickle=True)

X_tone        = np.load(TONE_DIR / 'X_tone.npy')
y_tone        = np.load(TONE_DIR / 'y_tone.npy')
splits_tone   = np.load(TONE_DIR / 'splits_tone.npy', allow_pickle=True)
paths_tone    = np.load(TONE_DIR / 'paths_tone.npy',  allow_pickle=True)

print(f"MFCC:       {X_mfcc.shape}  —  {len(X_mfcc)} mẫu, {X_mfcc.shape[1]} chiều")
print(f"Tone-Aware: {X_tone.shape}  —  {len(X_tone)} mẫu, {X_tone.shape[1]} chiều")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Khớp và đồng bộ dữ liệu theo đường dẫn file chung
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd

df_mfcc = pd.DataFrame({
    'file_path': paths_mfcc,
    'mfcc_idx': np.arange(len(paths_mfcc))
})
df_tone = pd.DataFrame({
    'file_path': paths_tone,
    'tone_idx': np.arange(len(paths_tone))
})

# Khớp 2 tập dữ liệu theo file_path chung
df_merged = pd.merge(df_mfcc, df_tone, on='file_path', how='inner')
print(f"\nKhớp dữ liệu: Tìm thấy {len(df_merged)} tệp tin chung được trích xuất thành công.")

if len(df_merged) == 0:
    print("[ERROR] Không tìm thấy tệp tin chung nào giữa hai tập đặc trưng!")
    sys.exit(1)

# Lọc và đồng bộ thứ tự đặc trưng
mfcc_idxs = df_merged['mfcc_idx'].values
tone_idxs = df_merged['tone_idx'].values

X_mfcc = X_mfcc[mfcc_idxs]
y_mfcc = y_mfcc[mfcc_idxs]
splits_mfcc = splits_mfcc[mfcc_idxs]

X_tone = X_tone[tone_idxs]
y_tone = y_tone[tone_idxs]
splits_tone = splits_tone[tone_idxs]

# Kiểm tra nhãn khớp nhau
if not np.array_equal(y_mfcc, y_tone):
    print("\n[ERROR] Nhãn MFCC và Tone-Aware vẫn không khớp sau khi đồng bộ!")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Late Fusion: ghép MFCC + Tone-Aware
# ─────────────────────────────────────────────────────────────────────────────
X_fused = np.hstack([X_mfcc, X_tone])  # (N, 40+24) = (N, 64)
y       = y_mfcc
splits  = splits_mfcc

print(f"\nFusion vector: {X_fused.shape}  (MFCC {X_mfcc.shape[1]}d + Tone {X_tone.shape[1]}d = {X_fused.shape[1]}d)")
print(f"Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Phân chia theo splits
# ─────────────────────────────────────────────────────────────────────────────
mask_train  = splits == 'train'
mask_seen   = splits == 'test_seen'
mask_unseen = splits == 'test_unseen'

X_train,  y_train  = X_fused[mask_train],  y[mask_train]
X_seen,   y_seen   = X_fused[mask_seen],   y[mask_seen]
X_unseen, y_unseen = X_fused[mask_unseen], y[mask_unseen]

print(f"\nTập Train:       {len(X_train)} mẫu")
print(f"Tập Test_seen:   {len(X_seen)}  mẫu")
print(f"Tập Test_unseen: {len(X_unseen)} mẫu")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Chuẩn hóa
# ─────────────────────────────────────────────────────────────────────────────
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_seen_sc   = scaler.transform(X_seen)
X_unseen_sc = scaler.transform(X_unseen)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Huấn luyện SVM Fusion
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- ĐANG HUẤN LUYỆN SVM FUSION (MFCC + TONE-AWARE) ---")
model = SVC(
    kernel='rbf',
    C=10.0,
    gamma='scale',
    probability=True,
    random_state=42,
)
model.fit(X_train_sc, y_train)
print("Hoàn thành huấn luyện SVM Fusion!")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Đánh giá
# ─────────────────────────────────────────────────────────────────────────────
def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(fpr[idx])


results: dict[str, tuple[float, float]] = {}


def evaluate(name: str, X_t: np.ndarray, y_t: np.ndarray) -> tuple[float, float]:
    y_pred = model.predict(X_t)
    y_prob = model.predict_proba(X_t)[:, 1]
    acc    = accuracy_score(y_t, y_pred)
    prec   = precision_score(y_t, y_pred, zero_division=0)
    rec    = recall_score(y_t, y_pred, zero_division=0)
    f1     = f1_score(y_t, y_pred, zero_division=0)
    eer    = compute_eer(y_t, y_prob)
    results[name] = (acc, eer)

    print(f"\n{'='*65}")
    print(f"  [{name.upper()}]  —  SVM + MFCC+Tone Fusion (64d)")
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
# 8. Bảng tổng hợp gợi ý so sánh
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  GỢI Ý SO SÁNH VỚI CÁC MÔ HÌNH KHÁC")
print("=" * 65)
print("  Chạy toàn bộ pipeline để có bảng so sánh đầy đủ:")
print()
print("  Mô hình                          | test_seen | test_unseen (EER)")
print("  ---------------------------------+-----------+------------------")
print("  SVM + MFCC (40d)                 |   ??.??%  |    ??.??%")
print("  SVM + LFCC (40d)                 |   ??.??%  |    ??.??%")
print("  SVM + Tone-Aware (24d)           |   ??.??%  |    ??.??%")
acc_seen,   eer_seen   = results.get('test_seen',   (0, 0))
acc_unseen, eer_unseen = results.get('test_unseen', (0, 0))
print(f"  SVM + MFCC+Tone Fusion (64d) [☆]| {acc_seen*100:6.2f}%  |   {eer_unseen*100:.2f}%")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Lưu mô hình
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(model,  SAVE_MODEL_DIR / 'svm_tone_fusion_model.pkl')
joblib.dump(scaler, SAVE_MODEL_DIR / 'scaler_tone_fusion.pkl')
print(f"\nMô hình đã lưu tại {SAVE_MODEL_DIR}/")
print("  svm_tone_fusion_model.pkl")
print("  scaler_tone_fusion.pkl")
