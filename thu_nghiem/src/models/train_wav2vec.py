import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

# --- 1. Load Dữ liệu 768 chiều ---
LOAD_DIR = 'data/features_wav2vec'
print("Đang tải ma trận dữ liệu 768 chiều...")
X = np.load(os.path.join(LOAD_DIR, 'X_wav2vec.npy'))
y = np.load(os.path.join(LOAD_DIR, 'y_wav2vec.npy'))

# --- 2. Chia tập Train/Val (80% học - 20% thi) ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Số mẫu Train: {len(X_train)} | Số mẫu Val: {len(X_val)}")

# --- 3. Chuẩn hóa dữ liệu ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 4. Cấu hình Mạng Nơ-ron (MLP) ---
print("\n--- ĐANG HUẤN LUYỆN MẠNG NƠ-RON (MLP) ---")
# Cấu hình: 2 lớp ẩn (512 nơ-ron và 256 nơ-ron), chống học vẹt (early_stopping)
model = MLPClassifier(
    hidden_layer_sizes=(512, 256), 
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,  # Tự dừng nếu model không tiến bộ để tránh học vẹt
    validation_fraction=0.1,
    random_state=42,
    verbose=True          # In quá trình học ra màn hình
)

model.fit(X_train_scaled, y_train)

# --- 5. Chấm điểm bài thi thực tế ---
print("\n--- KẾT QUẢ TRÊN TẬP VALIDATION (20%) ---")
y_pred = model.predict(X_val_scaled)
y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # Lấy probability của class 1

# Tính EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = fpr[eer_idx]

acc = accuracy_score(y_val, y_pred)
print("=" * 50)
print(f"🔥 ĐỘ CHÍNH XÁC (Wav2Vec2 + MLP): {acc * 100:.2f}%")
print(f"📊 EER (Equal Error Rate): {eer * 100:.2f}%")
print("=" * 50)

print("\nBÁO CÁO CHI TIẾT:")
print(classification_report(y_val, y_pred, target_names=['Người Thật', 'Giọng AI']))

print("MA TRẬN NHẦM LẪN:")
print(confusion_matrix(y_val, y_pred))

# --- 6. Lưu Mô hình Tối thượng ---
SAVE_MODEL_DIR = 'models_saved'
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(SAVE_MODEL_DIR, 'mlp_wav2vec_model.pkl'))
joblib.dump(scaler, os.path.join(SAVE_MODEL_DIR, 'scaler_wav2vec.pkl'))
print(f"\n✅ Đã lưu mô hình Mạng Nơ-ron tại {SAVE_MODEL_DIR}/")