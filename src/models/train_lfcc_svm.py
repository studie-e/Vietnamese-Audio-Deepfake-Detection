import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Load dữ liệu LFCC ---
LOAD_DIR = 'data/features_lfcc'
print("Đang tải dữ liệu LFCC...")
X = np.load(os.path.join(LOAD_DIR, 'X_lfcc.npy'))
y = np.load(os.path.join(LOAD_DIR, 'y_lfcc.npy'))

# --- 2. Chia tập Train (Học) và Val (Thi thử) - Tỷ lệ 80/20 ---
# stratify=y đảm bảo chia đều tỷ lệ Thật/Giả ở cả 2 tập
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Số lượng file để Học (Train): {len(X_train)}")
print(f"Số lượng file để Thi (Val): {len(X_val)}")

# --- 3. Chuẩn hóa dữ liệu (Standardization) ---
# Bước sống còn của SVM: Ép các biến LFCC về cùng thang đo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --- 4. Khởi tạo và Huấn luyện SVM ---
print("\n--- ĐANG HUẤN LUYỆN SIÊU PHẲNG (SVM) TRÊN ĐẶC TRƯNG LFCC ---")
# Cấu hình RBF thường rất mạnh với các bài toán phi tuyến tính như âm thanh
model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 5. Đánh giá trực tiếp trên tập Val ---
print("\n--- KẾT QUẢ BÀI THI TRÊN TẬP VALIDATION (20%) ---")
y_pred = model.predict(X_val_scaled)

acc = accuracy_score(y_val, y_pred)
print("=" * 50)
print(f"🚀 ĐỘ CHÍNH XÁC (ACCURACY): {acc * 100:.2f}%")
print("=" * 50)

print("\nBÁO CÁO CHI TIẾT (Classification Report):")
print(classification_report(y_val, y_pred, target_names=['Người Thật', 'Giọng AI']))

print("MA TRẬN NHẦM LẪN (Confusion Matrix):")
# [Đoán đúng Người thật,  Đoán nhầm Người thành AI]
#[Đoán nhầm AI thành Người, Đoán đúng AI]
print(confusion_matrix(y_val, y_pred))

# --- 6. Lưu mô hình ---
SAVE_MODEL_DIR = 'models_saved'
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(SAVE_MODEL_DIR, 'svm_lfcc_model.pkl'))
joblib.dump(scaler, os.path.join(SAVE_MODEL_DIR, 'scaler_lfcc.pkl'))
print(f"\n✅ Đã lưu mô hình thành công vào thư mục {SAVE_MODEL_DIR}/")