import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve


LOAD_DIR = os.path.join('data', 'features_model', 'svm')
X = np.load(os.path.join(LOAD_DIR, 'X_all.npy'))
y = np.load(os.path.join(LOAD_DIR, 'y_all.npy'))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)
model.fit(X_train_scaled, y_train)

# 5. Đánh giá độ chính xác
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Lấy probability của class 1

# Tính EER (Equal Error Rate)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = fpr[eer_idx]

acc = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🚀 ĐỘ CHÍNH XÁC TRÊN TẬP TEST (20%): {acc * 100:.2f}%")
print(f"📊 EER (Equal Error Rate): {eer * 100:.2f}%")
print("="*50)

print("\nBÁO CÁO CHI TIẾT:")
print(classification_report(y_test, y_pred, target_names=['Người Thật', 'Giọng AI']))

print("\nMA TRẬN NHẦM LẪN:")
print(confusion_matrix(y_test, y_pred))

os.makedirs('models_saved', exist_ok=True)
joblib.dump(model, 'models_saved/svm_voice_model.pkl')
joblib.dump(scaler, 'models_saved/scaler_final.pkl')
print("\n✅ Mô hình đã được lưu tại models_final/")