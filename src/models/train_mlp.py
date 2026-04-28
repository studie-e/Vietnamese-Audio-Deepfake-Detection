# File: src/models/train_mlp.py
import os
import numpy as np
import joblib # Thư viện chuẩn để lưu model scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Định nghĩa đường dẫn
FEATURES_DIR = "data/features_model/mlp"
MODEL_SAVE_DIR = "models_saved"

# Đảm bảo thư mục lưu model tồn tại
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_model():
    print("Loading features...")
    # Load dữ liệu .npy
    X_path = os.path.join(FEATURES_DIR, "X_mlp.npy")
    y_path = os.path.join(FEATURES_DIR, "y_mlp.npy")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Error: Feature files not found. Run mlp_features.py first.")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    
    # Chia tập dữ liệu: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Khởi tạo mô hình MLP
    # Bạn có thể tuning các tham số (hidden_layer_sizes, learning_rate, max_iter,...) sau
    print("\nTraining MLP model...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128),  # Cấu trúc hình phễu 2 lớp
        activation='relu',
        solver='adam',
        alpha=0.2,                    # Hình phạt vừa phải chống Học vẹt
        learning_rate_init=0.001,      
        max_iter=500,                  
        early_stopping=True,           # Bật tự động dừng để tối ưu hóa
        n_iter_no_change=20,           
        random_state=42,
        verbose=True                   # In quá trình học ra màn hình
    )
    
    # Huấn luyện
    mlp_model.fit(X_train, y_train)
    
    # Đánh giá trên tập test
    print("\nEvaluating model...")
    y_pred = mlp_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    # 0 = Real, 1 = AI
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "AI (1)"]))
    
    # Lưu model ra file .pkl
    model_save_path = os.path.join(MODEL_SAVE_DIR, "best_mlp.pkl")
    joblib.dump(mlp_model, model_save_path)
    print(f"\nModel successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train_model()