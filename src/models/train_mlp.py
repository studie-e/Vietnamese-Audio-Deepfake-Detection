from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_and_evaluate_mlp(X_train, y_train, X_test, y_test, save_path=None):
    print("🧠 BẮT ĐẦU HUẤN LUYỆN MẠNG NƠ-RON (MLP CLASSIFIER)")

    # Khởi tạo mô hình giống hệt code của bạn
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

    # Bắt đầu cho AI học
    mlp_model.fit(X_train, y_train)

    # Làm bài kiểm tra và tính điểm
    mlp_pred = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)

    print("\n" + "="*50)
    print(f"⭐ ĐỘ CHÍNH XÁC CHỐT HẠ (MLP): {mlp_acc * 100:.2f}%")
    print("="*50)
    print(classification_report(y_test, mlp_pred, target_names=['Giọng Thật (0)', 'Giọng AI (1)']))

    # Lưu mô hình AI đã học xong
    if save_path:
        joblib.dump(mlp_model, save_path)
        print(f"✅ Đã lưu mô hình tại: {save_path}\n")

    return mlp_model, mlp_pred, mlp_acc