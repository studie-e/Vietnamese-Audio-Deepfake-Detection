import os
from src.data_model.mlp_features import prepare_mlp_data
from src.models.train_mlp import train_and_evaluate_mlp
from src.visualize.plot_metrics import plot_mlp_results

def main():
    # 1. Tạo sẵn thư mục lưu trữ nếu chưa có
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models_saved', exist_ok=True)
    
    # 2. Đường dẫn tới file Numpy của bạn
    # Sửa lại thành đường dẫn trong máy tính của bạn (Dựa theo ảnh bạn cung cấp)
    X_path = r"D:\Projects\seminar\Vietnamese-Audio-Deepfake-Detection\data\fetures_model\MLP\X_super_data.npy"
    y_path = r"D:\Projects\seminar\Vietnamese-Audio-Deepfake-Detection\data\fetures_model\MLP\y_super_label.npy"
    
    # 3. Chạy xử lý dữ liệu
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_mlp_data(X_path, y_path)
    
    # 4. Chạy huấn luyện mô hình
    model_save_path = "models_saved/best_mlp.pkl"
    mlp_model, mlp_pred, mlp_acc = train_and_evaluate_mlp(
        X_train_scaled, y_train, X_test_scaled, y_test, save_path=model_save_path
    )
    
    # 5. Vẽ và lưu biểu đồ
    fig_save_path = "figures/seminar_mlp_report.png"
    plot_mlp_results(mlp_model, y_test, mlp_pred, mlp_acc, save_path=fig_save_path)
    
    print("\n🎉 HOÀN THÀNH TOÀN BỘ QUÁ TRÌNH!")

if __name__ == "__main__":
    main()