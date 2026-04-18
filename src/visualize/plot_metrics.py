import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_mlp_results(mlp_model, y_test, mlp_pred, mlp_acc, save_path=None):
    print("📊 Đang tạo bản vẽ báo cáo Seminar...")
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5)) 

    # --- BIỂU ĐỒ 1: QUÁ TRÌNH GIẢM LỖI (TRAINING LOSS) ---
    ax[0].plot(mlp_model.loss_curve_, color='red', linewidth=2)
    ax[0].set_title('Biểu đồ Sai số (Training Loss)', fontsize=14, fontweight='bold')
    ax[0].set_xlabel('Số vòng lặp (Epochs)')
    ax[0].set_ylabel('Sai số (Loss)')
    ax[0].grid(True, linestyle='--', alpha=0.7)

    # --- BIỂU ĐỒ 2: ĐỘ CHÍNH XÁC KIỂM TRA (VALIDATION ACCURACY) ---
    if hasattr(mlp_model, 'validation_scores_'):
        ax[1].plot(mlp_model.validation_scores_, color='blue', linewidth=2)
        ax[1].set_title('Độ chính xác Validation', fontsize=14, fontweight='bold')
        ax[1].set_xlabel('Số vòng lặp (Epochs)')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid(True, linestyle='--', alpha=0.7)

    # --- BIỂU ĐỒ 3: MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---
    cm = confusion_matrix(y_test, mlp_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax[2],
                xticklabels=['Đoán Thật', 'Đoán AI'], 
                yticklabels=['Thực tế Thật', 'Thực tế AI'])
    ax[2].set_title(f'Ma trận nhầm lẫn MLP\n(Accuracy: {mlp_acc*100:.2f}%)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    # Lưu ảnh thay vì chỉ hiển thị
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Đã lưu biểu đồ vào: {save_path}")
    else:
        plt.show()