import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_mlp_data(X_path, y_path):
    print("📥 Đang tải tập dữ liệu Super Features...")
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"-> Kích thước X gốc: {X.shape} (Gồm {X.shape[0]} file, mỗi file có {X.shape[1]} đặc trưng)")
    
    # Chia 80% dữ liệu để Huấn luyện (Train), 20% để Kiểm tra (Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa (Scaling) dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("-> Kích thước tập Train:", X_train_scaled.shape)
    print("-> Kích thước tập Test:", X_test_scaled.shape)
    print("✅ Bước 1: Dữ liệu đã được chia và Chuẩn hóa xong!\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler