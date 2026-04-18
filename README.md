# Nhận diện Giọng nói Deepfake (Vietnamese Audio Deepfake Detection)

Dự án sử dụng Mạng Nơ-ron nhân tạo (MLP) để phân loại và phát hiện giọng nói thật và giọng nói do AI tạo ra (Deepfake) dựa trên các đặc trưng âm thanh.

## Cấu trúc dự án
Dự án được chia module hóa để dễ dàng quản lý:
- `data/`: Chứa dữ liệu âm thanh gốc và các mảng đặc trưng numpy (`.npy`).
- `src/`: Chứa mã nguồn chính (trích xuất đặc trưng, huấn luyện mô hình MLP, vẽ biểu đồ).
- `models_saved/`: Nơi lưu trữ mô hình sau khi huấn luyện xong (`.pkl`).
- `figures/`: Nơi lưu các biểu đồ báo cáo đánh giá mô hình.
- `app.py`: File chạy chính (Main Pipeline) của toàn bộ dự án.

## Hướng dẫn cài đặt và chạy thử

**Bước 1: Cài đặt thư viện**
Mở terminal và chạy lệnh sau để cài đặt các công cụ cần thiết:
```bash
pip install -r requirements.txt