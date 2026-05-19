# Báo cáo Seminar: Hệ thống nhận diện giọng nói AI (AASIST)

Dự án phát hiện giọng nói giả mạo (Deepfake Audio / AI Generated Speech) được xây dựng bám sát theo kiến trúc của bài báo khoa học: **"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"**.

## Tính năng nổi bật của dự án
- **Front-end:** Trích xuất đặc trưng trực tiếp từ âm thanh thô (Raw Waveform) bằng lớp `SincNet` kết hợp các khối `Residual Blocks` (Không dùng Mel-Spectrogram).
- **Back-end:** Sử dụng Mạng đồ thị chú ý (`Graph Attention Networks`) để tự động tìm kiếm các dấu vết giả mạo (artefacts) tinh vi trên cả dải phổ và thời gian.
- **Hiệu năng cao - Dung lượng siêu nhẹ:** Mô hình chỉ đạt kích thước vài trăm KB nhưng đem lại độ chính xác cực cao (Đạt Validation Accuracy đỉnh: **95.40%**).
- **Live Demo:** Tích hợp script bốc thăm thử nghiệm ngẫu nhiên, chứng minh AI không học vẹt.

## Cấu trúc dự án
- `/dataset`: Chứa các mẫu âm thanh Train/Test.
- `/models/baseline.py`: Chứa toàn bộ kiến trúc lõi (SincNet + Graph Module).
- `train.py`: Vòng lặp huấn luyện, tích hợp cơ chế tự động lưu Best Model.
- `predict.py`: Script Demo dự đoán thực tế.
- `aasist_best_model.pth`: Trọng số của mô hình đã huấn luyện thành công.

## Chạy nghiệm thu (Demo Live)

**1. Mở Terminal / Command Prompt tại thư mục dự án**

**2. Chạy lệnh bốc thăm dự đoán ngẫu nhiên:**
```bash
python predict.py