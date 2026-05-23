# Vispoofdb XAI

Thư mục này chứa phần giải thích mô hình dành riêng cho dữ liệu và pipeline của `Vispoofdb`.

## Thành phần chính

- `vispoofdb_xai.py`: tạo SHAP explanations cho các model đang dùng trong app.
- `__init__.py`: export `VispoofdbAudioXAI` và `extract_vispoofdb_features`.

## Hỗ trợ hiện tại

- `XGBoost + MFCC-Delta`: TreeSHAP cho model ensemble / saved model.
- `Wav2Vec2 + SVM`: KernelSHAP cho chế độ fallback single-model.
- Biểu đồ trọng số mô hình / summary để hiển thị trong Streamlit.
