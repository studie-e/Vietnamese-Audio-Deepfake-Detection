# Vietnamese Audio Deepfake Detection

Hệ thống phát hiện giọng nói deepfake tiếng Việt — **Viet-Guard**.

Ensemble 3 model đại diện 3 nhóm feature khác nhau, kết hợp với XAI (SHAP / Gradient Saliency) và đánh giá robustness dưới nhiều điều kiện nhiễu thực tế.

> **Seminar — Nhóm 17**

---

## Tổng quan hệ thống

### 3 chế độ inference trong web app

| Chế độ | Model | Feature | Ghi chú |
|---|---|---|---|
| **Ensemble (3 models)** | SVM | LFCC (40d) | Spectral cổ điển |
| | XGBoost | MFCC-Delta (480d) | Temporal |
| | MLP | Wav2Vec2 (768d) | Deep semantic *(optional)* |
| **Single model** | MLP | Wav2Vec2 | Standalone |
| **Deep Learning** | AASIST | Raw waveform | Graph Neural Network |

### Kết quả thực nghiệm (Test Unseen — 21,195 samples)

| Model | Accuracy | EER |
|---|---:|---:|
| SVM + LFCC | **96.79%** | 3.53% |
| MLP + MFCC | 90.42% | **1.84%** |
| XGBoost + MFCC-Delta | 84.71% | 15.39% |
| MLP + Wav2Vec2 | 76.55% | 23.69% |
| AASIST (Deep Learning) | 81.98% | — |

> Kết quả đầy đủ: `vispoofdb/experiments/results_summary_final.csv`

---

## Cấu trúc repository

```
Vietnamese-Audio-Deepfake-Detection/
│
├── app.py                        # Streamlit web app (3 chế độ)
├── aasist_inference.py           # AASIST wrapper + XAI
├── run_full_pipeline.py          # Chạy toàn bộ pipeline 7 bước
├── requirements.txt
│
├── vispoofdb/                    # Package chính
│   ├── data/                     # Dataset (gitignored — file lớn)
│   │   ├── raw/                  # Dữ liệu gốc (~700 MB)
│   │   ├── clean_data/           # Đã xử lý (~1.4 GB, 21K samples)
│   │   │   ├── real/             # ~14,000 files
│   │   │   ├── fake/             # ~7,195 files
│   │   │   └── metadata.csv      # Nhãn + split info
│   │   └── features_*/           # Features đã trích xuất (.npy)
│   │
│   ├── models/                   # Training scripts + ensemble
│   │   ├── ensemble_system.py    # VietGuardEnsemble (3 model)
│   │   ├── train_lfcc_svm.py
│   │   ├── train_svm.py
│   │   ├── train_mlp.py
│   │   ├── train_wav2vec.py
│   │   ├── train_xgboost.py
│   │   ├── train_tone_svm.py
│   │   ├── train_tone_xgboost.py
│   │   ├── train_tone_fusion.py
│   │   ├── train_aasist.py
│   │   └── aasist/               # AASIST model architecture
│   │
│   ├── models_saved/             # Model đã train (.pkl, .pth — gitignored)
│   │
│   ├── data_model/               # Feature extractors
│   │   ├── wav2vec2.py           # Wav2Vec2 embeddings
│   │   ├── tone_features.py      # F0, jitter, shimmer, HNR
│   │   ├── lfcc_svm.py
│   │   ├── mlp_features.py
│   │   ├── svm_features.py
│   │   └── xgboost_features.py
│   │
│   ├── data_processing/          # Tiền xử lý dữ liệu thô
│   │   ├── vidb_extract_mfcc.py
│   │   ├── vidb_extract_processing.py
│   │   ├── vispoofdb_clean_data.py
│   │   └── vispoofdb_generate_metadata.py
│   │
│   ├── scripts/                  # Pipeline scripts
│   │   ├── scripts_data_process.py       # Bước 1: Xử lý data
│   │   ├── scripts_feature_extract.py    # Bước 2: Trích xuất features
│   │   ├── scripts_train.py              # Bước 3: Train tất cả model
│   │   ├── experiment_fusion.py          # Bước 4: Fusion experiments
│   │   ├── plot_results.py               # Bước 5: Vẽ biểu đồ
│   │   ├── eval_noise_augmentation.py    # Bước 6: Đánh giá noise robustness
│   │   └── quantize.py                  # Bước 7: Tối ưu AASIST
│   │
│   ├── xai/                      # Explainability (SHAP + visualizer)
│   │   ├── vispoofdb_xai.py      # VispoofdbAudioXAI (TreeSHAP / KernelSHAP)
│   │   └── __init__.py
│   │
│   └── experiments/              # Kết quả thực nghiệm
│       ├── results_summary_final.csv
│       └── noise_eval/           # Kết quả đánh giá noise robustness
│
└── thu_nghiem/                   # Code & data thử nghiệm ban đầu (legacy)
    ├── src/                      # Codebase cũ (visualizer vẫn được dùng)
    ├── data/                     # Data thử nghiệm
    ├── figures/                  # Biểu đồ cũ
    └── models_saved/             # Model cũ
```

---

## Cài đặt môi trường

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

**Thư viện tùy chọn:**
```bash
pip install audiomentations    # Noise augmentation nâng cao
# ffmpeg cần cài riêng nếu muốn test MP3 codec
```

---

## Chạy toàn bộ pipeline

### Cách 1 — Chạy 1 lệnh (tất cả 7 bước)

```bash
python run_full_pipeline.py
```

Bỏ qua Wav2Vec2 nếu chưa trích xuất (tiết kiệm ~1-7 tiếng):
```bash
python run_full_pipeline.py --skip-wav2vec
```

Tự động tắt máy sau khi xong:
```bash
python run_full_pipeline.py --shutdown
```

### Cách 2 — Chạy từng bước thủ công

| Bước | Script | Mô tả | Thời gian ước tính |
|---|---|---|---:|
| 1 | `vispoofdb/scripts/scripts_data_process.py` | Download & xử lý data thô | ~10 phút |
| 2 | `vispoofdb/scripts/scripts_feature_extract.py` | Trích xuất MFCC, LFCC, Wav2Vec2 | ~1-7 giờ |
| 3 | `vispoofdb/scripts/scripts_train.py` | Train tất cả model | ~30 phút |
| 4 | `vispoofdb/scripts/experiment_fusion.py` | Thử nghiệm fusion | ~10 phút |
| 5 | `vispoofdb/scripts/plot_results.py` | Vẽ biểu đồ kết quả | ~2 phút |
| 6 | `vispoofdb/scripts/eval_noise_augmentation.py` | Đánh giá robustness với noise | ~15 phút |
| 7 | `vispoofdb/scripts/quantize.py` | Nén AASIST model | ~5 phút |

```bash
python vispoofdb/scripts/scripts_data_process.py
python vispoofdb/scripts/scripts_feature_extract.py
python vispoofdb/scripts/scripts_train.py
# ...
```

### Chỉ train lại model (đã có features)

```bash
# Tất cả model
python vispoofdb/scripts/scripts_train.py

# Bỏ qua Wav2Vec2
python vispoofdb/scripts/scripts_train.py --skip-wav2vec

# Bỏ cả Wav2Vec2 và Tone models
python vispoofdb/scripts/scripts_train.py --skip-wav2vec --skip-tone

# Train từng model riêng
python vispoofdb/models/train_lfcc_svm.py
python vispoofdb/models/train_xgboost.py
python vispoofdb/models/train_wav2vec.py
python vispoofdb/models/train_aasist.py
```

---

## Chạy Web App

```bash
streamlit run app.py
```

Mở trình duyệt: `http://localhost:8501`

**Tính năng:**
- **Ensemble (3 models)**: SVM+LFCC × XGBoost+MFCC-Delta × MLP+Wav2Vec2 — soft voting
- **Single model**: MLP + Wav2Vec2
- **Deep Learning**: AASIST (Graph Neural Network)
- **XAI tab**: SHAP (TreeSHAP cho XGBoost, KernelSHAP cho Wav2Vec2) hoặc Gradient Saliency (AASIST)

> Nếu Wav2Vec2 không tải được (offline), Ensemble tự động chạy với 2 model còn lại.

---

## Đánh giá Noise Robustness

Script `eval_noise_augmentation.py` **không phải data augmentation để train** — nó test mô hình đã train với âm thanh bị làm nhiễu, đánh giá xem model có còn nhận diện được không.

Các scenario được test:

| Scenario | Mô tả |
|---|---|
| `clean` | Âm thanh gốc (baseline) |
| `noise_snr_20/10/0` | Nhiễu trắng ở các mức SNR |
| `telephone` | Lọc dải tần điện thoại (300–3400 Hz) |
| `mp3_128k/64k/32k` | Nén MP3 (mô phỏng gửi qua Zalo/Messenger) |

Kết quả đã có (tóm tắt):

| Scenario | AASIST EER | Wav2Vec+MLP EER |
|---|---:|---:|
| clean | **0.00** | 0.12 |
| noise SNR 10 dB | 0.16 | **0.62** |
| telephone | **0.00** | 0.32 |
| mp3_32k | **0.00** | 0.18 |

→ AASIST rất robust. Wav2Vec+MLP bị ảnh hưởng nặng khi có noise.

Kết quả lưu tại: `vispoofdb/experiments/noise_eval/`

---

## Yêu cầu phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|---|---|---|
| RAM | 8 GB | 16 GB+ |
| GPU | Không bắt buộc | NVIDIA CUDA (train nhanh hơn) |
| Disk | 5 GB (chỉ features) | 10 GB+ (cả raw data) |

> AMD GPU không hỗ trợ CUDA — PyTorch sẽ tự fallback sang CPU.
> Wav2Vec2 chạy CPU mode trên RAM ~4–6 GB, khoảng 1–7 giờ tùy dataset.

---

## Log kết quả

Mỗi lần chạy `scripts_train.py` tạo ra file log tại thư mục gốc:
```
training_results_YYYYMMDD_HHMMSS.txt
```

Mỗi bước trong `run_full_pipeline.py` ghi log tại:
```
logs/<script_name>.log
```
