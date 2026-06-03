# Viet-Guard — Phát hiện Giọng nói Deepfake Tiếng Việt

Hệ thống phát hiện giọng nói deepfake tiếng Việt sử dụng ensemble 3 nhóm đặc trưng âm học, kết hợp với mô hình học sâu AASIST và giải thích bằng XAI (SHAP / Gradient Saliency).

> Seminar — Nhóm 17 | Viện Trí tuệ Nhân tạo | Trường Đại học Công nghệ

---

## Kết quả thực nghiệm

**Dataset:** ViSpoofDB — 14.195 mẫu | Train 8.996 | Test Seen 2.599 | Test Unseen 2.600

### Mô hình cơ sở

| Mô hình | Feature | Test Seen Acc | Test Seen EER | Test Unseen Acc | Test Unseen EER |
|---|---|---:|---:|---:|---:|
| SVM | LFCC (40d) | 83.11% | 20.14% | 93.42% | 4.64% |
| MLP | Wav2Vec2 (768d) | 82.95% | 19.93% | 91.81% | 5.57% |
| MLP | MFCC (40d) | 82.38% | 21.36% | 91.31% | 8.36% |
| XGBoost | MFCC-Delta (480d) | 82.07% | 21.43% | 85.96% | 13.93% |
| SVM | Tone-Aware (24d) | 73.87% | 26.00% | 74.46% | 25.43% |
| **AASIST** | Raw waveform | **83.61%** | 21.14% | **97.19%** | **0.00%** |

### Ensemble & Fusion

| Phương pháp | Test Seen Acc | Test Seen EER | Test Unseen Acc | Test Unseen EER |
|---|---:|---:|---:|---:|
| Late Fusion (Soft Voting) | 84.11% | 18.50% | 94.62% | 4.93% |
| Stacking (Logistic Meta) | 84.76% | 18.07% | 94.69% | **2.64%** |

> Test Seen = Commercial TTS (FPT.AI, ElevenLabs) — thước đo thực tế hơn.  
> Test Unseen = gTTS — nguồn TTS đơn giản chưa thấy trong training.

---

## Cấu trúc repository

```
Vietnamese-Audio-Deepfake-Detection/
├── app.py                            # Streamlit web app
├── run_full_pipeline.py              # Chạy toàn bộ pipeline 7 bước
├── requirements.txt
├── README.md
│
├── vispoofdb/                        # Package chính
│   ├── data/                         # Dataset (lưu trên Google Drive)
│   │   └── clean_data/metadata.csv   # Nhãn + split (file này được track)
│   │
│   ├── models/                       # Training scripts + model architectures
│   │   ├── ensemble_system.py        # VietGuardEnsemble (3 model)
│   │   ├── aasist/
│   │   │   ├── aasist_inference.py   # AASIST inference wrapper + XAI
│   │   │   ├── train_aasist_model.py
│   │   │   └── models/baseline.py    # AASIST architecture
│   │   ├── train_lfcc_svm.py
│   │   ├── train_svm.py
│   │   ├── train_mlp.py
│   │   ├── train_wav2vec.py
│   │   ├── train_xgboost.py
│   │   └── train_aasist.py
│   │
│   ├── models_saved/                 # Model đã train (lưu trên Google Drive)
│   │
│   ├── scripts/                      # Pipeline scripts
│   │   ├── scripts_data_process.py   # Bước 1
│   │   ├── scripts_feature_extract.py # Bước 2
│   │   ├── scripts_train.py          # Bước 3 — train tất cả model
│   │   ├── experiment_fusion.py      # Bước 4 — fusion experiments
│   │   ├── plot_results.py           # Bước 5
│   │   ├── eval_noise_augmentation.py # Bước 6 — noise robustness
│   │   └── quantize.py               # Bước 7
│   │
│   ├── xai/vispoofdb_xai.py          # SHAP explainer
│   ├── experiments/                  # Kết quả CSV + biểu đồ
│   └── figures/                      # Biểu đồ ROC, DET, Confusion Matrix
│
└── thu_nghiem/                       # Code & data thử nghiệm ban đầu (legacy)
```

---

## Dataset

Data **không** được lưu trong git (quá lớn). Tải về từ Google Drive:

| Dataset | Link | Kích thước |
|---|---|---|
| **ViSpoofDB raw** (data chính) | [Google Drive](https://drive.google.com/drive/folders/1NZWOJi8g9nLfId1fSTkEc9Ay18P0c2LR?usp=sharing) | ~2.4 GB |
| **Thu nghiem raw** (data thử nghiệm) | [Google Drive](https://drive.google.com/drive/folders/1Dt2kEhL8IFRJ3cIQNiuddiVPKwF5bqLC?usp=sharing) | ~274 MB |

**Cấu trúc sau khi tải về:**
```
vispoofdb/data/
├── raw/
│   ├── real/       (~7.000 files WAV — VIVOS, VLSP)
│   └── fake/
│       ├── fpt/    (~2.000 files — FPT.AI TTS)
│       ├── viettel/
│       ├── elevenlabs/
│       ├── coqui/
│       └── gtts/   (test_unseen)
└── clean_data/
    ├── real/       (~14.000 files — augmented)
    ├── fake/       (~7.195 files)
    └── metadata.csv
```

---

## Cài đặt môi trường

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

---

## Hướng dẫn chạy

### Cách 1 — Chạy toàn bộ pipeline 1 lệnh

```bash
# Chạy đủ 7 bước (bao gồm Wav2Vec2 — mất 1-7 giờ)
python run_full_pipeline.py

# Bỏ Wav2Vec2 để tiết kiệm thời gian
python run_full_pipeline.py --skip-wav2vec

# Tự động tắt máy sau khi xong
python run_full_pipeline.py --shutdown
```

### Cách 2 — Chạy từng bước thủ công

| Bước | Lệnh | Mô tả | Thời gian |
|---:|---|---|---:|
| 1 | `python vispoofdb/scripts/scripts_data_process.py` | Xử lý & tạo metadata | ~5 phút |
| 2 | `python vispoofdb/scripts/scripts_feature_extract.py` | Trích xuất features | ~1–7 giờ |
| 3 | `python vispoofdb/scripts/scripts_train.py` | Train tất cả model | ~30 phút |
| 4 | `python vispoofdb/scripts/experiment_fusion.py` | Fusion experiments | ~15 phút |
| 5 | `python vispoofdb/scripts/plot_results.py` | Vẽ biểu đồ | ~2 phút |
| 6 | `python vispoofdb/scripts/eval_noise_augmentation.py` | Đánh giá noise | ~15 phút |
| 7 | `python vispoofdb/scripts/quantize.py` | Nén AASIST | ~5 phút |

**Bỏ qua Wav2Vec2 (nếu chưa có features):**
```bash
python vispoofdb/scripts/scripts_train.py --skip-wav2vec
```

### Train riêng từng model

```bash
python vispoofdb/models/train_lfcc_svm.py       # SVM + LFCC (~30 giây)
python vispoofdb/models/train_svm.py            # SVM + MFCC
python vispoofdb/models/train_mlp.py            # MLP + MFCC
python vispoofdb/models/train_xgboost.py        # XGBoost + MFCC-Delta (~20 phút)
python vispoofdb/models/train_wav2vec.py        # MLP + Wav2Vec2
python vispoofdb/models/aasist/train_aasist_model.py  # AASIST (~2 giờ trên T4 GPU)
```

---

## Chạy trên Google Colab (T4 GPU)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo
import os
os.chdir('/content/drive/MyDrive')
!git clone https://github.com/studie-e/Vietnamese-Audio-Deepfake-Detection.git
os.chdir('Vietnamese-Audio-Deepfake-Detection')

# 3. Cài đặt dependencies
!pip install -r requirements.txt -q

# 4. Chạy pipeline (data đã có trên Drive)
!python vispoofdb/scripts/scripts_train.py --skip-wav2vec
```

> Data raw cần upload lên Google Drive trước theo cấu trúc `vispoofdb/data/` như trên.

---

## Chạy Web App

```bash
streamlit run app.py
```

Mở: `http://localhost:8501`

**3 chế độ:**
- **Ensemble (3 models):** SVM+LFCC × XGBoost+MFCC-Delta × MLP+Wav2Vec2 — soft voting
- **Single model:** MLP + Wav2Vec2
- **Deep Learning:** AASIST (97.19% Test Unseen)

**XAI tab:**
- TreeSHAP / KernelSHAP cho sklearn models
- Gradient Saliency cho AASIST

> Nếu Wav2Vec2 không tải được (offline/thiếu model), Ensemble tự động fallback sang 2 model.

---

## Yêu cầu phần cứng

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA CUDA (cho AASIST) |
| Disk | 3 GB (chỉ features) | 6 GB (đủ pipeline) |

> AMD GPU không hỗ trợ CUDA — PyTorch fallback CPU.  
> Wav2Vec2 feature extraction: ~1–7 giờ trên CPU với 14K files.
