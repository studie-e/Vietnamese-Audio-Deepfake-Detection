<div align="center">

# Viet-Guard
### Vietnamese Audio Deepfake Detection

*Phát hiện giọng nói deepfake tiếng Việt sử dụng Ensemble đặc trưng âm học đa nhóm và học sâu AASIST*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

*Seminar — Nhóm 17 | Viện Trí tuệ Nhân tạo | Trường Đại học Công nghệ*

---

</div>

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Mục tiêu nghiên cứu](#2-mục-tiêu-nghiên-cứu)
3. [Dataset](#3-dataset)
4. [Kiến trúc hệ thống](#4-kiến-trúc-hệ-thống)
5. [Các mô hình](#5-các-mô-hình)
6. [XAI — Giải thích mô hình](#6-xai--giải-thích-mô-hình)
7. [Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
8. [Cài đặt](#8-cài-đặt)
9. [Hướng dẫn chạy](#9-hướng-dẫn-chạy)
10. [Web App](#10-web-app)
11. [Giới hạn và hướng phát triển](#11-giới-hạn-và-hướng-phát-triển)

---

## 1. Tóm tắt

Sự bùng nổ của công nghệ Text-to-Speech (TTS) và Voice Conversion (VC) đặt ra thách thức nghiêm trọng trong việc xác thực giọng nói số. **Viet-Guard** là hệ thống phát hiện giọng nói deepfake tiếng Việt được xây dựng theo hướng tiếp cận đa đặc trưng:

- Kết hợp 3 nhóm đặc trưng âm học: **Spectral (LFCC)**, **Temporal (MFCC-Delta)**, **Semantic (Wav2Vec2)**
- Triển khai **Soft-Voting Ensemble** để tổng hợp quyết định
- Tích hợp **AASIST** (mô hình học sâu xử lý raw waveform) làm so sánh baseline mạnh
- Cung cấp **giải thích mô hình** (XAI) thông qua SHAP và Gradient-based Saliency

---

## 2. Mục tiêu nghiên cứu

| Mục tiêu | Mô tả |
|---|---|
| **So sánh đặc trưng** | Đánh giá hiệu quả các nhóm feature (MFCC, LFCC, Wav2Vec2, Tone-Aware) trên tiếng Việt |
| **Ensemble learning** | Khảo sát Soft Voting, Stacking, Early Fusion trên bài toán phát hiện deepfake |
| **Generalization** | Kiểm tra khả năng tổng quát hoá trên nguồn TTS chưa thấy trong training |
| **Robustness** | Đánh giá độ bền vững dưới các điều kiện nhiễu thực tế (white noise, MP3, telephone) |
| **Interpretability** | Cung cấp giải thích quyết định thông qua XAI để tăng tính tin cậy |

---

## 3. Dataset

### ViSpoofDB — Vietnamese Spoof Database

Dataset tự xây dựng gồm **14.195 mẫu** tiếng Việt, cân bằng giữa giọng thật và giọng AI:

| Nhãn | Nguồn | Số mẫu |
|---|---|---:|
| **Real** | VIVOS, VLSP (người đọc tự nhiên) | ~7.000 |
| **Fake** | FPT.AI, Viettel TTS, ElevenLabs, Coqui TTS | ~5.600 |
| **Fake (unseen)** | gTTS — hệ thống TTS chưa thấy trong train | ~1.600 |

**Phân chia tập dữ liệu:**

```
Train:        8.996 mẫu  (63%)
Test Seen:    2.599 mẫu  (18%) — nguồn AI đã xuất hiện khi train
Test Unseen:  2.600 mẫu  (18%) — nguồn AI hoàn toàn mới
```

**Tải dữ liệu:**

| Phần | Link | Dung lượng |
|---|---|---|
| ViSpoofDB raw data | [Google Drive](https://drive.google.com/drive/folders/1NZWOJi8g9nLfId1fSTkEc9Ay18P0c2LR?usp=sharing) | ~2.4 GB |
| Thu nghiem raw data | [Google Drive](https://drive.google.com/drive/folders/1Dt2kEhL8IFRJ3cIQNiuddiVPKwF5bqLC?usp=sharing) | ~274 MB |

> Data không được lưu trong git. Đặt theo cấu trúc `vispoofdb/data/raw/{real,fake}/`.

---

## 4. Kiến trúc hệ thống

```
                    Audio Input (.wav / .mp3)
                           |
              +------------+------------+
              |                         |
     Feature Extraction            Raw Waveform
              |                         |
    +---------+---------+               |
    |         |         |               |
  LFCC     MFCC-Δ   Wav2Vec2        AASIST
  (40d)   (480d)    (768d)        (Deep Learning)
    |         |         |
  SVM     XGBoost    MLP
    |         |         |
    +---------+---------+
              |
        Soft-Voting
              |
         Prediction
     (Real / Deepfake)
              |
         XAI Explanation
      (SHAP / Saliency)
```

**Pipeline 7 bước:**

```
Bước 1: Data Processing    → augmentation, chuẩn hoá, metadata
Bước 2: Feature Extraction → LFCC, MFCC-Δ, Wav2Vec2, Tone-Aware, raw
Bước 3: Model Training     → 6 mô hình cơ sở + AASIST
Bước 4: Fusion Experiments → Late Fusion, Stacking, Early Fusion
Bước 5: Visualization      → ROC, DET, Confusion Matrix
Bước 6: Noise Evaluation   → robustness dưới white noise / MP3 / telephone
Bước 7: Quantization       → nén AASIST (dynamic INT8)
```

---

## 5. Các mô hình

### Mô hình cơ sở

| Mô hình | Đặc trưng | Chiều | Ghi chú |
|---|---|---:|---|
| SVM (RBF) | LFCC | 40 | Linear Cepstral — tổng quát hoá tốt |
| SVM (RBF) | MFCC | 40 | Mel-scale cepstral |
| MLP | MFCC | 40 | 3 hidden layers, early stopping |
| XGBoost | MFCC + Δ + ΔΔ | 480 | Hyperparameter tuning + early stopping |
| MLP | Wav2Vec2 | 768 | Self-supervised pre-trained features |
| SVM | Tone-Aware | 24 | F0, Jitter, Shimmer, HNR |
| **AASIST** | Raw waveform | — | Graph Attention Network |

### Ensemble của dự án

**VietGuardEnsemble** — Soft-Voting trên 3 nhóm đặc trưng:

```
Group 1: SVM + LFCC         (Spectral — Anti-Spoofing features)
Group 2: XGBoost + MFCC-Δ   (Temporal — Dynamic features)
Group 3: MLP + Wav2Vec2     (Semantic — Self-supervised deep features)
```

Quyết định cuối = trung bình xác suất của 3 model. Tự động fallback nếu một model lỗi.

---

## 6. XAI — Giải thích mô hình

| Phương pháp | Áp dụng cho | Giải thích |
|---|---|---|
| **TreeSHAP** | XGBoost | Feature importance chính xác, hiệu quả cao |
| **KernelSHAP** | SVM, MLP | Model-agnostic, chậm hơn |
| **Gradient Saliency** | AASIST | Vùng waveform quan trọng với quyết định |

---

## 7. Cấu trúc thư mục

```
Vietnamese-Audio-Deepfake-Detection/
├── app.py                             # Streamlit demo app
├── run_full_pipeline.py               # Orchestrator 7 bước
├── requirements.txt
├── README.md
│
├── vispoofdb/                         # Package nghiên cứu chính
│   ├── data/                          # Dữ liệu (lưu trên Google Drive)
│   │   └── clean_data/metadata.csv    # File này được track trong git
│   │
│   ├── models/                        # Model architecture + training
│   │   ├── ensemble_system.py         # VietGuardEnsemble
│   │   ├── aasist/
│   │   │   ├── aasist_inference.py    # Inference wrapper + XAI
│   │   │   ├── train_aasist_model.py  # Training script
│   │   │   └── models/baseline.py    # AASIST architecture (Graph Attention)
│   │   ├── train_lfcc_svm.py
│   │   ├── train_svm.py
│   │   ├── train_mlp.py
│   │   ├── train_xgboost.py
│   │   ├── train_wav2vec.py
│   │   └── train_aasist.py           # AASIST wrapper
│   │
│   ├── models_saved/                  # Checkpoints (lưu trên Google Drive)
│   │
│   ├── scripts/                       # Pipeline scripts
│   │   ├── scripts_data_process.py    # Bước 1
│   │   ├── scripts_feature_extract.py # Bước 2
│   │   ├── scripts_train.py           # Bước 3
│   │   ├── experiment_fusion.py       # Bước 4
│   │   ├── plot_results.py            # Bước 5
│   │   ├── eval_noise_augmentation.py # Bước 6
│   │   └── quantize.py               # Bước 7
│   │
│   ├── xai/vispoofdb_xai.py          # SHAP explainer module
│   ├── experiments/                   # CSV kết quả
│   └── figures/                       # Biểu đồ ROC, DET, CM
│
└── thu_nghiem/                        # Prototype ban đầu (legacy)
```

---

## 8. Cài đặt

**Yêu cầu:** Python 3.10+

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# Cài đặt dependencies
pip install -r requirements.txt
```

**Yêu cầu phần cứng:**

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA CUDA (AASIST nhanh hơn ~20x) |
| Disk | 3 GB | 6 GB (đủ toàn bộ pipeline) |

> AMD GPU không hỗ trợ CUDA — PyTorch tự động fallback sang CPU.

---

## 9. Hướng dẫn chạy

### Toàn bộ pipeline (1 lệnh)

```bash
# Đủ 7 bước
python run_full_pipeline.py

# Bỏ qua Wav2Vec2 feature extraction (tiết kiệm 1–7 giờ nếu đã có features)
python run_full_pipeline.py --skip-wav2vec

# Tự động tắt máy sau khi hoàn thành
python run_full_pipeline.py --shutdown
```

### Từng bước riêng lẻ

```bash
python vispoofdb/scripts/scripts_data_process.py      # Bước 1 (~5 phút)
python vispoofdb/scripts/scripts_feature_extract.py   # Bước 2 (~1–7 giờ)
python vispoofdb/scripts/scripts_train.py             # Bước 3 (~30 phút)
python vispoofdb/scripts/experiment_fusion.py         # Bước 4 (~15 phút)
python vispoofdb/scripts/plot_results.py              # Bước 5 (~2 phút)
python vispoofdb/scripts/eval_noise_augmentation.py   # Bước 6 (~15 phút)
python vispoofdb/scripts/quantize.py                  # Bước 7 (~5 phút)
```

### Train từng model riêng

```bash
python vispoofdb/models/train_lfcc_svm.py
python vispoofdb/models/train_xgboost.py
python vispoofdb/models/train_wav2vec.py
python vispoofdb/models/aasist/train_aasist_model.py   # Cần GPU để nhanh
```

### Chạy trên Google Colab (T4 GPU)

```python
# Mount Google Drive (data phải có sẵn trên Drive)
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive')

# Clone repo
!git clone https://github.com/studie-e/Vietnamese-Audio-Deepfake-Detection.git
os.chdir('Vietnamese-Audio-Deepfake-Detection')

# Cài dependencies
!pip install -r requirements.txt -q

# Chạy training (bỏ Wav2Vec2 nếu chưa có features)
!python vispoofdb/scripts/scripts_train.py --skip-wav2vec

# Hoặc train AASIST riêng (thấy epoch progress trực tiếp)
!python vispoofdb/models/aasist/train_aasist_model.py
```

---

## 10. Web App

```bash
streamlit run app.py
# Mở: http://localhost:8501
```

**Các chế độ phát hiện:**

| Chế độ | Mô tả |
|---|---|
| Ensemble (3 models) | SVM+LFCC × XGBoost+MFCC-Δ × MLP+Wav2Vec2, soft-voting |
| Single Model | MLP + Wav2Vec2 |
| Deep Learning | AASIST — raw waveform, gradient saliency |

> Nếu Wav2Vec2 không tải được, Ensemble tự động fallback sang 2 model còn lại.

---

## 11. Giới hạn và hướng phát triển

### Giới hạn hiện tại

| Vấn đề | Mô tả |
|---|---|
| White noise | Cả 2 model chính đều sụt xuống ~50% (random guess) khi SNR thấp |
| AASIST inference | Chậm trên CPU (~3–5 giây/file), cần GPU cho production |
| Tone-Aware features | Kém hiệu quả (~74%) so với spectral features (~83–93%) |
| Test Unseen bias | gTTS là TTS đơn giản nên kết quả unseen cao hơn thực tế |

### Hướng phát triển

- Augmentation với white noise trong quá trình training để tăng robustness
- Fine-tune Wav2Vec2 trực tiếp trên tiếng Việt (thay vì dùng frozen features)
- Thu thập thêm dữ liệu từ Commercial TTS mới (VALL-E X, VoiceBox, Bark)
- Triển khai real-time detection với WebRTC
- Thêm phân tích speaker diarization để phát hiện voice swap trong hội thoại
