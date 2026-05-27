# Vietnamese-Audio-Deepfake-Detection

Project: Phát hiện deepfake giọng nói tiếng Việt — tập hợp feature extractors, mô hình (SVM, XGBoost, MLP), và pipeline thử nghiệm (fusion / stacking).

**Nội dung chính**
- Mã trích xuất đặc trưng: MFCC, LFCC, Tone-Aware (F0, jitter, shimmer, HNR, ...), Wav2Vec2 embeddings.
- Mô hình huấn luyện: `SVM`, `XGBoost`, `MLP` và các chiến lược kết hợp: late-fusion, early-fusion, stacking.
- Scripts để chạy toàn bộ pipeline và so sánh: [vispoofdb/scripts/experiment_fusion.py](vispoofdb/scripts/experiment_fusion.py#L1).

## Cấu trúc repository (tóm tắt)
- `vispoofdb/data/` : feature files được lưu (MFCC, LFCC, tone, wav2vec, ...).
- `vispoofdb/models/` : scripts huấn luyện (train_*.py).
- `vispoofdb/models_saved/` : mô hình và scaler đã lưu (`*.pkl`).
- `vispoofdb/scripts/experiment_fusion.py` : script chạy tuần tự các thử nghiệm (baseline, swap-features, late-fusion, early-fusion, stacking) và xuất `vispoofdb/experiments/results_summary.csv`.
- `vispoofdb/data/clean_data/metadata.csv` : metadata (file paths, labels, splits).

## Yêu cầu (virtualenv)
Install dependencies (từ `requirements.txt` nếu cần):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng Wav2Vec2 bạn cần có PyTorch + transformers; chạy trên GPU khi có thể.

## ⚡ Quick Start: Hướng dẫn thứ tự chạy

Để chạy toàn bộ project từ đầu, thực hiện theo thứ tự sau:

### **Bước 1: Setup Environment**
```bash
# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### **Bước 2: Kiểm tra dữ liệu**
```bash
# Xác nhận dữ liệu clean_data/ và metadata.csv đã tồn tại
# (Nếu chưa có: sao chép từ raw data hoặc download dataset)
# Đường dẫn cần kiểm tra:
#   ✓ vispoofdb/data/clean_data/metadata.csv
#   ✓ vispoofdb/data/clean_data/real/ (~14K files)
#   ✓ vispoofdb/data/clean_data/fake/ (~7K files)
```

### **Bước 3: Trích xuất Features (nếu chưa có)**
Chỉ cần chạy 1 lần duy nhất:

```bash
# Tạo MFCC features (~5 phút)
python vispoofdb/data_processing/vidb_extract_mfcc.py

# Tạo Wav2Vec2 embeddings (~30 phút trên CPU)
python vispoofdb/data_model/wav2vec2.py

# Tạo LFCC features (~5 phút)
python vispoofdb/data_model/lfcc_svm.py

# Tạo Tone-Aware features (~10 phút)
python vispoofdb/data_model/tone_features.py
```

**⏱️ Tổng thời gian:** ~45-50 phút (lần đầu). Lần sau load từ `.npy` files sẽ nhanh hơn.

### **Bước 4: Huấn luyện Mô hình**

**Option A: Chạy toàn bộ 9 mô hình (KHUYÊN DÙNG)**
```bash
python vispoofdb/scripts/scripts_train.py
```

⏱️ **Thời gian:** ~20-30 phút (tuỳ CPU)

**Kết quả:** 
- 8 mô hình sklearn lưu tại: `vispoofdb/models_saved/` (`.pkl` files)
- AASIST lưu tại: `vispoofdb/models_saved/aasist_best_model.pth` (`.pth` file)

**Option B: Huấn luyện từng mô hình riêng lẻ**
```bash
# Ví dụ: Chỉ train SVM + LFCC
python vispoofdb/models/train_lfcc_svm.py         # ~1 phút

# Hoặc train AASIST riêng
python vispoofdb/models/train_aasist.py            # ~10 phút
```

### **Bước 5: Tạo Bảng Kết quả**
```bash
python create_final_results_table.py
```

**Kết quả lưu tại:**
- `vispoofdb/experiments/results_summary_final.csv`
- `vispoofdb/experiments/results_summary_final.txt`

### **Bước 6: Chạy Web App (Optional)**
```bash
streamlit run app.py
```

Mở: `http://localhost:8501` → Upload file audio → Xem kết quả + XAI

### **Bước 7: Đánh giá Noise Robustness (Optional)**
```bash
python vispoofdb/scripts/eval_noise_augmentation.py --n-samples 50 --model-type ensemble
```

---

### 📊 **Timeline tổng cộng:**
| Bước | Tác vụ | Thời gian |
|---|---|---:|
| 1 | Setup environment | ~5 min |
| 2 | Kiểm tra dữ liệu | ~1 min |
| 3 | Trích xuất features | ~50 min |
| 4 | Huấn luyện 9 mô hình | ~30 min |
| 5 | Tạo bảng kết quả | ~2 min |
| **TỔNG** | **Lần đầu** | **~90 min** |
| - | Lần sau (skip bước 3) | ~35 min |

---

## 🔄 Các Scenario Khác

### Scenario 1: **Chỉ test inference (không train lại)**
Nếu mô hình đã huấn luyện sẵn:

```bash
# Chạy web app với mô hình có sẵn
streamlit run app.py
```

### Scenario 2: **Train lại chỉ 1-2 mô hình**
Ví dụ: Chỉ train SVM + LFCC:

```bash
python vispoofdb/models/train_lfcc_svm.py
python create_final_results_table.py
```

### Scenario 3: **Deploy trên production**
```bash
# 1. Tối ưu AASIST model
python vispoofdb/scripts/quantize.py \
  --model-path vispoofdb/models_saved/aasist_best_model.pth \
  --n-benchmark 50

# 2. Đánh giá noise robustness
python vispoofdb/scripts/eval_noise_augmentation.py \
  --n-samples 100 --model-type aasist

# 3. Chạy app
streamlit run app.py
```

### Scenario 4: **Benchmarking & Comparison**
So sánh hiệu suất dưới nhiễu:

```bash
# Test ensemble dưới nhiễu
python vispoofdb/scripts/eval_noise_augmentation.py \
  --n-samples 200 --model-type ensemble --augmentor audiomentations
```

### Scenario 5: **Re-train từ đầu (mô hình mới)**
```bash
# Xóa mô hình cũ (nếu cần)
rm vispoofdb/models_saved/*.pkl vispoofdb/models_saved/*.pth

# Chạy lại pipeline
python vispoofdb/scripts/scripts_train.py
```

---

## Chuẩn bị dữ liệu

### Cấu trúc dữ liệu
Project sử dụng 2 loại dữ liệu chính:

**1. Raw data** (`vispoofdb/data/raw/`):
- Dữ liệu gốc chưa xử lý
- Cấu trúc: `raw/real/` (~7,000 files), `raw/ai/` (synthetic deepfakes)
- Kích thước: ~700 MB

**2. Clean data** (`vispoofdb/data/clean_data/`):
- Dữ liệu đã augment (nhân đôi real audio)
- Cấu trúc: `clean_data/real/` (~14,000 files), `clean_data/fake/` (~7,195 files)
- **Metadata**: `clean_data/metadata.csv` (21,195 samples)
  - Cột: `filename`, `split` (train/test_seen/test_unseen), `label` (real/fake)
- Kích thước: ~1.4 GB

### Trích xuất đặc trưng (nếu chưa có)
Nếu muốn tạo lại feature files từ đầu:

```bash
# Tạo MFCC features
python vispoofdb/data_processing/vidb_extract_mfcc.py

# Tạo Wav2Vec2 embeddings
python vispoofdb/data_model/wav2vec2.py

# Tạo LFCC features
python vispoofdb/data_model/lfcc_svm.py

# Tạo Tone-Aware features (F0, jitter, shimmer, ...)
python vispoofdb/data_model/tone_features.py
```

**Kết quả features lưu tại:** `vispoofdb/data/features_*/` (`.npy` files)

## Chạy huấn luyện / thử nghiệm

### 1. Chạy toàn bộ pipeline (9 mô hình: 8 sklearn + AASIST)
**Cách nhanh nhất để huấn luyện tất cả 9 mô hình tuần tự:**

```bash
# Kích hoạt venv
.venv\Scripts\activate

# Chạy pipeline
python vispoofdb/scripts/scripts_train.py
```

**Kết quả:** Mô hình lưu tại `vispoofdb/models_saved/` (**.pkl, .pth files**)

### 2. Huấn luyện từng mô hình riêng lẻ
Nếu muốn huấn luyện/test riêng một mô hình:

```bash
# SVM models
python vispoofdb/models/train_lfcc_svm.py          # SVM + LFCC
python vispoofdb/models/train_svm.py               # SVM + MFCC
python vispoofdb/models/train_tone_svm.py          # SVM + Tone-Aware

# MLP models
python vispoofdb/models/train_mlp.py               # MLP + MFCC
python vispoofdb/models/train_wav2vec.py           # MLP + Wav2Vec2

# XGBoost models
python vispoofdb/models/train_xgboost.py           # XGBoost + MFCC
python vispoofdb/models/train_tone_xgboost.py      # XGBoost + Tone-Aware

# Deep Learning
python vispoofdb/models/train_aasist.py            # AASIST (Deep Learning)
```

### 3. Tạo bảng thống kê kết quả
Sau khi huấn luyện xong, tạo bảng so sánh kết quả:

```bash
python create_final_results_table.py
```

**Kết quả:** Bảng lưu tại:
- `vispoofdb/experiments/results_summary_final.csv` (định dạng CSV)
- `vispoofdb/experiments/results_summary_final.txt` (định dạng text)

### 4. Chạy Streamlit Web App
Giao diện web để test phát hiện deepfake real-time:

```bash
streamlit run app.py
```

**Tính năng:**
- 🎯 **3 chế độ**: Ensemble (5 models) | Single Model (SVM Wav2Vec) | Deep Learning (AASIST)
- 📊 **Detection**: Upload file → Phát hiện AI/Real
- 🧠 **XAI**: Giải thích quyết định bằng SHAP (sklearn models) hoặc Gradient-based saliency (AASIST)

Ứng dụng mở tại: `http://localhost:8501`

### 5. Đánh giá robust với noise
Kiểm tra hiệu suất mô hình trước nhiễu âm (background noise, codec artifacts, ...):

```bash
python vispoofdb/scripts/eval_noise_augmentation.py --n-samples 100 --model-type ensemble
```

### 6. Tối ưu mô hình (Quantize + Pruning)
Nén mô hình AASIST để deployment:

```bash
python vispoofdb/scripts/quantize.py --model-path vispoofdb/models_saved/aasist_best_model.pth --n-benchmark 50
```

## Báo cáo kết quả (Kết quả cuối cùng)

### Hiệu suất 9 mô hình (Test_Unseen)
Kết quả từ lần training gần nhất (21,195 ViSpoofDB samples):

| # | Mô hình | Test_Seen | Test_Unseen | EER_Seen | EER_Unseen |
|---|---|---:|---:|---:|---:|
| 1 | **SVM + LFCC** | 92.82% | **96.79%** ⭐ | 5.92% | **3.53%** ⭐ |
| 2 | SVM + MFCC | 82.71% | 88.81% | 14.74% | 3.67% |
| 3 | MLP + MFCC | 83.73% | 90.42% | 13.52% | **1.84%** ⭐⭐ |
| 4 | XGBoost + MFCC | 87.04% | 84.71% | 11.76% | 15.39% |
| 5 | MLP + Wav2Vec2 | 95.26% | 76.55% | 5.00% | 23.69% |
| 6 | SVM + Tone-Aware | 83.29% | 76.80% | 17.00% | 22.88% |
| 7 | XGBoost + Tone-Aware | 82.74% | 74.36% | 17.27% | 25.56% |
| 8 | SVM + Fusion | 84.94% | 87.33% | 12.80% | 6.10% |
| 9 | **AASIST** (Deep Learning) | 82.0% | 81.98% | N/A | N/A |

**Nhận xét:**
- ✅ **Best Accuracy**: SVM + LFCC (96.79%)
- ✅ **Best EER**: MLP + MFCC (1.84%)
- ✅ **AASIST**: Baseline tốt (Epoch 2/20 — có thể resume để cải thiện)

Chi tiết đầy đủ → [`vispoofdb/experiments/results_summary_final.csv`](vispoofdb/experiments/results_summary_final.csv)

### Kích thước mô hình & Inference time
| Mô hình | Dung lượng | Inference/sample |
|---|---:|---:|
| SVM (LFCC, MFCC, Tone) | ~2-5 MB | ~1 ms |
| MLP (Wav2Vec2) | ~1 MB | ~5 ms |
| XGBoost | ~3 MB | ~2 ms |
| **AASIST** | **0.64 MB** | **~50 ms** (CPU) |

## Utility Scripts & Advanced Features

### Đánh giá Noise Robustness
Kiểm tra hiệu suất dưới ảnh hưởng của nhiễu (background, codec, ...):

```bash
python vispoofdb/scripts/eval_noise_augmentation.py --n-samples 100 --model-type ensemble
```

CLI flags:
- `--n-samples`: Số sample để test
- `--model-type`: `single|ensemble|aasist`
- `--augmentor`: `audiomentations|simple`

### Tối ưu Mô hình (Quantize + Prune)
Nén mô hình AASIST cho deployment:

```bash
python vispoofdb/scripts/quantize.py \
  --model-path vispoofdb/models_saved/aasist_best_model.pth \
  --n-benchmark 50
```

**Yêu cầu thêm:**
- `audiomentations`: `pip install audiomentations` (optional)
- `ffmpeg`: [Download](https://ffmpeg.org/download.html) (optional)

## Cấu trúc Repository

```
.
├── app.py                                    # 🎯 Streamlit web app (3 modes: Ensemble/Single/AASIST)
├── aasist_inference.py                       # 🧠 AASIST wrapper + XAI explainer
├── README.md                                 # Documentation
├── requirements.txt                          # Dependencies
├── create_final_results_table.py              # 📊 Generate results table
│
├── AASIST/                                   # Deep learning model (Anti-Spoofing with Automatic Speaker Verification)
│   ├── train.py
│   ├── dataset.py
│   ├── models/baseline.py
│   └── ...
│
├── data/
│   ├── raw/                                  # Original data (~700 MB)
│   │   ├── real/     (7K files)
│   │   └── ai/       (synthetic deepfakes)
│   ├── clean_data/                           # Augmented data (~1.4 GB, 21K samples)
│   │   ├── real/     (14K files)
│   │   ├── fake/     (7.2K files)
│   │   └── metadata.csv
│   └── features_*/                           # Extracted features (.npy)
│       ├── features_mfcc/
│       ├── features_lfcc/
│       ├── features_wav2vec/
│       └── ...
│
├── vispoofdb/
│   ├── data/                                 # ViSpoofDB dataset (large files — .gitignored)
│   │   ├── raw/
│   │   ├── clean_data/
│   │   └── features_*/
│   ├── models/                               # Training scripts (9 mô hình)
│   │   ├── train_lfcc_svm.py
│   │   ├── train_svm.py                      # MFCC
│   │   ├── train_mlp.py                      # MFCC
│   │   ├── train_wav2vec.py                  # Wav2Vec2
│   │   ├── train_xgboost.py
│   │   ├── train_tone_svm.py
│   │   ├── train_tone_xgboost.py
│   │   ├── train_aasist.py                   # AASIST wrapper
│   │   └── ...
│   ├── models_saved/                         # Trained models (.pkl, .pth — .gitignored)
│   ├── data_processing/                      # Feature extraction
│   │   ├── vidb_extract_mfcc.py
│   │   ├── vidb_extract_processing.py
│   │   └── ...
│   ├── data_model/                           # Feature converters
│   │   ├── lfcc_svm.py
│   │   ├── mlp_features.py
│   │   ├── svm_features.py
│   │   ├── tone_features.py
│   │   ├── wav2vec2.py
│   │   ├── xgboost_features.py
│   │   └── ...
│   ├── scripts/                              # Utility scripts
│   │   ├── scripts_train.py                  # 🚀 Run all 9 models sequentially
│   │   ├── eval_noise_augmentation.py        # 🔊 Noise robustness evaluation
│   │   ├── quantize.py                       # 📦 Model compression
│   │   ├── experiment_fusion.py
│   │   ├── plot_results.py
│   │   └── ...
│   ├── experiments/                          # Results
│   │   ├── results_summary_final.csv         # 📊 Final results table
│   │   ├── results_summary_final.txt
│   │   └── noise_eval/
│   ├── xai/                                  # Explainability (XAI)
│   │   ├── shap_explainer.py
│   │   ├── visualizer.py
│   │   └── ...
│   └── ...
│
└── src/                                      # Older/alternative implementations
    ├── data_model/
    ├── data_processing/
    ├── models/
    ├── visualize/
    ├── xai/
    └── ...
```

---

**Contact & Questions**: Tạo issue trên GitHub hoặc liên hệ qua email.


