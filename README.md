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

## Chuẩn bị dữ liệu
1. Đặt các file audio vào `vispoofdb/data/clean_data/raw/` theo cấu trúc metadata.
2. Tạo `metadata.csv` (đã có sẵn trong `vispoofdb/data/clean_data/metadata.csv`).
3. Chạy các script trích xuất đặc trưng (nếu chưa có numpy features):

```bash
python vispoofdb/data_processing/extract_processing.py
python vispoofdb/data_model/wav2vec2.py   # Wav2Vec2 embeddings
python vispoofdb/models/lfcc_svm.py       # tạo LFCC features
python vispoofdb/models/svm_features.py   # MFCC/SVM features pipeline
python vispoofdb/models/tone_features.py  # tone-aware features
```

## Chạy huấn luyện / thử nghiệm
- Huấn luyện từng mô hình (ví dụ):

```bash
python vispoofdb/models/train_lfcc_svm.py
python vispoofdb/models/train_tone_svm.py
python vispoofdb/models/train_tone_xgboost.py
```

- Chạy toàn bộ thử nghiệm fusion/stacking và xuất bảng so sánh:

```bash
.venv\Scripts\python.exe vispoofdb/scripts/experiment_fusion.py
```

Kết quả sẽ được lưu tại: `vispoofdb/experiments/results_summary.csv`.

## Báo cáo kết quả (tóm tắt)
Kết quả mẫu (từ lần chạy gần nhất) — metric chính: EER trên `test_unseen`:

| Mô hình | Test_unseen Acc | Test_unseen EER |
|---|---:|---:|
| svm_lfcc (baseline) | 73.19% | 12.49% |
| svm_tone (baseline) | 86.35% | 14.38% |
| xgboost_tone (baseline) | 87.63% | 12.57% |
| svm_on_wav2vec (swap) | 97.02% | 2.63% |
| late_fusion_avg (soft voting) | 89.66% | 9.94% |
| svm_early_fusion (MFCC pooled + Tone + LFCC) | 74.43% | 3.29% |
| stacking_logreg (OOF → Logistic) | 94.00% | 4.77% |

Xem chi tiết đầy đủ tại: [vispoofdb/experiments/results_summary.csv](vispoofdb/experiments/results_summary.csv#L1).

# Vietnamese-Audio-Deepfake-Detection