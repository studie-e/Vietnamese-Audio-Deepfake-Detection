# Bổ Sung Tone-Aware Features (Đặc Trưng Nhận Biết Thanh Điệu)

## Tổng Quan

Tiếng Việt có 6 thanh điệu với đường viền F0 (pitch contour) đặc trưng riêng biệt.
Phân tích dữ liệu cho thấy đường viền cao độ là nơi khác biệt rõ nhất giữa giọng thật và giọng giả AI.
Pipeline hiện tại chỉ dùng MFCC/LFCC — các đặc trưng phổ tần, không mô hình hóa F0 một cách có hệ thống.

**Mục tiêu**: Xây dựng module `tone_features.py` trích xuất bộ đặc trưng F0/pitch phong phú, tích hợp vào pipeline hiện tại dưới dạng mô hình riêng biệt (SVM + Tone-Aware Features), và cập nhật các script điều phối.

---

## Proposed Changes

### Component 1: Feature Extraction Module

#### [NEW] [tone_features.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/data_model/tone_features.py)

Module trích xuất **bộ đặc trưng Tone-Aware** đa chiều gồm:

| Nhóm | Đặc trưng | Số chiều | Lý do |
|------|-----------|----------|-------|
| **F0 Thống kê** | mean, std, median, min, max, range | 6 | Phân bố cao độ tổng thể |
| **F0 Đường viền** | slope, curvature, voiced_rate | 3 | Dạng đường nét F0 (rising/falling/flat) |
| **Jitter & Shimmer** | local jitter, RAP jitter, local shimmer, dB shimmer | 4 | Biến động chu kỳ F0 — giọng AI thường ổn định bất thường |
| **F0 Biến đổi theo thời gian** | delta F0 mean/std, delta² F0 mean/std | 4 | Tốc độ thay đổi âm điệu |
| **HNR (Harmonic Noise Ratio)** | mean HNR | 1 | Tỉ lệ hài/nhiễu — liên quan trực tiếp đến tự nhiên của giọng |
| **ZCR & Energy** | ZCR mean/std, RMS energy mean/std | 4 | Hỗ trợ phát hiện vùng hữu thanh/vô thanh |
| **MFCC1 (Năng lượng)** | MFCC-1 mean/std | 2 | Tương quan với intensity của tone |

**Tổng: ~24 chiều** — gọn, có thể giải thích, bổ sung tốt cho MFCC.

Pipeline trích xuất:
1. Load audio (16kHz)
2. Ước lượng F0 bằng `librosa.pyin` (Probabilistic YIN — chính xác hơn YIN thông thường)
3. Lọc vùng voiced (F0 > 0)
4. Tính các nhóm đặc trưng trên
5. Lưu `X_tone.npy`, `y_tone.npy`, `splits_tone.npy` → `features_model/tone/`

---

### Component 2: Model Training

#### [NEW] [train_tone_svm.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/models/train_tone_svm.py)

SVM với kernel RBF, train trên Tone-Aware features (24 chiều).
Đánh giá trên `test_seen` và `test_unseen`, báo cáo Accuracy + EER.
Lưu model → `models_saved/svm_tone_model.pkl`

#### [NEW] [train_tone_xgboost.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/models/train_tone_xgboost.py)

XGBoost với hyperparameter tuning trên Tone-Aware features.
Cho phép so sánh trực tiếp với XGBoost MFCC-Delta (480 chiều) hiện tại.
Lưu model → `models_saved/xgboost_tone_model.pkl`

#### [NEW] [train_tone_fusion.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/models/train_tone_fusion.py)

**Late Fusion**: Kết hợp Tone features (24 chiều) + MFCC features (40 chiều) bằng cách ghép vector → 64 chiều → SVM.
Đây là thực nghiệm chính để đo **đóng góp thực sự của Tone features** khi kết hợp.

---

### Component 3: Pipeline Scripts

#### [MODIFY] [scripts_feature_extract.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/scripts/scripts_feature_extract.py)

Thêm bước 6 vào PIPELINE:
```
Bước 6/6 — tone_features.py → Tone-Aware 24 chiều → features_model/tone/
```

#### [MODIFY] [scripts_train.py](file:///d:/hien/Dai%20hoc/Nam%202/ki2/seminar/Vietnamese-Audio-Deepfake-Detection/vispoofdb/scripts/scripts_train.py)

Thêm 3 bước vào PIPELINE:
```
Bước 6/8 — train_tone_svm.py     → SVM + Tone-Aware
Bước 7/8 — train_tone_xgboost.py → XGBoost + Tone-Aware
Bước 8/8 — train_tone_fusion.py  → SVM + MFCC+Tone Fusion
```

---

## Verification Plan

### Automated Tests
```
python vispoofdb/data_model/tone_features.py
# → Kiểm tra X_tone.npy shape (N, 24), không có NaN

python vispoofdb/models/train_tone_svm.py
# → In accuracy + EER cho test_seen và test_unseen

python vispoofdb/models/train_tone_xgboost.py
# → In accuracy + EER, so sánh với XGBoost MFCC

python vispoofdb/models/train_tone_fusion.py
# → Kiểm tra xem fusion có cải thiện hơn MFCC đơn thuần không
```

### Manual Verification
- So sánh bảng kết quả: `SVM+MFCC` vs `SVM+Tone` vs `SVM+MFCC+Tone (Fusion)`
- Kiểm tra EER trên `test_unseen` — metric quan trọng nhất cho khả năng tổng quát hóa

---

## Open Questions

> [!IMPORTANT]
> **Chọn mô hình nào muốn train?**
> Mặc định sẽ thêm cả 3: SVM+Tone, XGBoost+Tone, và Fusion. Bạn có muốn bỏ bớt mô hình nào không?

> [!NOTE]
> **Về thư viện PYIN**: `librosa.pyin` có sẵn trong librosa ≥ 0.8. Nếu phiên bản librosa cũ hơn sẽ dùng `librosa.yin` thay thế (ít chính xác hơn một chút).
