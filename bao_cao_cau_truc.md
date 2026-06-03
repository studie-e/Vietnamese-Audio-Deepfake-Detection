# Cấu trúc Báo cáo Khoa học — Viet-Guard

## Phân tích báo cáo hiện tại (seminar.pdf — 8 trang)

### Nội dung đã có
- Phần 3.3: Trực quan hóa đặc trưng (MFCC, Pitch/F0 contour)
- Phần 3.4.2–3: Trích xuất đặc trưng (MFCC-Delta 480d cho XGBoost, Wav2Vec2 cho MLP)
- Bảng 1: Kết quả 5 mô hình cơ sở (SVM-LFCC tốt nhất: 89.38%, EER 0.1042)
- Phần 4.2: Mô tả ensemble soft voting (chưa có bảng kết quả ensemble)
- Tài liệu tham khảo [3]–[10]
- Phụ lục: Phân công công việc (Agile/Scrum)
- Bảng 2: Cấu trúc dataset mục tiêu
- Hướng phát triển: nén model, triển khai browser extension/API

### Những gì còn thiếu
- Abstract hoàn chỉnh
- Phần 1 — Introduction (có thể đã có nhưng bị cắt ở trang 1-2)
- Bảng kết quả Ensemble (chỉ mô tả, chưa có số liệu)
- Phần XAI / Explainability chưa có
- Phần đánh giá Noise Robustness chưa có
- Phần Kết luận chưa rõ ràng

---

## Cấu trúc báo cáo hoàn chỉnh đề xuất

> Format: Báo cáo khoa học LaTeX (IEEE/ACM style), ~10–14 trang 2 cột

---

### ABSTRACT (½ trang)

- Bài toán: Phát hiện giọng nói deepfake tiếng Việt
- Phương pháp đề xuất: Ensemble 3 model (SVM+LFCC, XGBoost+MFCC-Delta, MLP+Wav2Vec2)
- Dataset: ViSpoofDB — 14,195 mẫu (real + synthetic từ 4 nguồn AI)
- Kết quả nổi bật: SVM+LFCC đạt 89.38%, Ensemble đạt X.XX%
- Đóng góp: XAI (SHAP), đánh giá robustness dưới noise

---

### 1. GIỚI THIỆU (1 trang)

**1.1 Bối cảnh và động lực**
- Sự phát triển của TTS, voice cloning, deepfake audio
- Rủi ro tại Việt Nam: lừa đảo qua điện thoại, giọng nói giả mạo

**1.2 Phát biểu bài toán**
- Input: file âm thanh tiếng Việt
- Output: Real / Fake + xác suất + giải thích

**1.3 Đóng góp của nghiên cứu** *(bullet points rõ ràng)*
- Xây dựng ViSpoofDB — bộ dữ liệu tiếng Việt 14K+ mẫu
- Ensemble 3 nhóm feature với lý do chọn lựa rõ ràng
- Tích hợp XAI (TreeSHAP, KernelSHAP, Gradient Saliency)
- Đánh giá robustness dưới noise thực tế (telephone, MP3, SNR)
- Web app demo (Viet-Guard)

**1.4 Cấu trúc bài báo**

---

### 2. CƠ SỞ LÝ THUYẾT VÀ CÔNG TRÌNH LIÊN QUAN (1–1.5 trang)

**2.1 Đặc trưng âm thanh**
- MFCC: đặc tính âm sắc bề mặt
- LFCC: bộ lọc tuyến tính — ít bị ảnh hưởng bởi nền giọng AI
- MFCC-Delta & Delta-Delta: thông tin temporal (thay đổi theo thời gian)
- Wav2Vec 2.0: self-supervised deep embeddings

**2.2 Mô hình phân loại**
- SVM: margin maximization, phù hợp high-dim low-sample
- XGBoost: gradient boosting, xử lý tốt temporal features
- MLP: học phi tuyến từ deep embeddings
- AASIST: Graph Neural Network cho anti-spoofing

**2.3 Phương pháp Ensemble**
- Soft voting: trung bình xác suất
- Lý do chọn 3 nhóm feature (spectral / temporal / deep)

**2.4 Explainable AI (XAI)**
- SHAP (TreeSHAP, KernelSHAP)
- Gradient-based saliency (AASIST)

**2.5 Công trình liên quan** *(ngắn gọn, 3–4 công trình)*
- Trích dẫn [6][7][8][9][10] đã có trong PDF

---

### 3. PHƯƠNG PHÁP (3–4 trang) ← *Mở rộng từ nội dung hiện tại*

**3.1 Tổng quan hệ thống** *(1 sơ đồ kiến trúc pipeline)*

```
Audio Input
    ↓
Feature Extraction (LFCC | MFCC-Delta | Wav2Vec2)
    ↓
3 Models (SVM | XGBoost | MLP)
    ↓
Soft Voting Ensemble
    ↓
[Real / Fake] + Confidence + XAI Explanation
```

**3.2 Tập dữ liệu ViSpoofDB**
- Nguồn giọng thật: VIVOS, VLSP (2,000 mẫu)
- Nguồn giọng AI: FPT.AI, Viettel, ElevenLabs, Coqui XTTS, VALL-E X (mỗi loại ~2,000)
- Phân chia: train / test_seen / test_unseen
- Bảng 2 (đã có trong PDF) — cấu trúc dataset

**3.3 Trích xuất đặc trưng** *(đã có trong PDF, bổ sung thêm)*
- 3.3.1 LFCC (40 chiều) → SVM
- 3.3.2 MFCC-Delta-Delta2 + Statistical Pooling (480 chiều) → XGBoost
- 3.3.3 Wav2Vec 2.0 embeddings (768 chiều) → MLP
- 3.3.4 Trực quan hóa đặc trưng (MFCC heatmap, Pitch contour) *(đã có trong PDF)*

**3.4 Huấn luyện mô hình**
- 3.4.1 SVM + LFCC: kernel RBF, GridSearch hyperparameter
- 3.4.2 XGBoost + MFCC-Delta: max_depth, n_estimators, learning_rate
- 3.4.3 MLP + Wav2Vec2: kiến trúc mạng, dropout, early stopping

**3.5 Ensemble (Soft Voting)**
- Công thức: P_final = (P_lfcc + P_xgb + P_w2v) / 3
- Lý do chọn 3 nhóm: mỗi nhóm nhìn audio từ góc độ khác nhau

**3.6 Explainability (XAI)**
- TreeSHAP cho XGBoost: SHAP values theo từng feature MFCC
- KernelSHAP cho Wav2Vec2: group 64 chiều embedding
- Gradient Saliency cho AASIST: saliency map trên waveform

---

### 4. THỰC NGHIỆM VÀ KẾT QUẢ (2–3 trang)

**4.1 Thiết lập thực nghiệm**
- Môi trường: Python 3.12, scikit-learn, XGBoost, PyTorch
- Phần cứng: [ghi thông số máy]
- Metric đánh giá: Accuracy, Precision, Recall, F1, EER

**4.2 Kết quả 5 mô hình cơ sở** *(Bảng 1 — đã có trong PDF)*

| Model | Feature | Accuracy | Precision | Recall | F1 | EER |
|---|---|---|---|---|---|---|
| SVM | LFCC | 89.38% | 0.89 | 0.90 | 0.89 | 0.1042 |
| MLP | Wav2Vec2 | 88.75% | 0.90 | 0.87 | 0.89 | 0.1083 |
| XGBoost | MFCC-Δ | 86.88% | 0.86 | 0.87 | 0.86 | 0.1250 |
| MLP | MFCC | 86.25% | 0.85 | 0.88 | 0.86 | 0.1333 |
| SVM | MFCC | 84.58% | 0.84 | 0.86 | 0.85 | 0.1542 |

**4.3 Kết quả Ensemble** *(CẦN BỔ SUNG — chưa có số liệu)*

| Hệ thống | Test Seen | Test Unseen | EER |
|---|---|---|---|
| SVM + LFCC (best single) | ? | ? | ? |
| Ensemble (3 models) | ? | ? | ? |

> Nhận xét: Ensemble có cải thiện không? Tại sao?

**4.4 Phân tích XAI** *(CẦN BỔ SUNG)*
- SHAP waterfall plot: feature nào quan trọng nhất với XGBoost?
- Wav2Vec2 KernelSHAP: nhóm embedding nào quyết định?
- Nhận xét: model "nhìn vào đâu" để quyết định?

**4.5 Đánh giá Robustness dưới Noise** *(CẦN BỔ SUNG — đã có data)*

| Scenario | AASIST EER | Wav2Vec+MLP EER | Ghi chú |
|---|---|---|---|
| Clean | 0.00 | 0.12 | Baseline |
| Noise SNR 10dB | 0.16 | 0.62 | Nhiễu trắng vừa |
| Telephone | 0.00 | 0.32 | Lọc 300–3400 Hz |
| MP3 32kbps | 0.00 | 0.18 | Nén codec |

> Nhận xét: AASIST robust, Wav2Vec+MLP nhạy cảm với noise

**4.6 Thảo luận**
- Điểm mạnh: AASIST robust, SVM+LFCC nhẹ và nhanh
- Điểm yếu: Wav2Vec2 đòi hỏi tài nguyên, bị ảnh hưởng bởi noise
- Hạn chế: dataset chưa đa dạng phương ngữ, giọng địa phương

---

### 5. KẾT LUẬN (½ trang)

- Tóm tắt đóng góp
- Kết quả tốt nhất đạt được
- Hướng phát triển:
  - Mở rộng dataset (phương ngữ, điều kiện thực tế hơn)
  - Nén model cho real-time (quantization, pruning)
  - Triển khai API / Browser Extension

---

### TÀI LIỆU THAM KHẢO

*(Đã có trong PDF [3]–[10], thêm:)*
- [1] AASIST paper
- [2] VIVOS dataset paper
- [3]–[10] như hiện tại

---

### PHỤ LỤC *(theo yêu cầu đề)*

**A. Phân công công việc** *(đã có trong PDF)*
- Bảng Agile/Scrum
- Phân công: Hiển (SVM+LFCC), Hiền (XGBoost+XAI), Hoan (Wav2Vec2+MLP)

**B. Khó khăn gặp phải**
- Wav2Vec2 chậm (7+ giờ trích xuất 14K file)
- AMD GPU không hỗ trợ CUDA → CPU-only
- Mất cân bằng lớp (real nhiều hơn fake)
- Encoding tiếng Việt trên Windows terminal

**C. Đề xuất / Hướng phát triển chi tiết**
- Batch processing để tăng tốc Wav2Vec2
- Weighted ensemble thay vì soft voting đều
- Fine-tune Wav2Vec2 trên tiếng Việt

---

## Tóm tắt những việc cần làm để hoàn chỉnh báo cáo

| # | Việc cần làm | Nguồn dữ liệu | Ưu tiên |
|---|---|---|---|
| 1 | Chạy pipeline → có số liệu Ensemble | `vispoofdb/experiments/` | 🔴 Cao |
| 2 | Chụp màn hình SHAP plot từ app | Web app | 🔴 Cao |
| 3 | Bổ sung bảng Noise Robustness | `noise_eval_summary_all_models.csv` | 🟡 Trung bình |
| 4 | Viết phần Kết luận | — | 🔴 Cao |
| 5 | Viết phần XAI (3.6 + 4.4) | Code + screenshots | 🟡 Trung bình |
| 6 | Hoàn thiện Abstract | — | 🔴 Cao |
| 7 | Viết phụ lục B (Khó khăn) | — | 🟢 Thấp |
| 8 | Làm slide trình bày | — | 🔴 Cao |
