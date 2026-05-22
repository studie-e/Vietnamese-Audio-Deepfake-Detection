"""
tone_features.py
================
Trích xuất bộ đặc trưng Tone-Aware (nhận biết thanh điệu) cho tiếng Việt.

Tiếng Việt có 6 thanh điệu với đường viền F0 đặc trưng:
  - Thanh ngang (flat)
  - Thanh huyền (falling-low)
  - Thanh sắc (rising-high)
  - Thanh nặng (falling-constricted)
  - Thanh hỏi (dipping/rising)
  - Thanh ngã (rising-constricted/glottalized)

Giọng AI thường tái tạo F0 không tự nhiên: quá ổn định, thiếu jitter/shimmer,
đường viền pitch quá mượt hoặc sai hình dạng thanh điệu.

Bộ đặc trưng (24 chiều):
  [0-5]   F0 thống kê: mean, std, median, min, max, range
  [6-8]   F0 đường viền: linear_slope, voiced_rate, mean_abs_delta
  [9-12]  Jitter & Shimmer: local_jitter, rap_jitter, local_shimmer, db_shimmer
  [13-16] Delta F0: delta_mean, delta_std, delta2_mean, delta2_std
  [17]    HNR (Harmonic-to-Noise Ratio) ước lượng
  [18-21] ZCR & Energy: zcr_mean, zcr_std, rms_mean, rms_std
  [22-23] MFCC-1 (năng lượng): mfcc1_mean, mfcc1_std

Lưu: features_model/tone/X_tone.npy, y_tone.npy, splits_tone.npy
"""

from pathlib import Path
import sys
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / 'vispoofdb' / 'data' / 'clean_data'
METADATA_PATH  = CLEAN_DATA_DIR / 'metadata.csv'
SAVE_DIR       = BASE_DIR / 'vispoofdb' / 'data' / 'features_model' / 'tone'

SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR  = 16000
FMIN       = 60.0    # Hz — ngưỡng F0 tối thiểu (giọng người)
FMAX       = 500.0   # Hz — ngưỡng F0 tối đa (bao phủ mọi thanh điệu VN)
FRAME_LEN  = 2048
HOP_LENGTH = 512

FEATURE_DIM = 24
FEATURE_NAMES = [
    # F0 statistics [0-5]
    'f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 'f0_range',
    # F0 contour [6-8]
    'f0_linear_slope', 'voiced_rate', 'f0_mean_abs_delta',
    # Jitter & Shimmer [9-12]
    'local_jitter', 'rap_jitter', 'local_shimmer', 'db_shimmer',
    # Delta F0 [13-16]
    'delta_f0_mean', 'delta_f0_std', 'delta2_f0_mean', 'delta2_f0_std',
    # HNR [17]
    'hnr_mean',
    # ZCR & Energy [18-21]
    'zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
    # MFCC-1 energy [22-23]
    'mfcc1_mean', 'mfcc1_std',
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: ước lượng F0 bằng PYIN (Probabilistic YIN)
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Trả về (f0, voiced_flag).
    Dùng pyin nếu librosa >= 0.8.1, fallback về yin nếu cũ hơn.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=FRAME_LEN,
            hop_length=HOP_LENGTH,
            fill_na=0.0,
        )
        return f0, voiced_flag
    except AttributeError:
        # Fallback: yin (librosa < 0.8)
        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH)
        voiced_flag = f0 > 0
        return f0, voiced_flag


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tính jitter (biến động chu kỳ pitch)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_jitter(f0_voiced: np.ndarray) -> tuple[float, float]:
    """
    local_jitter: mean(|T[i] - T[i-1]|) / mean(T)
    rap_jitter:   mean(|T[i] - mean(T[i-1], T[i], T[i+1])|) / mean(T)
    T[i] = 1/F0[i] (chu kỳ pitch)
    """
    if len(f0_voiced) < 3:
        return 0.0, 0.0

    # Lọc các giá trị F0 hợp lệ (tránh chia 0)
    f0_valid = f0_voiced[f0_voiced > 10]
    if len(f0_valid) < 3:
        return 0.0, 0.0

    T = 1.0 / f0_valid  # period array

    # Local jitter
    diffs = np.abs(np.diff(T))
    local_jitter = np.mean(diffs) / (np.mean(T) + 1e-8)

    # RAP jitter (3-point average)
    rap_diffs = []
    for i in range(1, len(T) - 1):
        avg_3 = (T[i - 1] + T[i] + T[i + 1]) / 3.0
        rap_diffs.append(abs(T[i] - avg_3))
    rap_jitter = np.mean(rap_diffs) / (np.mean(T) + 1e-8) if rap_diffs else 0.0

    return float(local_jitter), float(rap_jitter)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: tính shimmer (biến động biên độ)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_shimmer(y: np.ndarray, f0_voiced: np.ndarray) -> tuple[float, float]:
    """
    Ước lượng shimmer từ RMS của từng frame pitch.
    local_shimmer: mean(|A[i] - A[i-1]|) / mean(A)
    db_shimmer:    20 * mean(|log10(A[i]/A[i-1])|)
    """
    # Dùng RMS per frame làm đại diện biên độ
    frame_rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME_LEN,
        hop_length=HOP_LENGTH,
    )[0]

    # Chỉ lấy frames voiced (khớp số frame với f0)
    min_len = min(len(frame_rms), len(f0_voiced))
    rms_voiced = frame_rms[:min_len][f0_voiced[:min_len] > 0]

    if len(rms_voiced) < 2:
        return 0.0, 0.0

    rms_safe = rms_voiced + 1e-10
    diffs = np.abs(np.diff(rms_safe))
    local_shimmer = float(np.mean(diffs) / (np.mean(rms_safe) + 1e-8))

    # dB shimmer
    ratios = rms_safe[1:] / (rms_safe[:-1] + 1e-10)
    db_shimmer = float(20.0 * np.mean(np.abs(np.log10(ratios + 1e-10))))

    return local_shimmer, db_shimmer


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ước lượng HNR (Harmonic-to-Noise Ratio)
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_hnr(y: np.ndarray, sr: int, f0_mean: float) -> float:
    """
    Ước lượng HNR đơn giản dựa trên năng lượng hài âm vs. tổng năng lượng.
    """
    if f0_mean <= 0:
        return 0.0

    # Tính STFT
    D = librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP_LENGTH)
    mag = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LEN)

    # Lấy tần số hài (harmonic) F0, 2F0, 3F0, ... tối đa 6 hài
    harmonic_energy = 0.0
    for h in range(1, 7):
        target_freq = h * f0_mean
        if target_freq > freqs[-1]:
            break
        idx = np.argmin(np.abs(freqs - target_freq))
        # Lấy bin xung quanh (±2 bins)
        lo, hi = max(0, idx - 2), min(mag.shape[0], idx + 3)
        harmonic_energy += np.sum(mag[lo:hi, :] ** 2)

    total_energy = np.sum(mag ** 2) + 1e-10
    hnr_ratio = harmonic_energy / total_energy

    # Chuyển sang dB
    hnr_db = 10 * np.log10(hnr_ratio / (1 - hnr_ratio + 1e-10) + 1e-10)
    return float(np.clip(hnr_db, -30, 40))


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính: trích xuất bộ đặc trưng Tone-Aware
# ─────────────────────────────────────────────────────────────────────────────
def extract_tone_features(file_path: str | Path) -> np.ndarray | None:
    """
    Trích xuất vector đặc trưng Tone-Aware 24 chiều từ một file audio.

    Returns
    -------
    np.ndarray of shape (24,), hoặc None nếu lỗi.
    """
    try:
        y, sr = librosa.load(str(file_path), sr=TARGET_SR)

        # Đảm bảo audio đủ dài (ít nhất 0.1 giây)
        if len(y) < sr * 0.1:
            return None

        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        # ── 1. Ước lượng F0 ──────────────────────────────────────────────
        f0, voiced_flag = _estimate_f0(y, sr)

        voiced_f0 = f0[voiced_flag > 0] if voiced_flag is not None else f0[f0 > 0]
        voiced_rate = float(np.sum(voiced_flag > 0) / (len(f0) + 1e-8)) \
            if voiced_flag is not None else float(np.mean(f0 > 0))

        if len(voiced_f0) == 0:
            # Không phát hiện được voiced region — trả về zeros
            return features

        # ── 2. F0 Statistics [0-5] ───────────────────────────────────────
        features[0] = float(np.mean(voiced_f0))        # f0_mean
        features[1] = float(np.std(voiced_f0))         # f0_std
        features[2] = float(np.median(voiced_f0))      # f0_median
        features[3] = float(np.min(voiced_f0))         # f0_min
        features[4] = float(np.max(voiced_f0))         # f0_max
        features[5] = float(np.max(voiced_f0) - np.min(voiced_f0))  # f0_range

        # ── 3. F0 Contour Shape [6-8] ────────────────────────────────────
        # Linear slope của toàn đường viền F0 voiced
        x = np.arange(len(voiced_f0), dtype=float)
        if len(x) > 1:
            slope = float(np.polyfit(x, voiced_f0, 1)[0])
        else:
            slope = 0.0
        features[6] = slope                             # f0_linear_slope
        features[7] = voiced_rate                       # voiced_rate
        features[8] = float(np.mean(np.abs(np.diff(voiced_f0)))) if len(voiced_f0) > 1 else 0.0
        # mean_abs_delta (trung bình biến động tuyệt đối giữa các frame)

        # ── 4. Jitter & Shimmer [9-12] ───────────────────────────────────
        local_jitter, rap_jitter = _compute_jitter(voiced_f0)
        local_shimmer, db_shimmer = _compute_shimmer(y, voiced_flag if voiced_flag is not None else f0 > 0)
        features[9]  = local_jitter
        features[10] = rap_jitter
        features[11] = local_shimmer
        features[12] = db_shimmer

        # ── 5. Delta F0 [13-16] ──────────────────────────────────────────
        # Tính trên toàn bộ chuỗi F0 (kể cả unvoiced = 0)
        f0_seq = f0.copy()
        delta_f0  = np.diff(f0_seq)
        delta2_f0 = np.diff(delta_f0)

        features[13] = float(np.mean(delta_f0))         # delta_f0_mean
        features[14] = float(np.std(delta_f0))          # delta_f0_std
        features[15] = float(np.mean(delta2_f0)) if len(delta2_f0) > 0 else 0.0  # delta2_f0_mean
        features[16] = float(np.std(delta2_f0))  if len(delta2_f0) > 0 else 0.0  # delta2_f0_std

        # ── 6. HNR [17] ──────────────────────────────────────────────────
        features[17] = _estimate_hnr(y, sr, float(features[0]))

        # ── 7. ZCR & Energy [18-21] ──────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=FRAME_LEN, hop_length=HOP_LENGTH
        )[0]
        rms = librosa.feature.rms(
            y=y, frame_length=FRAME_LEN, hop_length=HOP_LENGTH
        )[0]

        features[18] = float(np.mean(zcr))      # zcr_mean
        features[19] = float(np.std(zcr))       # zcr_std
        features[20] = float(np.mean(rms))      # rms_mean
        features[21] = float(np.std(rms))       # rms_std

        # ── 8. MFCC-1 (Energy-related) [22-23] ──────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1, hop_length=HOP_LENGTH)
        features[22] = float(np.mean(mfccs[0]))  # mfcc1_mean
        features[23] = float(np.std(mfccs[0]))   # mfcc1_std

        # Kiểm tra NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    except Exception as e:
        print(f"[ERROR] Lỗi xử lý {Path(file_path).name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main: xử lý toàn bộ dataset
# ─────────────────────────────────────────────────────────────────────────────
def process_all_data():
    print("=" * 65)
    print("  TONE-AWARE FEATURE EXTRACTION (24 chiều)")
    print("  Trích xuất đặc trưng F0/Pitch cho phát hiện deepfake tiếng Việt")
    print("=" * 65)
    print(f"\nMetadata: {METADATA_PATH}\n")
    print(f"Các đặc trưng sẽ trích xuất ({FEATURE_DIM} chiều):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i:2d}] {name}")
    print()

    if not METADATA_PATH.exists():
        print(f"[ERROR] Không tìm thấy metadata.csv tại {METADATA_PATH}")
        print("Hãy chạy vispoofdb_generate_metadata.py trước!")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"Tổng số file trong metadata: {len(df)}")
    print(df.groupby(['label', 'split']).size().to_string())
    print()

    X, y_arr, splits_arr, paths = [], [], [], []
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích xuất Tone-Aware"):
        file_path = CLEAN_DATA_DIR / row['file_path']
        label = 0 if row['label'] == 'real' else 1
        split = row['split']

        if not file_path.exists():
            print(f"[WARN] Không tìm thấy file: {file_path}")
            errors += 1
            continue

        feat = extract_tone_features(file_path)
        if feat is not None:
            X.append(feat)
            y_arr.append(label)
            splits_arr.append(split)
            paths.append(row['file_path'])
        else:
            errors += 1

    X      = np.array(X, dtype=np.float32)
    y_out  = np.array(y_arr)
    splits = np.array(splits_arr)
    paths  = np.array(paths)

    print(f"\n{'='*65}")
    print(f"  HOÀN THÀNH!")
    print(f"  Tổng số file đã xử lý: {len(X)} / {len(df)}")
    print(f"  Số file lỗi / bỏ qua:  {errors}")
    print(f"  Kích thước ma trận X:   {X.shape}  (N x {FEATURE_DIM})")
    print(f"  Phân phối splits: {dict(zip(*np.unique(splits, return_counts=True)))}")
    print(f"{'='*65}\n")

    # Lưu kết quả
    np.save(SAVE_DIR / 'X_tone.npy',      X)
    np.save(SAVE_DIR / 'y_tone.npy',      y_out)
    np.save(SAVE_DIR / 'splits_tone.npy', splits)
    np.save(SAVE_DIR / 'paths_tone.npy',  paths)

    print(f"Đã lưu tại: {SAVE_DIR}")
    print(f"  X_tone.npy      — đặc trưng Tone-Aware {X.shape}")
    print(f"  y_tone.npy      — nhãn {y_out.shape}")
    print(f"  splits_tone.npy — phân chia train/test {splits.shape}")
    print(f"  paths_tone.npy  — đường dẫn file {paths.shape}")


if __name__ == "__main__":
    process_all_data()
