"""Evaluate models under real-world noise augmentations.

Creates corrupted variants (additive noise at several SNRs, MP3 compression at several bitrates,
and telephone bandpass) for a random subset of test files and reports performance degradation
against clean audio.

Usage:
    python vispoofdb/scripts/eval_noise_augmentation.py --n-samples 50

Notes:
 - Requires `ffmpeg` on PATH for codec-based corruptions (MP3). If not available, codec tests are skipped.
 - Evaluates ALL individual models (SVM-LFCC, XGBoost-MFCC, MLP-MFCC, MLP-Wav2Vec, AASIST, ...)
   AND the proposed VietGuardEnsemble fusion pipeline — so robustness can be compared directly.
 - If `spafe` is not installed, the ensemble falls back to librosa-based LFCC approximation.
"""
import os
import sys
import argparse
import tempfile
import shutil
import subprocess
import random
import math
import warnings
from pathlib import Path

# Fix Windows console encoding (cp1252 không hỗ trợ ký tự Unicode như ✓ ✗)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
try:
    # pyrefly: ignore [missing-import]
    from audiomentations import Compose, AddGaussianNoise, RoomSimulator, Mp3Compression, LowPassFilter, HighPassFilter, TimeStretch, PitchShift, Gain
    AUDIOMENTATIONS_AVAILABLE = True
except Exception:
    AUDIOMENTATIONS_AVAILABLE = False


def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def add_white_noise(y, snr_db):
    # signal power
    sig_power = np.mean(y ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    # generate white noise
    noise = np.random.randn(len(y)).astype(np.float32)
    cur_noise_power = np.mean(noise ** 2)
    noise = noise * math.sqrt(target_noise_power / (cur_noise_power + 1e-12))
    return y + noise


def bandpass_telephone(y, sr, low=300, high=3400, order=6):
    sos = signal.butter(order, [low, high], btype='band', fs=sr, output='sos')
    return signal.sosfilt(sos, y)


def codec_mp3_roundtrip(in_wav, out_wav, bitrate='32k'):
    # create tmp mp3 then decode back to wav via ffmpeg
    tmp_mp3 = out_wav + ".mp3"
    cmd_enc = ["ffmpeg", "-y", "-i", in_wav, "-b:a", bitrate, tmp_mp3]
    cmd_dec = ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "16000", out_wav]
    subprocess.check_call(cmd_enc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(cmd_dec, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.remove(tmp_mp3)
    except Exception:
        pass


def compute_eer(y_true, y_score):
    """Tính EER. Trả về 0.5 nếu không đủ thông tin (all same label / all NaN)."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5  # chỉ có 1 class → EER không xác định
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        diff = np.abs(fpr - fnr)
        if np.all(np.isnan(diff)):
            return 0.5
        idx = np.nanargmin(diff)
        return float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        return 0.5


def _load_ensemble(models_dir):
    """Load VietGuardEnsemble với nhiều fallback:
    1. Import trực tiếp qua package path (nếu chạy từ root dự án)
    2. Inject sys.path rồi import
    3. Nếu `spafe` thiếu, dùng PatchedEnsemble với LFCC giả từ librosa
    Trả về object ensemble hoặc None nếu thất bại hoàn toàn.
    """
    import sys

    models_dir_str = str(models_dir)

    # --- Thử import bình thường ---
    for attempt in range(2):
        try:
            if attempt == 1:
                # Thêm thư mục gốc dự án vào sys.path
                project_root = str(Path(__file__).resolve().parents[2])
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
            from vispoofdb.models.ensemble_system import VietGuardEnsemble
            return VietGuardEnsemble(models_dir=models_dir_str)
        except ImportError as ie:
            if 'spafe' in str(ie).lower():
                # spafe không có — dùng bản vá
                print(f"[INFO] spafe không được cài đặt, dùng LFCC xấp xỉ từ librosa cho ensemble")
                return _build_patched_ensemble(models_dir_str)
            # Module không tìm thấy sau khi thử sys.path inject → tiếp tục vòng lặp
        except Exception as e:
            print(f"[WARN] Lần thử {attempt+1} load VietGuardEnsemble thất bại: {e}")

    # --- Cuối cùng thử patched ensemble (không dùng spafe) ---
    try:
        return _build_patched_ensemble(models_dir_str)
    except Exception as e:
        print(f"[ERROR] Không thể tạo patched ensemble: {e}")
        return None


def _build_patched_ensemble(models_dir_str):
    """Tạo một ensemble wrapper không phụ thuộc spafe.
    Dùng librosa MFCC làm xấp xỉ LFCC cho nhóm Spectral.
    Đây đảm bảo pipeline fusion luôn được đánh giá dù spafe có hay không.
    """
    import joblib as _jl

    class _PatchedEnsemble:
        """VietGuardEnsemble không cần spafe — LFCC xấp xỉ bằng librosa MFCC-40.
        Dùng weighted soft voting (inverse-EER): w_i = 1/EER_i
        """

        # Trọng số mặc định dựa trên EER validation clean của từng nhánh
        DEFAULT_WEIGHTS = {
            'svm_lfcc': 1.0 / 0.35,   # ≈ 2.86
            'xgb_mfcc': 1.0 / 0.15,   # ≈ 6.67
            'mlp_w2v':  1.0 / 0.05,   # = 20.0
        }

        def __init__(self, mdir):
            self._models = {}  # name -> (model, scaler_or_None, feat_fn)

            # --- Sub-model 1: SVM + LFCC (must-have) ---
            try:
                svm = _jl.load(os.path.join(mdir, 'svm_lfcc_model.pkl'))
                scaler = _jl.load(os.path.join(mdir, 'scaler_lfcc.pkl'))
                self._models['svm_lfcc'] = (svm, scaler, self._extract_lfcc_approx)
            except Exception as e:
                print(f"[WARN] ensemble: svm_lfcc không load được: {e}")

            # --- Sub-model 2: XGBoost + MFCC-480 ---
            try:
                xgb = _jl.load(os.path.join(mdir, 'best_xgboost.pkl'))
                self._models['xgb_mfcc'] = (xgb, None, self._extract_mfcc_480)
            except Exception as e:
                print(f"[WARN] ensemble: xgb_mfcc không load được: {e}")

            # --- Sub-model 3: MLP + Wav2Vec2 ---
            try:
                mlp = _jl.load(os.path.join(mdir, 'mlp_wav2vec_model.pkl'))
                scaler_w2v = _jl.load(os.path.join(mdir, 'scaler_wav2vec.pkl'))
                self._models['mlp_w2v'] = (mlp, scaler_w2v, None)  # feat_fn riêng
            except Exception as e:
                print(f"[WARN] ensemble: mlp_wav2vec không load được: {e}")

            if not self._models:
                raise RuntimeError("Không có sub-model nào trong ensemble load được!")

            loaded_names = list(self._models.keys())
            w_info = {k: f"{self.DEFAULT_WEIGHTS.get(k, 1.0):.2f}" for k in loaded_names}
            print(f"[INFO] Patched ensemble (weighted): {len(loaded_names)} sub-models: {w_info}")

            # --- Wav2Vec2 transformer (cho mlp_w2v) ---
            import torch as _torch
            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.wav2vec_available = False
            self.processor = None
            self.w2v_model_transformer = None
            if 'mlp_w2v' in self._models:
                try:
                    from transformers import Wav2Vec2Processor, Wav2Vec2Model
                    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                    self.w2v_model_transformer = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                    self.w2v_model_transformer.to(self.device)
                    self.wav2vec_available = True
                except Exception as e:
                    print(f"[WARN] ensemble: Wav2Vec2 không tải được: {e}")
                    self._models.pop('mlp_w2v', None)

        def _extract_lfcc_approx(self, y, sr):
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            return np.mean(mfcc, axis=1).reshape(1, -1)

        def _extract_mfcc_480(self, y, sr):
            mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            delta  = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            combined = np.vstack([mfccs, delta, delta2])
            return np.concatenate([
                np.mean(combined, axis=1), np.std(combined, axis=1),
                np.max(combined, axis=1),  np.min(combined, axis=1),
            ]).reshape(1, -1)

        def _extract_wav2vec(self, y, sr):
            if not self.wav2vec_available:
                return None
            import torch as _torch
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
            with _torch.no_grad():
                outputs = self.w2v_model_transformer(**inputs)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return feat.reshape(1, -1)

        def predict_audio(self, file_path):
            try:
                y, sr = librosa.load(file_path, sr=16000)
                model_keys = []
                prob_list  = []

                for name, (model, scaler, feat_fn) in self._models.items():
                    try:
                        if name == 'mlp_w2v':
                            feat = self._extract_wav2vec(y, sr)
                            if feat is None:
                                continue
                            feat_scaled = scaler.transform(feat)
                        else:
                            feat = feat_fn(y, sr)
                            feat_scaled = scaler.transform(feat) if scaler else feat

                        if hasattr(model, 'predict_proba'):
                            p = float(model.predict_proba(feat_scaled)[0][1])
                        else:
                            df = float(model.decision_function(feat_scaled)[0])
                            p = 1.0 / (1.0 + np.exp(-df))

                        model_keys.append(name)
                        prob_list.append(p)
                    except Exception:
                        pass  # bỏ qua sub-model lỗi

                if not prob_list:
                    return {"success": False, "error": "Tất cả sub-model đều lỗi"}

                # Weighted Soft Voting — trọng số nghịch đảo EER
                weights = np.array([self.DEFAULT_WEIGHTS.get(k, 1.0) for k in model_keys])
                probs   = np.array(prob_list)
                final_ai_prob = float(np.dot(weights, probs) / weights.sum())

                return {
                    "success": True,
                    "is_fake": bool(final_ai_prob >= 0.5),
                    "confidence_ai": float(final_ai_prob),
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    return _PatchedEnsemble(models_dir_str)



class SingleWav2VecDetector:
    """Lightweight copy of the SingleWav2VecDetector used by the app.
    Loads a sklearn model + scaler and the facebook/wav2vec2-base encoder to extract embeddings.
    """
    def __init__(self, model_path, scaler_path):
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2v_model.to(self.device)

    def _extract_wav2vec(self, y, sr):
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
        with self._no_grad():
            outputs = self.w2v_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features.reshape(1, -1)

    from contextlib import contextmanager

    @contextmanager
    def _no_grad(self):
        import torch
        with torch.no_grad():
            yield

    def predict_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=16000)
        feat = self._extract_wav2vec(y, sr)
        Xs = self.scaler.transform(feat)
        if hasattr(self.model, 'predict_proba'):
            p = float(self.model.predict_proba(Xs)[0][1])
        else:
            df = float(self.model.decision_function(Xs)[0])
            p = 1.0 / (1.0 + np.exp(-df))
        return {"success": True, "is_fake": bool(p >= 0.5), "confidence_ai": float(p)}


class SklearnDetector:
    """Wrapper for sklearn models (SVM, XGBoost, MLP)."""
    def __init__(self, model_path, scaler_path, feature_extractor):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and Path(scaler_path).exists() else None
        self.feature_extractor = feature_extractor
    
    def predict_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=16000)
            features = self.feature_extractor(y, sr)
            
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            if hasattr(self.model, 'predict_proba'):
                p = float(self.model.predict_proba(features)[0][1])
            else:
                p_score = float(self.model.decision_function(features)[0])
                p = 1.0 / (1.0 + np.exp(-p_score))
            
            return {"success": True, "is_fake": bool(p >= 0.5), "confidence_ai": float(p)}
        except Exception as e:
            return {"success": False, "error": str(e), "confidence_ai": np.nan}


class AASISTDetector:
    """Wrapper for AASIST deep learning model."""
    def __init__(self, model_path, device=None):
        import sys
        import torch
        from pathlib import Path
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Import AASIST tu vispoofdb/models/aasist/
        aasist_root = Path(__file__).resolve().parents[2] / 'vispoofdb' / 'models' / 'aasist'
        sys.path.insert(0, str(aasist_root))
        
        from models.baseline import Full_AASIST_Model
        
        self.model = Full_AASIST_Model().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()
    
    def predict_audio(self, file_path):
        import torch
        try:
            y, sr = librosa.load(file_path, sr=16000)
            # Prepare audio
            if len(y) < 64000:
                y = np.pad(y, (0, 64000 - len(y)))
            else:
                y = y[:64000]
            
            x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                p = float(probs[0, 1].cpu().numpy())
            
            return {"success": True, "is_fake": bool(p >= 0.5), "confidence_ai": float(p)}
        except Exception as e:
            return {"success": False, "error": str(e), "confidence_ai": np.nan}


def create_feature_extractors():
    """Create feature extractors for different sklearn models."""
    import sys
    from pathlib import Path
    
    base = Path(__file__).resolve().parents[2]
    data_dir = base / 'vispoofdb' / 'data'
    
    extractors = {}
    
    # 1. LFCC features (40 chiều)
    try:
        from spafe.features.lfcc import lfcc
        def extract_lfcc(y, sr):
            lfccs = lfcc(sig=y, fs=sr, num_ceps=40, nfilts=128)
            return np.mean(lfccs, axis=0)
        extractors['lfcc'] = extract_lfcc
    except Exception as e:
        # Fallback bằng librosa MFCC-40 mean
        def extract_lfcc_fallback(y, sr):
            import librosa
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            return np.mean(mfcc, axis=1)
        extractors['lfcc'] = extract_lfcc_fallback
    
    # 2. MFCC40 features (40 chiều) cho svm_mfcc và mlp_mfcc
    try:
        def extract_mfcc40(y, sr):
            import librosa
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            return np.mean(mfcc, axis=1)
        extractors['mfcc40'] = extract_mfcc40
    except:
        pass

    # 3. MFCC 480 features (480 chiều) cho xgboost_mfcc
    try:
        def extract_mfcc_480(y, sr):
            import librosa
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            combined = np.vstack([mfccs, delta, delta2])
            return np.concatenate([
                np.mean(combined, axis=1), np.std(combined, axis=1),
                np.max(combined, axis=1),  np.min(combined, axis=1)
            ])
        extractors['mfcc_480'] = extract_mfcc_480
    except:
        pass
    
    # 4. Tone features (24 chiều) cho svm_tone và xgboost_tone
    try:
        def extract_tone_24(y, sr):
            fmin = 60.0
            fmax = 500.0
            frame_len = 2048
            hop_length = 512
            
            try:
                import librosa
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=fmin, fmax=fmax, sr=sr,
                    frame_length=frame_len, hop_length=hop_length, fill_na=0.0
                )
            except AttributeError:
                import librosa
                f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
                voiced_flag = f0 > 0
                
            voiced_f0 = f0[voiced_flag > 0] if voiced_flag is not None else f0[f0 > 0]
            voiced_rate = float(np.sum(voiced_flag > 0) / (len(f0) + 1e-8)) if voiced_flag is not None else float(np.mean(f0 > 0))
            
            features = np.zeros(24, dtype=np.float32)
            if len(voiced_f0) > 0:
                features[0] = float(np.mean(voiced_f0))
                features[1] = float(np.std(voiced_f0))
                features[2] = float(np.median(voiced_f0))
                features[3] = float(np.min(voiced_f0))
                features[4] = float(np.max(voiced_f0))
                features[5] = float(np.max(voiced_f0) - np.min(voiced_f0))
                
                x = np.arange(len(voiced_f0), dtype=float)
                features[6] = float(np.polyfit(x, voiced_f0, 1)[0]) if len(x) > 1 else 0.0
                features[7] = voiced_rate
                features[8] = float(np.mean(np.abs(np.diff(voiced_f0)))) if len(voiced_f0) > 1 else 0.0
                
                f0_valid = voiced_f0[voiced_f0 > 10]
                if len(f0_valid) >= 3:
                    T = 1.0 / f0_valid
                    features[9] = float(np.mean(np.abs(np.diff(T))) / (np.mean(T) + 1e-8))
                    rap_diffs = []
                    for i in range(1, len(T) - 1):
                        avg_3 = (T[i - 1] + T[i] + T[i + 1]) / 3.0
                        rap_diffs.append(abs(T[i] - avg_3))
                    features[10] = np.mean(rap_diffs) / (np.mean(T) + 1e-8) if rap_diffs else 0.0
                
                frame_rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_length)[0]
                min_len = min(len(frame_rms), len(voiced_flag))
                rms_voiced = frame_rms[:min_len][voiced_flag[:min_len] > 0]
                if len(rms_voiced) >= 2:
                    rms_safe = rms_voiced + 1e-10
                    features[11] = float(np.mean(np.abs(np.diff(rms_safe))) / (np.mean(rms_safe) + 1e-8))
                    ratios = rms_safe[1:] / (rms_safe[:-1] + 1e-10)
                    features[12] = float(20.0 * np.mean(np.abs(np.log10(ratios + 1e-10))))
            
            delta_f0 = np.diff(f0)
            delta2_f0 = np.diff(delta_f0)
            features[13] = float(np.mean(delta_f0)) if len(delta_f0) > 0 else 0.0
            features[14] = float(np.std(delta_f0)) if len(delta_f0) > 0 else 0.0
            features[15] = float(np.mean(delta2_f0)) if len(delta2_f0) > 0 else 0.0
            features[16] = float(np.std(delta2_f0)) if len(delta2_f0) > 0 else 0.0
            
            if features[0] > 0:
                import librosa
                D = librosa.stft(y, n_fft=frame_len, hop_length=hop_length)
                mag = np.abs(D)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_len)
                harmonic_energy = 0.0
                for h in range(1, 7):
                    target_freq = h * features[0]
                    if target_freq > freqs[-1]:
                        break
                    idx = np.argmin(np.abs(freqs - target_freq))
                    lo, hi = max(0, idx - 2), min(mag.shape[0], idx + 3)
                    harmonic_energy += np.sum(mag[lo:hi, :] ** 2)
                total_energy = np.sum(mag ** 2) + 1e-10
                hnr_ratio = harmonic_energy / total_energy
                hnr_db = 10 * np.log10(hnr_ratio / (1 - hnr_ratio + 1e-10) + 1e-10)
                features[17] = float(np.clip(hnr_db, -30, 40))
                
            import librosa
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_length)[0]
            rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_length)[0]
            features[18] = float(np.mean(zcr))
            features[19] = float(np.std(zcr))
            features[20] = float(np.mean(rms))
            features[21] = float(np.std(rms))
            
            import librosa
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1, hop_length=hop_length)
            features[22] = float(np.mean(mfccs[0]))
            features[23] = float(np.std(mfccs[0]))
            
            return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
        extractors['tone'] = extract_tone_24
    except:
        pass
    
    # 5. Fusion features (64 chiều) cho svm_fusion
    try:
        def extract_fusion_64(y, sr):
            mfcc = extractors['mfcc40'](y, sr)
            tone = extractors['tone'](y, sr)
            return np.concatenate([mfcc, tone])
        extractors['fusion'] = extract_fusion_64
    except:
        pass

    # 6. WAV2VEC features (768 chiều)
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w2v_model.to(device).eval()
        
        def extract_wav2vec(y, sr):
            inputs = processor(y, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = w2v_model(**inputs)
                feat = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return feat
        
        extractors['wav2vec'] = extract_wav2vec
    except:
        pass
    
    return extractors


def load_metadata():
    """Tải metadata CSV từ các vị trí đã biết.
    Chuẩn hoá cột thành 'file_path' và 'label' bất kể tên cột gốc.
    """
    candidates = [
        Path('vispoofdb/data/clean_data/metadata.csv'),
        Path('thu_nghiem/data/metadata.csv'),
        Path('data/metadata.csv'),
    ]
    df = None
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError(
            'metadata.csv không tìm thấy. Đã tìm ở: ' + ', '.join(str(c) for c in candidates)
        )

    # --- Chuẩn hoá cột ---
    # Format vispoofdb chuẩn:  file_path, label
    # Format thu_nghiem:       Tên_File, Phân_Loại  (hoặc Nhan_So)
    col_map = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    # Tìm cột file_path
    for alias in ['file_path', 'filename', 'tên_file', 'ten_file', 'tên file',
                  'file', 'name', 'audio_path', 'path']:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = 'file_path'
            break

    # Tìm cột label
    for alias in ['label', 'phân_loại', 'phan_loai', 'phân loại',
                  'nhãn', 'nhan', 'class', 'type', 'category']:
        if alias in lower_cols:
            col_map[lower_cols[alias]] = 'label'
            break

    if col_map:
        df = df.rename(columns=col_map)

    # Chuẩn hoá giá trị nhãn → 'real' hoặc 'fake'
    if 'label' in df.columns:
        label_map = {
            'thật': 'real', 'real': 'real', '0': 'real', 0: 'real',
            'ai': 'fake',   'fake': 'fake', 'giả': 'fake', 'giả mạo': 'fake',
            '1': 'fake',    1: 'fake',
        }
        df['label'] = df['label'].apply(
            lambda x: label_map.get(str(x).strip().lower(), str(x).strip().lower())
        )

    return df


def build_file_path(row):
    """Tìm đường dẫn thực tế của file audio từ metadata row.
    Thử nhiều thư mục theo thứ tự ưu tiên.
    """
    fname = row.get('file_path', '')
    label = row.get('label', 'real')

    # Ánh xạ nhãn → thư mục con
    label_to_dir = {
        'real': ['real', 'Thật'],
        'fake': ['ai', 'fake', 'AI', 'Fake', 'giả mạo'],
    }
    subdirs = label_to_dir.get(label, [label])

    search_roots = [
        Path('thu_nghiem/data/clean_data'),
        Path('thu_nghiem/data/raw'),
        Path('vispoofdb/data/clean_data'),
        Path('vispoofdb/data/raw'),
    ]

    # Thử file_path trực tiếp (nếu là relative path đầy đủ)
    for root in search_roots:
        p = root / fname
        if p.exists():
            return str(p.resolve())

    # Thử với thư mục con theo nhãn
    for root in search_roots:
        for sub in subdirs:
            p = root / sub / fname
            if p.exists():
                return str(p.resolve())

    # Trả về đường đoán tốt nhất (để thông báo lỗi rõ ràng)
    return str((search_roots[0] / subdirs[0] / fname).resolve())


def evaluate_all_models(args):
    """Evaluate all 9 models under noise conditions."""
    import sys
    from pathlib import Path
    
    metadata = load_metadata()
    
    # Choose test_unseen split
    if 'split' in metadata.columns:
        pool = metadata[metadata['split'].str.contains('test', na=False)].copy()
    else:
        pool = metadata.copy()
    
    # Sample balanced subset
    labels = pool['label'].unique().tolist()
    by_label = {lab: pool[pool['label'] == lab] for lab in labels}
    n_per_label = max(1, args.n_samples // max(1, len(labels)))
    samples = []
    for lab, df in by_label.items():
        if len(df) == 0:
            continue
        samples.extend(df.sample(n=min(n_per_label, len(df)), random_state=42).to_dict('records'))
    
    if len(samples) == 0:
        raise RuntimeError('No samples selected for evaluation')
    
    base = Path(__file__).resolve().parents[2]
    models_dir = base / 'vispoofdb' / 'models_saved'
    extractors = create_feature_extractors()
    
    # Load all 9 models
    detectors = {}
    
    # 8 Sklearn models
    sklearn_models = [
        ('svm_lfcc', models_dir / 'svm_lfcc_model.pkl', models_dir / 'scaler_lfcc.pkl', 'lfcc'),
        ('svm_mfcc', models_dir / 'svm_voice_model.pkl', models_dir / 'scaler_final.pkl', 'mfcc40'),
        ('mlp_mfcc', models_dir / 'best_mlp.pkl', models_dir / 'scaler_mlp.pkl', 'mfcc40'),
        ('xgboost_mfcc', models_dir / 'best_xgboost.pkl', None, 'mfcc_480'),
        ('mlp_wav2vec', models_dir / 'mlp_wav2vec_model.pkl', models_dir / 'scaler_wav2vec.pkl', 'wav2vec'),
        ('svm_tone', models_dir / 'svm_tone_model.pkl', models_dir / 'scaler_tone.pkl', 'tone'),
        ('xgboost_tone', models_dir / 'xgboost_tone_model.pkl', None, 'tone'),
        ('svm_fusion', models_dir / 'svm_tone_fusion_model.pkl', models_dir / 'scaler_tone_fusion.pkl', 'fusion'),
    ]
    
    for name, model_path, scaler_path, feat_type in sklearn_models:
        if model_path.exists() and feat_type in extractors:
            try:
                detectors[name] = SklearnDetector(str(model_path), str(scaler_path) if scaler_path else None, extractors[feat_type])
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    # 1 AASIST model
    aasist_path = models_dir / 'aasist_best_model.pth'
    if aasist_path.exists():
        try:
            detectors['aasist'] = AASISTDetector(str(aasist_path))
            print(f"✓ Loaded AASIST")
        except Exception as e:
            print(f"✗ Failed to load AASIST: {e}")

    # VietGuardEnsemble (pipeline hệ thống fusion đề xuất)
    ensemble = _load_ensemble(models_dir)
    if ensemble is not None:
        detectors['ensemble_fusion'] = ensemble
        print(f"✓ Loaded VietGuardEnsemble (fusion pipeline)")
    else:
        print(f"✗ VietGuardEnsemble không load được — kiểm tra log phía trên")

    
    if len(detectors) == 0:
        print("[ERROR] No models could be loaded")
        return
    
    print(f"\n✓ Evaluating {len(detectors)} models under noise conditions...\n")
    
    # Noise scenarios
    scenarios = [('clean', None)]
    if args.augmentor == 'audiomentations' and AUDIOMENTATIONS_AVAILABLE:
        CONDITIONS = {
            'phone_call': Compose([
                LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=3400, p=1.0),
                HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=300, p=1.0),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.8),
                Mp3Compression(min_bitrate=16, max_bitrate=32, p=1.0),
            ]),
        }
        for k in CONDITIONS.keys():
            scenarios.append((k, {'type': 'audiomentations', 'aug': CONDITIONS[k]}))
    else:
        for snr in [20, 10, 0]:
            scenarios.append((f'noise_snr_{snr}', {'type': 'noise', 'snr': snr}))
        scenarios.append(('telephone', {'type': 'telephone'}))
        if has_ffmpeg():
            for br in ['128k', '64k', '32k']:
                scenarios.append((f'mp3_{br}', {'type': 'mp3', 'bitrate': br}))
    
    out_dir = Path('vispoofdb/experiments/noise_eval')
    fig_dir = Path('vispoofdb/figures/noise')
    ensure_dir(out_dir)
    ensure_dir(fig_dir)
    
    records = []
    
    for scen_name, scen in scenarios:
        print(f"Scenario: {scen_name}")
        y_trues = []
        y_scores = {k: [] for k in detectors.keys()}
        
        for row in samples:
            src_path = build_file_path(row)
            if not os.path.exists(src_path):
                continue
            
            y, sr = librosa.load(src_path, sr=16000)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
                tmp_out = tmpf.name
            
            try:
                if scen is None:
                    sf.write(tmp_out, y, sr)
                else:
                    t = scen['type']
                    if t == 'noise':
                        y2 = add_white_noise(y, scen['snr'])
                        sf.write(tmp_out, y2, sr)
                    elif t == 'telephone':
                        y2 = bandpass_telephone(y, sr)
                        sf.write(tmp_out, y2, sr)
                    elif t == 'mp3':
                        tmp_in = tmp_out + '.in.wav'
                        sf.write(tmp_in, y, sr)
                        try:
                            codec_mp3_roundtrip(tmp_in, tmp_out, bitrate=scen['bitrate'])
                        finally:
                            try:
                                os.remove(tmp_in)
                            except:
                                pass
                    elif t == 'audiomentations':
                        try:
                            aug = scen['aug']
                            noisy = aug(samples=y, sample_rate=sr)
                            sf.write(tmp_out, noisy, sr)
                        except:
                            sf.write(tmp_out, y, sr)
                
                # Evaluate all detectors
                for name, det in detectors.items():
                    try:
                        res = det.predict_audio(tmp_out)
                        if not res.get('success', True):
                            score = np.nan
                        else:
                            score = float(res.get('confidence_ai', np.nan))
                    except Exception as e:
                        score = np.nan
                    
                    y_scores[name].append(score)
                
                # True label
                true = 1 if row.get('label') != 'real' else 0
                y_trues.append(true)
            
            finally:
                try:
                    os.remove(tmp_out)
                except:
                    pass
        
        # Compute metrics per detector
        for name in detectors.keys():
            scores = np.array(y_scores[name], dtype=float)
            mask = ~np.isnan(scores)
            if mask.sum() == 0:
                continue
            
            y_true_arr = np.array(y_trues)[mask]
            y_score_arr = scores[mask]
            y_pred = (y_score_arr >= 0.5).astype(int)
            
            acc = accuracy_score(y_true_arr, y_pred)
            prec = precision_score(y_true_arr, y_pred, zero_division=0)
            rec = recall_score(y_true_arr, y_pred, zero_division=0)
            f1 = f1_score(y_true_arr, y_pred, zero_division=0)
            eer = compute_eer(y_true_arr, y_score_arr)
            
            records.append({
                'scenario': scen_name,
                'detector': name,
                'n_samples': int(mask.sum()),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'eer': eer,
            })
    
    df_summary = pd.DataFrame(records)
    df_summary.to_csv(out_dir / 'noise_eval_summary_all_models.csv', index=False)
    
    # (Loại bỏ vẽ biểu đồ EER đơn lẻ cho từng mô hình để tránh dư thừa)

    # ── 2. Combined comparison: ALL detectors on one chart ───────────────────
    _plot_combined_eer_comparison(df_summary, fig_dir)

    # ── 3. Accuracy degradation bar chart (clean vs noisiest condition) ───────
    _plot_accuracy_degradation(df_summary, fig_dir)

    # ── 4. Print summary table to console ─────────────────────────────────────
    _print_summary_table(df_summary)

    print(f"\n✓ Results saved to {out_dir}")
    print(f"✓ Plots saved to {fig_dir}")
    print(f"✓ Summary CSV: {out_dir / 'noise_eval_summary_all_models.csv'}")


def _plot_combined_eer_comparison(df_summary, fig_dir):
    """Vẽ EER của TẤT CẢ detector trên cùng 1 biểu đồ, highlight ensemble."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    detectors = sorted(df_summary['detector'].unique())
    scenarios  = df_summary[df_summary['detector'] == detectors[0]]['scenario'].tolist()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Màu sắc: ensemble nổi bật hơn
    colors = cm.tab10.colors
    for i, name in enumerate(detectors):
        sub = df_summary[df_summary['detector'] == name].set_index('scenario')
        eer_vals = [sub.loc[s, 'eer'] if s in sub.index else float('nan') for s in scenarios]

        is_ensemble = 'ensemble' in name.lower()
        lw   = 3.5 if is_ensemble else 1.5
        ms   = 10  if is_ensemble else 6
        ls   = '-' if is_ensemble else '--'
        zord = 10  if is_ensemble else 2
        color = '#e74c3c' if is_ensemble else colors[i % len(colors)]
        label = f'★ {name} (Fusion Pipeline)' if is_ensemble else name

        ax.plot(scenarios, eer_vals, marker='o', linewidth=lw, markersize=ms,
                linestyle=ls, color=color, zorder=zord, label=label)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Noise / Corruption Scenario', fontsize=12)
    ax.set_ylabel('EER  (↓ better)', fontsize=12)
    ax.set_title('Noise Robustness — EER Comparison: All Detectors vs. Fusion Pipeline',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Vùng "clean" highlight
    if scenarios and scenarios[0] == 'clean':
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.text(0.25, ax.get_ylim()[1] * 0.97, 'clean', ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(fig_dir / 'comparison_all_models_eer.png', dpi=150)
    plt.close()
    print(f"✓ Combined EER comparison chart → {fig_dir / 'comparison_all_models_eer.png'}")


def _plot_accuracy_degradation(df_summary, fig_dir):
    """Bar chart: Accuracy trên clean vs. scenario nhiễu tệ nhất — mỗi detector."""
    import matplotlib.pyplot as plt
    import numpy as np

    detectors = sorted(df_summary['detector'].unique())

    # Accuracy clean
    clean_acc = {}
    for name in detectors:
        row = df_summary[(df_summary['detector'] == name) & (df_summary['scenario'] == 'clean')]
        clean_acc[name] = float(row['accuracy'].values[0]) if len(row) > 0 else float('nan')

    # Scenario nhiễu nhất = scenario có EER cao nhất trung bình
    noise_scenarios = df_summary[df_summary['scenario'] != 'clean']['scenario'].unique()
    if len(noise_scenarios) == 0:
        return

    mean_eer_by_scen = (
        df_summary[df_summary['scenario'] != 'clean']
        .groupby('scenario')['eer'].mean()
    )
    worst_scenario = mean_eer_by_scen.idxmax()

    noisy_acc = {}
    for name in detectors:
        row = df_summary[(df_summary['detector'] == name) & (df_summary['scenario'] == worst_scenario)]
        noisy_acc[name] = float(row['accuracy'].values[0]) if len(row) > 0 else float('nan')

    # Vẽ
    x = np.arange(len(detectors))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    clean_vals = [clean_acc.get(d, float('nan')) for d in detectors]
    noisy_vals = [noisy_acc.get(d, float('nan')) for d in detectors]

    bars_clean = ax.bar(x - w/2, clean_vals, w, label='Clean', color='#2ecc71', alpha=0.85)
    bars_noisy = ax.bar(x + w/2, noisy_vals, w, label=f'Worst noise ({worst_scenario})',
                        color='#e74c3c', alpha=0.85)

    # Highlight ensemble bars
    for i, name in enumerate(detectors):
        if 'ensemble' in name.lower():
            bars_clean[i].set_edgecolor('black')
            bars_clean[i].set_linewidth(2.5)
            bars_noisy[i].set_edgecolor('black')
            bars_noisy[i].set_linewidth(2.5)

    # Giá trị trên cột
    for bar in list(bars_clean) + list(bars_noisy):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy  (↑ better)', fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title('Accuracy: Clean vs. Worst Noise — All Detectors\n(bold outline = Fusion Pipeline)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'accuracy_degradation_all_models.png', dpi=150)
    plt.close()
    print(f"✓ Accuracy degradation chart → {fig_dir / 'accuracy_degradation_all_models.png'}")


def _print_summary_table(df_summary):
    """In bảng tổng hợp EER và Accuracy (clean + avg noise) ra console."""
    detectors = sorted(df_summary['detector'].unique())
    noise_df  = df_summary[df_summary['scenario'] != 'clean']
    clean_df  = df_summary[df_summary['scenario'] == 'clean']

    avg_noise = noise_df.groupby('detector')[['eer', 'accuracy', 'f1']].mean()
    clean_row = clean_df.groupby('detector')[['eer', 'accuracy', 'f1']].mean()

    header = f"\n{'Detector':<22} {'Clean Acc':>10} {'Clean EER':>10} {'Noisy Acc':>10} {'Noisy EER':>10} {'Noisy F1':>10}"
    print("\n" + "="*80)
    print("  NOISE ROBUSTNESS SUMMARY — Clean vs. Average Noisy Conditions")
    print("="*80)
    print(header)
    print("-"*80)
    for name in detectors:
        c_acc = clean_row.loc[name, 'accuracy'] if name in clean_row.index else float('nan')
        c_eer = clean_row.loc[name, 'eer']      if name in clean_row.index else float('nan')
        n_acc = avg_noise.loc[name, 'accuracy'] if name in avg_noise.index else float('nan')
        n_eer = avg_noise.loc[name, 'eer']      if name in avg_noise.index else float('nan')
        n_f1  = avg_noise.loc[name, 'f1']       if name in avg_noise.index else float('nan')
        star  = ' ★' if 'ensemble' in name.lower() else ''
        print(f"  {name+star:<22} {c_acc:>10.3f} {c_eer:>10.3f} {n_acc:>10.3f} {n_eer:>10.3f} {n_f1:>10.3f}")
    print("="*80)
    print("  ★ = VietGuardEnsemble (Fusion Pipeline — mô hình đề xuất)")
    print("="*80 + "\n")


def evaluate(args):
    metadata = load_metadata()

    # choose test_unseen split if available
    if 'split' in metadata.columns:
        pool = metadata[metadata['split'].str.contains('test', na=False)].copy()
    else:
        pool = metadata.copy()

    # sample balanced subset
    labels = pool['label'].unique().tolist()
    by_label = {lab: pool[pool['label'] == lab] for lab in labels}
    n_per_label = max(1, args.n_samples // max(1, len(labels)))
    samples = []
    for lab, df in by_label.items():
        if len(df) == 0:
            continue
        samples.extend(df.sample(n=min(n_per_label, len(df)), random_state=42).to_dict('records'))

    if len(samples) == 0:
        raise RuntimeError('No samples selected for evaluation')

    # load models
    detectors = {}
    single_model_p = Path('vispoofdb/experiments/svm_on_wav2vec.pkl')
    single_scaler_p = Path('vispoofdb/experiments/svm_on_wav2vec_scaler.pkl')
    if single_model_p.exists() and single_scaler_p.exists():
        print('Loading single SVM-on-Wav2Vec model...')
        detectors['single'] = SingleWav2VecDetector(str(single_model_p), str(single_scaler_p))
    else:
        print('Single wav2vec model not found; skipping single-model evaluation')

    # ensemble if possible
    try:
        from vispoofdb.models.ensemble_system import VietGuardEnsemble
        try:
            detectors['ensemble'] = VietGuardEnsemble(models_dir='vispoofdb/models_saved')
            print('Loaded VietGuardEnsemble')
        except Exception as e:
            print('Ensemble model not available or incomplete:', e)
    except Exception:
        print('Could not import VietGuardEnsemble; skipping ensemble')

    if len(detectors) == 0:
        # if user asked for aasist model type, we'll attempt to load later
        print('No detectors loaded yet; will try AASIST if requested')

    # augmentation scenarios — choose augmentor
    scenarios = [('clean', None)]
    if args.augmentor == 'audiomentations' and AUDIOMENTATIONS_AVAILABLE:
        # import conditions similar to user's noise_eval.py
        CONDITIONS = {
            'phone_call': Compose([
                LowPassFilter(min_cutoff_freq=3000, max_cutoff_freq=3400, p=1.0),
                HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=300, p=1.0),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.8),
                Mp3Compression(min_bitrate=16, max_bitrate=32, p=1.0),
            ]),
            'zalo_poor_network': Compose([
                AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=1.0),
                Mp3Compression(min_bitrate=8, max_bitrate=24, p=1.0),
                LowPassFilter(min_cutoff_freq=4000, max_cutoff_freq=6000, p=1.0),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
            ]),
            'voice_message': Compose([
                Mp3Compression(min_bitrate=32, max_bitrate=64, p=1.0),
                AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.003, p=0.7),
                TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
            ]),
            'noisy_environment': Compose([
                AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1.0),
                RoomSimulator(
                    min_size_x=3.6, max_size_x=10.0,
                    min_size_y=3.6, max_size_y=10.0,
                    min_size_z=2.4, max_size_z=4.0,
                    min_source_x=0.1, max_source_x=3.5,
                    min_source_y=0.1, max_source_y=3.5,
                    min_source_z=1.0, max_source_z=2.1,
                    min_mic_distance=0.15, max_mic_distance=0.35,
                    p=0.9
                ),
                Mp3Compression(min_bitrate=32, max_bitrate=96, p=0.5),
            ]),
        }
        for k in CONDITIONS.keys():
            scenarios.append((k, {'type': 'audiomentations', 'aug': CONDITIONS[k]}))
    else:
        # basic scenarios (simple noise / telephone / mp3 via ffmpeg)
        for snr in [20, 10, 0]:
            scenarios.append((f'noise_snr_{snr}', {'type': 'noise', 'snr': snr}))
        scenarios.append(('telephone', {'type': 'telephone'}))
        if has_ffmpeg():
            for br in ['128k', '64k', '32k']:
                scenarios.append((f'mp3_{br}', {'type': 'mp3', 'bitrate': br}))
        else:
            print('ffmpeg not found — skipping codec-based corruptions')

    out_dir = Path('vispoofdb/experiments/noise_eval')
    fig_dir = Path('vispoofdb/figures/noise')
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    records = []

    for scen_name, scen in scenarios:
        print('Running scenario:', scen_name)
        y_trues = []
        y_scores = {k: [] for k in detectors.keys()}

        for row in samples:
            src_path = build_file_path(row)
            if not os.path.exists(src_path):
                print('Missing file, skipping:', src_path)
                continue

            # load clean audio
            y, sr = librosa.load(src_path, sr=16000)

            # create corrupted version in temp wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
                tmp_out = tmpf.name

            if scen is None:
                sf.write(tmp_out, y, sr)
            else:
                t = scen['type']
                if t == 'noise':
                    y2 = add_white_noise(y, scen['snr'])
                    sf.write(tmp_out, y2, sr)
                elif t == 'telephone':
                    y2 = bandpass_telephone(y, sr)
                    sf.write(tmp_out, y2, sr)
                elif t == 'mp3':
                    tmp_in = tmp_out + '.in.wav'
                    sf.write(tmp_in, y, sr)
                    try:
                        codec_mp3_roundtrip(tmp_in, tmp_out, bitrate=scen['bitrate'])
                    finally:
                        try:
                            os.remove(tmp_in)
                        except Exception:
                            pass
                elif t == 'audiomentations':
                    try:
                        aug = scen['aug']
                        noisy = aug(samples=y, sample_rate=sr)
                        sf.write(tmp_out, noisy, sr)
                    except Exception as e:
                        print('audiomentations failed:', e)
                        sf.write(tmp_out, y, sr)
                else:
                    sf.write(tmp_out, y, sr)

            # run detectors
            for name, det in detectors.items():
                try:
                    res = det.predict_audio(tmp_out)
                    if not res.get('success', True):
                        score = np.nan
                    else:
                        score = float(res.get('confidence_ai', np.nan))
                except Exception as e:
                    print('Detector error', name, e)
                    score = np.nan
                y_scores[name].append(score)

            # true label: map 'real'->0, others ->1
            true = 1 if row.get('label') != 'real' else 0
            y_trues.append(true)

            # remove temp
            try:
                os.remove(tmp_out)
            except Exception:
                pass

        # compute metrics per detector
        for name in detectors.keys():
            scores = np.array(y_scores[name], dtype=float)
            mask = ~np.isnan(scores)
            if mask.sum() == 0:
                continue
            y_true_arr = np.array(y_trues)[mask]
            y_score_arr = scores[mask]
            y_pred = (y_score_arr >= 0.5).astype(int)
            acc = accuracy_score(y_true_arr, y_pred)
            prec = precision_score(y_true_arr, y_pred, zero_division=0)
            rec = recall_score(y_true_arr, y_pred, zero_division=0)
            f1 = f1_score(y_true_arr, y_pred, zero_division=0)
            eer = compute_eer(y_true_arr, y_score_arr)

            records.append({
                'scenario': scen_name,
                'detector': name,
                'n_samples': int(mask.sum()),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'eer': eer,
            })

            # save raw scores for this scenario+detector
            df_scores = pd.DataFrame({
                'file_id': [r.get('file_id') for r in samples][:len(y_score_arr)],
                'true_label': y_true_arr,
                'score': y_score_arr,
            })
            df_scores.to_csv(out_dir / f'scores_{name}_{scen_name}.csv', index=False)

    df_summary = pd.DataFrame(records)
    df_summary.to_csv(out_dir / 'noise_eval_summary.csv', index=False)

    # simple plots: EER by scenario per detector
    for name in set(df_summary['detector']):
        sub = df_summary[df_summary['detector'] == name]
        plt.figure(figsize=(8, 4))
        plt.plot(sub['scenario'], sub['eer'], marker='o')
        plt.xticks(rotation=40)
        plt.xlabel('Scenario')
        plt.ylabel('EER')
        plt.title(f'EER by corruption — {name}')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f'eer_{name}.png')
        plt.close()

    print('Done. Results in', out_dir, 'and', fig_dir)


def main():
    parser = argparse.ArgumentParser(description='Evaluate all models under noise conditions')
    parser.add_argument('--n-samples', type=int, default=50, help='Total samples to evaluate (balanced by label)')
    parser.add_argument('--augmentor', choices=['audiomentations', 'simple'], default='audiomentations', help='Augmentor type')
    args = parser.parse_args()
    args.augmentor = args.augmentor if AUDIOMENTATIONS_AVAILABLE else 'simple'
    
    print("\n" + "="*70)
    print("  EVALUATING ALL MODELS UNDER NOISE CONDITIONS")
    print("="*70)
    
    try:
        evaluate_all_models(args)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
