"""Evaluate models under real-world noise augmentations.

Creates corrupted variants (additive noise at several SNRs, MP3 compression at several bitrates,
and telephone bandpass) for a random subset of test files and reports performance degradation
against clean audio.

Usage:
    python vispoofdb/scripts/eval_noise_augmentation.py --n-samples 50

Notes:
 - Requires `ffmpeg` on PATH for codec-based corruptions (MP3). If not available, codec tests are skipped.
 - By default evaluates the single SVM-on-Wav2Vec model saved under
   `vispoofdb/experiments/svm_on_wav2vec.pkl`. If the ensemble models exist, it will also evaluate the
   `VietGuardEnsemble` from `src.data_processing.ensemble_system`.
"""
import os
import argparse
import tempfile
import shutil
import subprocess
import random
import math
import warnings
from pathlib import Path

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
    # y_true: binary (1=ai/fake), y_score: probability of class 1
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # find point where fpr and fnr are closest
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer


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
    
    # LFCC features
    try:
        lfcc_data = np.load(data_dir / 'features_lfcc' / 'X_lfcc.npy')
        mean_lfcc = np.mean(lfcc_data, axis=0)
        std_lfcc = np.std(lfcc_data, axis=0)
        
        def extract_lfcc(y, sr):
            import librosa
            lfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512)
            return np.concatenate([np.mean(lfcc, axis=1), np.std(lfcc, axis=1)])
        
        extractors['lfcc'] = extract_lfcc
    except:
        pass
    
    # MFCC40 features
    try:
        def extract_mfcc40(y, sr):
            import librosa
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
        
        extractors['mfcc40'] = extract_mfcc40
    except:
        pass
    
    # WAV2VEC features
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
    # prefer vispoofdb metadata then fallback to top-level data/metadata.csv
    p1 = Path('vispoofdb/data/clean_data/metadata.csv')
    p2 = Path('data/metadata.csv')
    if p1.exists():
        return pd.read_csv(p1)
    elif p2.exists():
        return pd.read_csv(p2)
    else:
        raise FileNotFoundError('metadata.csv not found')


def build_file_path(row):
    # Try clean_data first, then fallback to raw
    p_clean = Path('vispoofdb/data/clean_data') / row['file_path']
    if p_clean.exists():
        return str(p_clean.resolve())
    p_raw = Path('vispoofdb/data/raw') / row['file_path']
    if p_raw.exists():
        return str(p_raw.resolve())
    return str(p_clean.resolve())


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
        ('xgboost_mfcc', models_dir / 'best_xgboost.pkl', None, 'mfcc40'),
        ('mlp_wav2vec', models_dir / 'mlp_wav2vec_model.pkl', models_dir / 'scaler_wav2vec.pkl', 'wav2vec'),
        ('svm_tone', models_dir / 'svm_tone_model.pkl', models_dir / 'scaler_tone.pkl', 'mfcc40'),
        ('xgboost_tone', models_dir / 'xgboost_tone_model.pkl', None, 'mfcc40'),
        ('svm_fusion', models_dir / 'svm_tone_fusion_model.pkl', models_dir / 'scaler_fusion.pkl', 'mfcc40'),
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
    
    # Plot EER by scenario for each detector
    for name in set(df_summary['detector']):
        sub = df_summary[df_summary['detector'] == name]
        plt.figure(figsize=(10, 5))
        plt.plot(sub['scenario'], sub['eer'], marker='o', linewidth=2, markersize=8)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Scenario', fontsize=11)
        plt.ylabel('EER', fontsize=11)
        plt.title(f'EER by Corruption — {name}', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f'eer_{name}.png', dpi=150)
        plt.close()
    
    print(f"\n✓ Results saved to {out_dir}")
    print(f"✓ Plots saved to {fig_dir}")
    print(f"✓ Summary CSV: {out_dir / 'noise_eval_summary_all_models.csv'}")


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
        from src.data_processing.ensemble_system import VietGuardEnsemble
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
