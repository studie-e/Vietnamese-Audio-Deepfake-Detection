"""
VietGuardEnsemble
=================
Ensemble 3 model dai dien 3 nhom feature khac nhau:

  1. SVM + LFCC        — Spectral co dien (lightweight baseline)
  2. XGBoost + MFCC-Delta — Temporal + gradient boosting (bat thay doi theo thoi gian)
  3. MLP + Wav2Vec2    — Deep semantic features (optional, fallback neu khong tai duoc)

Ly do chon 3 nhom nay:
  - Moi nhom nhin audio tu goc do khac nhau -> bo sung cho nhau
  - Tranh du thua (truoc day co 2 SVM + 2 MLP tren cung feature MFCC)
  - Ensemble co narrative ro rang cho bao cao / seminar
"""

import os
import librosa
import numpy as np
import joblib
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from spafe.features.lfcc import lfcc
import warnings
warnings.filterwarnings('ignore')


class VietGuardEnsemble:
    """
    Ensemble 3 model, weighted soft voting theo EER validation.

    Trong so (inverse-EER weighting) — mo hinh tot hon duoc vote nhieu hon:
        - Spectral : LFCC  (40 chieu)       → SVM        w ≈ 2.86
        - Temporal : MFCC-Delta (480 chieu) → XGBoost    w ≈ 6.67
        - Deep     : Wav2Vec2 (768 chieu)   → MLP        w ≈ 20.0

    Cong thuc:
        P_final = (w_lfcc*P_lfcc + w_xgb*P_xgb + w_w2v*P_w2v) / (w_lfcc + w_xgb + w_w2v)

    Nguyen ly: w_i = 1 / max(EER_i, 0.01)
    EER validation: SVM+LFCC=0.35, XGBoost+MFCC=0.15, MLP+Wav2Vec2=0.05
    """

    # Trong so mac dinh dua tren EER validation (co the ghi de qua constructor)
    DEFAULT_WEIGHTS = {
        'svm_lfcc': 1.0 / 0.35,   # ≈ 2.86
        'xgb_mfcc': 1.0 / 0.15,   # ≈ 6.67
        'mlp_w2v':  1.0 / 0.05,   # = 20.0
    }

    def __init__(self, models_dir='models_saved', weights=None):
        self.models_dir = models_dir
        # Trong so: dung DEFAULT_WEIGHTS hoac truyen tu ngoai vao
        self._weights = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)

        # --- Nhom 1: Spectral co dien ---
        self.svm_lfcc    = joblib.load(os.path.join(models_dir, 'svm_lfcc_model.pkl'))
        self.scaler_lfcc = joblib.load(os.path.join(models_dir, 'scaler_lfcc.pkl'))

        # --- Nhom 2: Temporal + ML ---
        self.xgb_model   = joblib.load(os.path.join(models_dir, 'best_xgboost.pkl'))
        # XGBoost duoc train truc tiep tren feature chua scale

        # --- Nhom 3: Deep semantic (Wav2Vec2) ---
        self.mlp_w2v     = joblib.load(os.path.join(models_dir, 'mlp_wav2vec_model.pkl'))
        self.scaler_w2v  = joblib.load(os.path.join(models_dir, 'scaler_wav2vec.pkl'))

        # Load Wav2Vec2 transformer — fallback gracefully neu khong tai duoc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.w2v_model  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.w2v_model.to(self.device)
            self.wav2vec_available = True
        except Exception as e:
            print(f"[WARN] Khong the tai Wav2Vec2: {e}")
            print("[WARN] Ensemble se chay voi 2 model (LFCC+SVM va XGBoost).")
            self.processor  = None
            self.w2v_model  = None
            self.wav2vec_available = False

    # ── Feature extraction ───────────────────────────────────────────────────

    def _extract_lfcc(self, y, sr):
        """LFCC 40 chieu — dai dien Spectral co dien."""
        lfccs = lfcc(sig=y, fs=sr, num_ceps=40, nfilts=128)
        return np.mean(lfccs, axis=0).reshape(1, -1)

    def _extract_mfcc_480(self, y, sr):
        """MFCC + Delta + Delta2 (480 chieu) — dai dien Temporal."""
        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta  = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        combined = np.vstack([mfccs, delta, delta2])
        return np.concatenate([
            np.mean(combined, axis=1),
            np.std(combined,  axis=1),
            np.max(combined,  axis=1),
            np.min(combined,  axis=1),
        ]).reshape(1, -1)

    def _extract_wav2vec(self, y, sr):
        """Wav2Vec2 embedding 768 chieu — tra ve None neu khong san sang."""
        if not self.wav2vec_available:
            return None
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.w2v_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features.reshape(1, -1)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=16000)

            # Trich xuat feature
            feat_lfcc    = self._extract_lfcc(y, sr)
            feat_mfcc480 = self._extract_mfcc_480(y, sr)
            feat_w2v     = self._extract_wav2vec(y, sr)  # None neu khong co

            # Xac suat tung model
            p_lfcc = float(self.svm_lfcc.predict_proba(
                self.scaler_lfcc.transform(feat_lfcc))[0][1])

            p_xgb  = float(self.xgb_model.predict_proba(feat_mfcc480)[0][1])

            # Weighted Soft Voting (inverse-EER weighting)
            model_keys = ['svm_lfcc', 'xgb_mfcc']
            prob_list   = [p_lfcc, p_xgb]
            names       = ["SVM + LFCC", "XGBoost + MFCC-Delta"]

            if feat_w2v is not None:
                p_w2v = float(self.mlp_w2v.predict_proba(
                    self.scaler_w2v.transform(feat_w2v))[0][1])
                model_keys.append('mlp_w2v')
                prob_list.append(p_w2v)
                names.append("MLP + Wav2Vec2")

            weights_used = np.array([self._weights.get(k, 1.0) for k in model_keys])
            probs_arr    = np.array(prob_list)
            final_ai_prob = float(np.dot(weights_used, probs_arr) / weights_used.sum())

            return {
                "success": True,
                "is_fake": bool(final_ai_prob >= 0.5),
                "confidence_ai": float(final_ai_prob),
                "details": prob_list,
                "model_names": names,
                "weights": weights_used.tolist(),
                "wav2vec_available": self.wav2vec_available,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
