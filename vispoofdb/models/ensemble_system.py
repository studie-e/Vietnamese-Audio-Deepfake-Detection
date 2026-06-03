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
    def __init__(self, models_dir='models_saved'):
        self.models_dir = models_dir

        # Load Models
        self.svm_lfcc   = joblib.load(os.path.join(models_dir, 'svm_lfcc_model.pkl'))
        self.scaler_lfcc = joblib.load(os.path.join(models_dir, 'scaler_lfcc.pkl'))

        self.mlp_w2v   = joblib.load(os.path.join(models_dir, 'mlp_wav2vec_model.pkl'))
        self.scaler_w2v = joblib.load(os.path.join(models_dir, 'scaler_wav2vec.pkl'))

        self.svm_mfcc  = joblib.load(os.path.join(models_dir, 'svm_voice_model.pkl'))
        self.scaler_svm = joblib.load(os.path.join(models_dir, 'scaler_final.pkl'))

        self.xgb_model = joblib.load(os.path.join(models_dir, 'best_xgboost.pkl'))

        self.mlp_model = joblib.load(os.path.join(models_dir, 'best_mlp.pkl'))
        self.scaler_mlp = joblib.load(os.path.join(models_dir, 'scaler_mlp.pkl'))

        # Load Wav2Vec2 — fallback gracefully neu khong tai duoc (offline / cache loi)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.w2v_model  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            self.w2v_model.to(self.device)
            self.wav2vec_available = True
        except Exception as e:
            print(f"[WARN] Khong the tai Wav2Vec2: {e}")
            print("[WARN] Ensemble se chay voi 4 model (khong co Wav2Vec2).")
            self.processor = None
            self.w2v_model  = None
            self.wav2vec_available = False

    # ── Feature extraction ───────────────────────────────────────────────────

    def _extract_lfcc(self, y, sr):
        lfccs = lfcc(sig=y, fs=sr, num_ceps=40, nfilts=128)
        return np.mean(lfccs, axis=0).reshape(1, -1)

    def _extract_wav2vec(self, y, sr):
        """Tra ve None neu Wav2Vec2 khong san sang."""
        if not self.wav2vec_available:
            return None
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.w2v_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features.reshape(1, -1)

    def _extract_mfcc_40(self, y, sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0).reshape(1, -1)

    def _extract_mfcc_480(self, y, sr):
        mfccs  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta  = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        combined = np.vstack([mfccs, delta, delta2])
        mean_val = np.mean(combined, axis=1)
        std_val  = np.std(combined,  axis=1)
        max_val  = np.max(combined,  axis=1)
        min_val  = np.min(combined,  axis=1)
        return np.concatenate([mean_val, std_val, max_val, min_val]).reshape(1, -1)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=16000)

            feat_lfcc     = self._extract_lfcc(y, sr)
            feat_w2v      = self._extract_wav2vec(y, sr)   # None neu Wav2Vec2 khong co
            feat_mfcc_40  = self._extract_mfcc_40(y, sr)
            feat_mfcc_480 = self._extract_mfcc_480(y, sr)

            # Xac suat tung model (4 model base luon chay duoc)
            p_lfcc = float(self.svm_lfcc.predict_proba(self.scaler_lfcc.transform(feat_lfcc))[0][1])
            p_svm  = float(self.svm_mfcc.predict_proba(self.scaler_svm.transform(feat_mfcc_40))[0][1])
            p_xgb  = float(self.xgb_model.predict_proba(feat_mfcc_480)[0][1])
            p_mlp  = float(self.mlp_model.predict_proba(self.scaler_mlp.transform(feat_mfcc_40))[0][1])

            probs   = [p_lfcc, p_svm, p_xgb, p_mlp]
            details = [p_lfcc, p_svm, p_xgb, p_mlp]
            names   = ["LFCC+SVM", "MFCC+SVM", "XGBoost", "MFCC+MLP"]

            # Them Wav2Vec2 neu co
            if feat_w2v is not None:
                p_w2v = float(self.mlp_w2v.predict_proba(self.scaler_w2v.transform(feat_w2v))[0][1])
                # Insert tai vi tri 1 de giu thu tu hien thi: LFCC, W2V, SVM, XGB, MLP
                probs.insert(1, p_w2v)
                details.insert(1, p_w2v)
                names.insert(1, "Wav2Vec+MLP")

            # Soft Voting (trung binh dong deu theo so model thuc te)
            final_ai_prob = sum(probs) / len(probs)

            return {
                "success": True,
                "is_fake": bool(final_ai_prob >= 0.5),
                "confidence_ai": float(final_ai_prob),
                "details": details,
                "model_names": names,
                "wav2vec_available": self.wav2vec_available,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
