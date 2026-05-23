"""
Vispoofdb XAI engine.

This module is tailored for the Vispoofdb dataset and the current app:
- TreeSHAP for XGBoost-style models when available.
- KernelSHAP for the Wav2Vec2 single-model fallback.
- Aggregated Wav2Vec2 groups to keep the UI readable.

The returned structure is compatible with the existing Streamlit visualizers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

try:
    import shap
except Exception as exc:  # pragma: no cover
    shap = None
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None


def _mfcc40_names() -> list[str]:
    return [f"MFCC-{i + 1}" for i in range(40)]


def _mfcc480_names() -> list[str]:
    names: list[str] = []
    stats = ["mean", "std", "max", "min"]
    signals = ["MFCC", "Δ-MFCC", "ΔΔ-MFCC"]
    for stat in stats:
        for sig in signals:
            for i in range(40):
                names.append(f"{sig}-{i + 1} ({stat})")
    return names


def _wav2vec_group_names(n_features: int, group_size: int = 64) -> list[str]:
    names: list[str] = []
    for start in range(0, n_features, group_size):
        end = min(start + group_size, n_features)
        names.append(f"W2V dims {start + 1}-{end}")
    return names


def _safe_predict_proba(model, scaler, X: np.ndarray) -> np.ndarray:
    X_scaled = scaler.transform(X) if scaler is not None else X
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_scaled)
    decision = model.decision_function(X_scaled)
    decision = np.asarray(decision, dtype=float).ravel()
    p_fake = 1.0 / (1.0 + np.exp(-decision))
    return np.column_stack([1.0 - p_fake, p_fake])


def extract_vispoofdb_features(detector, y: np.ndarray, sr: int) -> dict:
    """Extract the audio feature bundle used by the current app."""
    features = {}
    if hasattr(detector, "_extract_lfcc"):
        features["lfcc"] = detector._extract_lfcc(y, sr)
    if hasattr(detector, "_extract_wav2vec"):
        features["w2v"] = detector._extract_wav2vec(y, sr)
    if hasattr(detector, "_extract_mfcc_40"):
        features["mfcc40"] = detector._extract_mfcc_40(y, sr)
    if hasattr(detector, "_extract_mfcc_480"):
        features["mfcc480"] = detector._extract_mfcc_480(y, sr)
    return features


@dataclass
class ExplainResult:
    shap_values: np.ndarray
    base_value: float
    feature_names: list[str]
    top_k: list[dict]


class VispoofdbAudioXAI:
    """Explain audio predictions for the Vispoofdb app."""

    def __init__(self, detector, n_background: int = 8, kernel_nsamples: int = 24):
        if shap is None:  # pragma: no cover
            raise ImportError(f"shap is required for XAI: {_SHAP_IMPORT_ERROR}")

        self.detector = detector
        self.n_background = n_background
        self.kernel_nsamples = kernel_nsamples
        self._xgb_explainer = None
        self._w2v_kernel_explainer = None

    @staticmethod
    def _to_1d(obj) -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return obj.astype(float).ravel()
        try:
            return np.array([float(x) for x in np.asarray(obj).ravel()], dtype=float)
        except Exception:
            return np.array([float(obj)], dtype=float)

    @staticmethod
    def _extract_sv(shap_vals, class_idx: int = 1) -> np.ndarray:
        if isinstance(shap_vals, (list, tuple)):
            idx = class_idx if len(shap_vals) > class_idx else 0
            item = shap_vals[idx]
            item_arr = VispoofdbAudioXAI._to_1d(item)
            if isinstance(item, np.ndarray) and item.ndim == 2:
                return item[0].astype(float)
            return item_arr

        arr = np.asarray(shap_vals, dtype=float)
        if arr.ndim == 3:
            idx = class_idx if arr.shape[0] > class_idx else 0
            return arr[idx, 0, :].astype(float)
        if arr.ndim == 2:
            return arr[0, :].astype(float)
        return arr.ravel().astype(float)

    @staticmethod
    def _pack(sv: np.ndarray, base: float, feature_names: list[str]) -> dict:
        sv = np.asarray(sv, dtype=float).ravel()
        n = min(len(sv), len(feature_names))
        sv = sv[:n]
        feature_names = feature_names[:n]

        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:15]
        top_k = [
            {
                "rank": int(rank + 1),
                "feature": feature_names[i],
                "shap_value": float(sv[i]),
                "abs_shap": float(abs_sv[i]),
                "direction": "tăng nghi ngờ AI 🔴" if sv[i] > 0 else "giảm nghi ngờ AI 🟢",
            }
            for rank, i in enumerate(top_idx)
        ]
        return {
            "shap_values": sv,
            "base_value": float(base),
            "feature_names": feature_names,
            "top_k": top_k,
        }

    @staticmethod
    def _group_values(values: np.ndarray, names: list[str], group_size: int = 64) -> tuple[np.ndarray, list[str]]:
        values = np.asarray(values, dtype=float).ravel()
        grouped = []
        grouped_names = []
        for start in range(0, len(values), group_size):
            end = min(start + group_size, len(values))
            grouped.append(float(np.sum(values[start:end])))
            grouped_names.append(names[start // group_size] if start // group_size < len(names) else f"Group {start + 1}-{end}")
        return np.array(grouped, dtype=float), grouped_names

    @staticmethod
    def _summarize(results: dict) -> dict:
        model_keys = [k for k in results if not k.startswith("_")]
        weights = {}
        for key in model_keys:
            weights[key] = float(np.mean(np.abs(results[key]["shap_values"])))
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}
        dominant = max(weights, key=weights.get) if weights else "N/A"
        notes = [f"{dominant} đang chi phối quyết định hiện tại."] if weights else ["Chưa đủ dữ liệu để tổng hợp insight."]
        return {"model_weights": weights, "dominant_model": dominant, "notes": notes}

    def _ensure_tree_explainer(self):
        if self._xgb_explainer is None and hasattr(self.detector, "xgb_model"):
            self._xgb_explainer = shap.TreeExplainer(self.detector.xgb_model)

    def _explain_xgb(self, mfcc480: np.ndarray) -> dict | None:
        if not hasattr(self.detector, "xgb_model") or mfcc480 is None:
            return None
        self._ensure_tree_explainer()
        shap_vals = self._xgb_explainer.shap_values(mfcc480)
        sv = self._extract_sv(shap_vals, class_idx=1)
        ev = self._xgb_explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev).ravel()
            base = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            base = float(ev)
        return self._pack(sv, base, _mfcc480_names())

    def _explain_w2v(self, w2v: np.ndarray) -> dict | None:
        if w2v is None or not hasattr(self.detector, "model") or not hasattr(self.detector, "scaler"):
            return None
        if self._w2v_kernel_explainer is None:
            n_features = w2v.shape[1]
            background = np.zeros((self.n_background, n_features), dtype=float)

            def predict_fn(X):
                return _safe_predict_proba(self.detector.model, self.detector.scaler, np.asarray(X, dtype=float))

            self._w2v_kernel_explainer = shap.KernelExplainer(predict_fn, background)

        shap_vals = self._w2v_kernel_explainer.shap_values(
            w2v,
            nsamples=min(self.kernel_nsamples, 200),
            l1_reg=0,
            silent=True,
        )
        ev = self._w2v_kernel_explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev_arr = np.asarray(ev).ravel()
            base = float(ev_arr[1] if len(ev_arr) > 1 else ev_arr[0])
        else:
            base = float(ev)

        sv = self._extract_sv(shap_vals, class_idx=1)
        group_size = 64
        grouped_sv, grouped_names = self._group_values(sv, _wav2vec_group_names(len(sv), group_size), group_size=group_size)
        return self._pack(grouped_sv, base, grouped_names)

    def explain(self, y: np.ndarray, sr: int) -> dict:
        """Explain the current audio sample using whichever Vispoofdb model is available."""
        feats = extract_vispoofdb_features(self.detector, y, sr)
        results: dict = {}

        xgb_res = self._explain_xgb(feats.get("mfcc480"))
        if xgb_res is not None:
            results["XGBoost"] = xgb_res

        w2v_res = self._explain_w2v(feats.get("w2v"))
        if w2v_res is not None:
            results["Wav2Vec2"] = w2v_res

        results["_ensemble_summary"] = self._summarize(results)
        return results
