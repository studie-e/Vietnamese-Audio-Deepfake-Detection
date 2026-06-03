"""
XAI Engine — Vietnamese Audio Deepfake Detection
=================================================
Cung cấp giải thích SHAP cho từng model trong ensemble:
  - XGBoost     → TreeExplainer  (nhanh, chính xác)
  - SVM / MLP   → KernelExplainer (model-agnostic, lấy mẫu nền)

Feature groups được gán nhãn tiếng Việt để dễ đọc trên UI.
"""

import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Tên feature cho từng không gian đặc trưng
# ---------------------------------------------------------------------------

def _mfcc40_names() -> list[str]:
    """40 hệ số MFCC trung bình."""
    return [f"MFCC-{i+1} (mean)" for i in range(40)]

def _mfcc480_names() -> list[str]:
    """
    XGBoost dùng 480 features:
    [mean(MFCC), mean(Δ), mean(ΔΔ), std(MFCC), std(Δ), std(ΔΔ),
     max(MFCC), max(Δ), max(ΔΔ), min(MFCC), min(Δ), min(ΔΔ)]
    Mỗi nhóm thống kê × 3 loại (mfcc/delta/delta2) × 40 hệ số = 480.
    """
    names = []
    stats   = ["mean", "std", "max", "min"]
    signals = ["MFCC", "Δ-MFCC", "ΔΔ-MFCC"]
    for stat in stats:
        for sig in signals:
            for i in range(40):
                names.append(f"{sig}-{i+1} ({stat})")
    return names  # 4*3*40 = 480

def _lfcc40_names() -> list[str]:
    """40 hệ số LFCC trung bình."""
    return [f"LFCC-{i+1} (mean)" for i in range(40)]

def _wav2vec_names(n: int = 768) -> list[str]:
    """Wav2Vec2 hidden-state vector (768 chiều)."""
    # Nhóm theo từng 64 chiều để giảm noise khi hiển thị top-k
    return [f"W2V-dim {i+1}" for i in range(n)]


# ---------------------------------------------------------------------------
# Nhóm feature thân thiện (dùng để aggregate SHAP về mức cao hơn)
# ---------------------------------------------------------------------------

MFCC_GROUP_LABELS = {
    "MFCC coefficients (1-13)":  list(range(0, 13)),    # formant/phoneme
    "MFCC coefficients (14-26)": list(range(13, 26)),   # spectral detail
    "MFCC coefficients (27-40)": list(range(26, 40)),   # fine-grain
}


# ---------------------------------------------------------------------------
# Lớp chính
# ---------------------------------------------------------------------------

class AudioXAIExplainer:
    """
    Gói gọn toàn bộ logic SHAP cho VietGuard ensemble.

    Parameters
    ----------
    ensemble : VietGuardEnsemble
        Instance đã load models.
    n_background : int
        Số mẫu nền dùng cho KernelExplainer (tăng → chính xác hơn nhưng chậm hơn).
        Mặc định 10 — đủ cho demo, cân bằng tốc độ & chất lượng.
    """

    def __init__(self, ensemble, n_background: int = 10):
        self.ens = ensemble
        self.n_background = n_background
        # Cache explainer objects — xây một lần, tái sử dụng mãi.
        self._explainer_lfcc   = None
        self._explainer_svm_mfcc = None
        self._explainer_w2v    = None
        self._explainer_mlp    = None
        self._explainer_xgb    = None

    # ------------------------------------------------------------------
    # Background data — zeros là điểm trung tính ổn định
    # ------------------------------------------------------------------

    def _get_background(self, n_features: int, n_bg: int | None = None) -> np.ndarray:
        """Sinh background data bằng zeros."""
        return np.zeros((n_bg or self.n_background, n_features))

    # ------------------------------------------------------------------
    # Khởi tạo (lazy) các explainer — chỉ build lần đầu
    # ------------------------------------------------------------------

    def _init_explainers(self, w2v_dim: int = 768):
        """Build XGBoost TreeExplainer — chỉ 1 lần, cache lại."""
        if self._explainer_xgb is not None:
            return

        # XGBoost TreeExplainer — mặc định dùng raw (log-odds)
        # Trong app.py sẽ apply sigmoid để hiển thị đúng xác suất
        self._explainer_xgb = shap.TreeExplainer(self.ens.xgb_model)

    # ------------------------------------------------------------------
    # Hàm predict_proba wrapper cho từng model (KernelExplainer cần nó)
    # ------------------------------------------------------------------

    def _pred_svm_lfcc(self, X):
        return self.ens.svm_lfcc.predict_proba(
            self.ens.scaler_lfcc.transform(X))

    def _pred_mlp_w2v(self, X):
        return self.ens.mlp_w2v.predict_proba(
            self.ens.scaler_w2v.transform(X))

    def _pred_svm_mfcc(self, X):
        return self.ens.svm_mfcc.predict_proba(
            self.ens.scaler_svm.transform(X))

    def _pred_xgb(self, X):
        return self.ens.xgb_model.predict_proba(X)

    def _pred_mlp(self, X):
        return self.ens.mlp_model.predict_proba(
            self.ens.scaler_mlp.transform(X))

    # ------------------------------------------------------------------
    # Core: tính SHAP values cho 1 instance
    # ------------------------------------------------------------------

    def explain(self, features: dict) -> dict:
        """
        Tính SHAP values cho tất cả 5 models.

        Parameters
        ----------
        features : dict
            Output của _extract_all_features(), gồm:
            {
              'lfcc':    np.ndarray (1, 40)
              'w2v':     np.ndarray (1, 768)
              'mfcc40':  np.ndarray (1, 40)
              'mfcc480': np.ndarray (1, 480)
            }

        Returns
        -------
        dict  keyed by model name, mỗi entry chứa:
            {
              'shap_values': np.ndarray   # shape (n_features,)
              'base_value':  float
              'feature_names': list[str]
              'top_k': list[dict]         # top 15 features sorted by |shap|
            }
        """
        w2v_dim = features["w2v"].shape[1]
        self._init_explainers(w2v_dim)

        results = {}

        # XGBoost TreeSHAP — chính xác, nhanh, đáng tin cậy
        results["XGBoost"] = self._explain_tree_cached(
            instance=features["mfcc480"],
            feature_names=_mfcc480_names(),
        )

        # Ensemble summary (dùng cho UI phiếu bầu)
        results["_ensemble_summary"] = self._ensemble_summary(results)

        return results

    # ------------------------------------------------------------------
    # KernelExplainer — dùng object đã cache, không tạo lại
    # ------------------------------------------------------------------

    @staticmethod
    def _to_1d(obj) -> np.ndarray:
        """
        Chuyển bất kỳ object nào (ndarray, list, scalar, 0-d array…)
        thành numpy array 1-D float64 một cách an toàn.
        """
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0:
                return obj.reshape(1)
            return obj.astype(float).ravel()
        # list hoặc iterable — dùng vòng lặp để tránh lỗi ragged/0-d
        try:
            return np.array([float(x) for x in np.asarray(obj).ravel()])
        except Exception:
            return np.array([float(obj)])

    @staticmethod
    def _extract_sv(shap_vals, class_idx: int = 1) -> np.ndarray:
        """
        Trích xuất vector SHAP 1-D cho 1 sample từ mọi format:
          - list of arrays/lists : [cls0, cls1]  → lấy cls[class_idx][0]
          - ndarray 3D           : (n_cls, n_smp, n_feat) → [class_idx, 0, :]
          - ndarray 2D           : (n_smp, n_feat)        → [0, :]
          - ndarray 1D           : (n_feat,)              → as-is
        """
        _to1d = AudioXAIExplainer._to_1d

        if isinstance(shap_vals, (list, tuple)):
            idx  = class_idx if len(shap_vals) > class_idx else 0
            item = shap_vals[idx]
            # item có thể là ndarray (1,40), list[40], hoặc 0-d array
            item_arr = _to1d(item)
            # Nếu item là (1, n_features) — lấy hàng đầu
            if isinstance(item, np.ndarray) and item.ndim == 2:
                return item[0].astype(float)
            return item_arr

        # ndarray path
        try:
            arr = np.asarray(shap_vals, dtype=float)
        except (TypeError, ValueError):
            arr = np.array(shap_vals, dtype=object)
            arr = np.array([float(x) for x in arr.ravel()])

        if arr.ndim == 3:
            idx = class_idx if arr.shape[0] > class_idx else 0
            return arr[idx, 0, :].astype(float)
        elif arr.ndim == 2:
            return arr[0, :].astype(float)
        else:
            return arr.astype(float).ravel()


    @staticmethod
    def _explain_kernel_cached(explainer, instance, feature_names,
                               nsamples: int = 32) -> dict:
        n_features = instance.shape[1] if instance.ndim > 1 else instance.shape[0]

        # "aic"/"bic" dùng LassoLarsIC — yêu cầu nsamples > n_features.
        # l1_reg=0 dùng OLS — không có ràng buộc, phân phối đều cho các feature.
        l1_reg = 0

        # Cap nsamples ở 200 để không quá chậm
        safe_nsamples = min(nsamples, 200)

        shap_vals = explainer.shap_values(instance, nsamples=safe_nsamples,
                                          l1_reg=l1_reg, silent=True)

        ev = explainer.expected_value

        # expected_value: scalar / list / ndarray tuỳ version SHAP
        def _safe_ev(ev_raw, class_idx=1):
            if isinstance(ev_raw, (list, np.ndarray)):
                arr = np.asarray(ev_raw).ravel()
                return float(arr[class_idx]) if len(arr) > class_idx else float(arr[0])
            return float(ev_raw)

        # Xác định class_idx: nếu 2 class → lấy class 1 (fake)
        if isinstance(shap_vals, list):
            class_idx = 1 if len(shap_vals) > 1 else 0
        else:
            arr = np.asarray(shap_vals)
            class_idx = 1 if (arr.ndim == 3 and arr.shape[0] > 1) else 0

        sv   = AudioXAIExplainer._extract_sv(shap_vals, class_idx)
        base = _safe_ev(ev, class_idx)

        return AudioXAIExplainer._pack(sv, base, feature_names)

    # ------------------------------------------------------------------
    # TreeExplainer — dùng object đã cache
    # ------------------------------------------------------------------

    def _explain_tree_cached(self, instance, feature_names) -> dict:
        shap_vals = self._explainer_xgb.shap_values(instance)
        sv = self._extract_sv(shap_vals, class_idx=1)

        ev = self._explainer_xgb.expected_value
        def _safe_ev(ev_raw, class_idx=1):
            if isinstance(ev_raw, (list, np.ndarray)):
                arr = np.asarray(ev_raw).ravel()
                return float(arr[class_idx]) if len(arr) > class_idx else float(arr[0])
            return float(ev_raw)

        base = _safe_ev(ev, class_idx=1)
        return self._pack(sv, base, feature_names)

    # ------------------------------------------------------------------
    # Đóng gói kết quả + top-k
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(sv: np.ndarray, base: float, feature_names: list) -> dict:
        sv = np.asarray(sv).ravel()
        # Đảm bảo sv và feature_names khớp nhau
        n = min(len(sv), len(feature_names))
        sv = sv[:n]
        feature_names = feature_names[:n]

        abs_sv = np.abs(sv)
        # Top 15 features theo |SHAP|
        top_idx = np.argsort(abs_sv)[::-1][:15]
        top_k = [
            {
                "rank": int(r + 1),
                "feature": feature_names[i],
                "shap_value": float(sv[i]),
                "abs_shap": float(abs_sv[i]),
                "direction": "tăng nghi ngờ AI 🔴" if sv[i] > 0 else "giảm nghi ngờ AI 🟢"
            }
            for r, i in enumerate(top_idx)
        ]
        return {
            "shap_values":   sv,
            "base_value":    base,
            "feature_names": feature_names,
            "top_k":         top_k
        }

    # ------------------------------------------------------------------
    # Ensemble summary: tổng hợp insight từ tất cả model
    # ------------------------------------------------------------------

    @staticmethod
    def _ensemble_summary(results: dict) -> dict:
        """
        Tổng hợp tầm quan trọng tương đối của từng model
        dựa trên độ lớn SHAP trung bình (mean |SHAP|).
        """
        model_keys = [k for k in results if not k.startswith("_")]
        model_importance = {}
        for k in model_keys:
            sv = results[k]["shap_values"]
            model_importance[k] = float(np.mean(np.abs(sv)))

        total = sum(model_importance.values()) or 1.0
        model_weight = {k: v / total for k, v in model_importance.items()}

        # Nhận xét tự động
        dominant = max(model_weight, key=model_weight.get)
        notes = []
        if model_weight.get("XGBoost", 0) > 0.35:
            notes.append("⚡ XGBoost đang chi phối quyết định — delta-MFCC là đặc trưng chủ đạo.")
        if model_weight.get("Wav2Vec + MLP", 0) > 0.30:
            notes.append("🤖 Wav2Vec2 có ảnh hưởng lớn — ngữ nghĩa âm học cấp cao được phát hiện.")
        if model_weight.get("LFCC + SVM", 0) > 0.25:
            notes.append("🎵 LFCC/SVM nhạy với phổ tần số tuyến tính — dấu hiệu codec hoặc vocoder.")
        if not notes:
            notes.append(f"📊 {dominant} có ảnh hưởng lớn nhất trong lần phân tích này.")

        return {
            "model_weights": model_weight,
            "dominant_model": dominant,
            "notes": notes
        }


# ---------------------------------------------------------------------------
# Helper: trích xuất tất cả features từ 1 file âm thanh đã load
# ---------------------------------------------------------------------------

def extract_all_features(ensemble, y: np.ndarray, sr: int) -> dict:
    """
    Tiện ích để trích xuất tất cả 4 loại feature vector
    từ dữ liệu audio đã load, dùng chung với ensemble.

    Returns
    -------
    dict với keys: 'lfcc', 'w2v', 'mfcc40', 'mfcc480'
    """
    return {
        "lfcc":    ensemble._extract_lfcc(y, sr),
        "w2v":     ensemble._extract_wav2vec(y, sr),
        "mfcc40":  ensemble._extract_mfcc_40(y, sr),
        "mfcc480": ensemble._extract_mfcc_480(y, sr),
    }
