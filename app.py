import streamlit as st
import os
import tempfile
import librosa
import numpy as np
import joblib
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from src.data_processing.ensemble_system import VietGuardEnsemble
from vispoofdb.xai import VispoofdbAudioXAI
from src.xai.visualizer import (
    plot_waterfall,
    plot_ensemble_weights,
    plot_confidence_gauge,
    plot_mfcc_group_importance,
    plot_cross_model_heatmap,
)
from aasist_inference import AASISTDetector, AASISTXAIExplainer

# ── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Viet-Guard: Anti-Deepfake",
    page_icon="🛡️",
    layout="wide",
)

# ── CSS tuỳ chỉnh ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Nền tối toàn trang */
    .stApp { background-color: #0F172A; color: #F1F5F9; }
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #0F172A 60%, #1a1040 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; letter-spacing: 2px; }
    .main-header p  { color: #94A3B8; margin: 0.4rem 0 0; font-size: 1rem; }
    /* Card */
    .result-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin: 0.6rem 0;
        border-left: 4px solid #6366F1;
    }
    /* Metric override */
    div[data-testid="metric-container"] {
        background: #1E293B;
        border-radius: 10px;
        padding: 0.8rem;
        border: 1px solid #334155;
    }
    div[data-testid="metric-container"] label { color: #94A3B8 !important; }
    div[data-testid="metric-container"] div   { color: #F1F5F9 !important; }
    /* Tab */
    .stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 8px; gap: 24px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; border-radius: 6px; padding-left: 12px; padding-right: 12px; }
    .stTabs [aria-selected="true"] { background: #334155 !important; color: #F1F5F9 !important; }
    /* Spinner */
    .stSpinner > div { border-top-color: #6366F1 !important; }
    /* Note box */
    .note-box {
        background: #172033;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #E2E8F0;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ VIET-GUARD</h1>
    <p>Hệ thống phát hiện &amp; giải thích Giọng nói Deepfake tiếng Việt · Nhóm 17</p>
    <p style="font-size:0.82rem;color:#475569;margin-top:0.6rem;">
        Ensemble 5 mô hình AI · SHAP Explainability · Vietnamese Audio Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ── Inference mode selector ─────────────────────────────────────────────────
mode = st.selectbox(
    "Chế độ inference:",
    ["Ensemble (5 models)", "Single model — SVM Wav2Vec", "Deep Learning — AASIST"]
) 


# ── Helper: single-model wrapper ────────────────────────────────────────────
class SingleWav2VecDetector:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.w2v_model.to(self.device)

    def _extract_wav2vec(self, y, sr):
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.w2v_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return features.reshape(1, -1)

    def predict_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=16000)
        feat = self._extract_wav2vec(y, sr)
        try:
            if hasattr(self.model, "predict_proba"):
                p = float(self.model.predict_proba(self.scaler.transform(feat))[0][1])
            else:
                # fallback to decision_function then sigmoid
                df = float(self.model.decision_function(self.scaler.transform(feat))[0])
                p = 1.0 / (1.0 + np.exp(-df))
            return {
                "success": True,
                "is_fake": bool(p >= 0.5),
                "confidence_ai": float(p),
                "details": [float(p)],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system(mode_name="ensemble"):
    if mode_name == "single":
        model_p = os.path.join("vispoofdb", "models_saved", "mlp_on_wav2vec.pkl")
        scaler_p = os.path.join("vispoofdb", "models_saved", "scaler_wav2vec.pkl")
        if not os.path.exists(model_p) or not os.path.exists(scaler_p):
            raise FileNotFoundError("Không tìm thấy SVM Wav2Vec model/scaler")
        det = SingleWav2VecDetector(model_p, scaler_p)
        xai = VispoofdbAudioXAI(det, n_background=8)
        return det, xai
    
    elif mode_name == "aasist":
        model_p = os.path.join("vispoofdb", "models_saved", "aasist_best_model.pth")
        if not os.path.exists(model_p):
            raise FileNotFoundError(f"Không tìm thấy AASIST model: {model_p}")
        det = AASISTDetector(model_p)
        xai = AASISTXAIExplainer(det)
        return det, xai
    
    else:  # ensemble
        ens = VietGuardEnsemble(models_dir="vispoofdb/models_saved")
        xai = VispoofdbAudioXAI(ens, n_background=8)
        return ens, xai

with st.spinner("⏳ Đang khởi động hệ thống AI…"):
    try:
        if mode.startswith("Deep"):
            detector, explainer = load_system(mode_name="aasist")
        elif mode.startswith("Single"):
            detector, explainer = load_system(mode_name="single")
        else:
            detector, explainer = load_system(mode_name="ensemble")
    except FileNotFoundError as e:
        st.warning(f"Model bị thiếu: {e}")
        st.info("💡 Vui lòng chọn chế độ khác hoặc huấn luyện model trước.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khởi tạo hệ thống: {e}")
        raise

st.success("Hệ thống đã sẵn sàng — upload file để bắt đầu phân tích!")
st.divider()

# ── Upload ───────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([2, 1])
with col_up:
    st.markdown("### Tải lên file âm thanh cần kiểm tra")
    uploaded_file = st.file_uploader(
        "Hỗ trợ: .wav, .mp3, .m4a", type=["wav", "mp3", "m4a"]
    )

with col_info:
    st.markdown("""
    <div class="note-box">
    <b>💡 Hướng dẫn sử dụng</b><br>
    1. Upload file ghi âm (.wav / .mp3)<br>
    2. Bấm <b>Phân tích</b><br>
    3. Xem kết quả tại tab <b>Phát hiện</b><br>
    4. Khám phá lý do AI quyết định tại tab <b>Giải thích XAI</b>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    st.audio(uploaded_file)

    run_xai = st.toggle("Bật phân tích XAI (chậm hơn ~20–40 giây)", value=False)

    if st.button("Bắt đầu Phân tích", use_container_width=True, type="primary"):

        # Ghi file tạm để librosa đọc 1 lần duy nhất
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Load waveform 1 lần — dùng chung cho cả detect lẫn XAI
        with st.spinner("Đang đọc file âm thanh…"):
            y_audio, sr_audio = librosa.load(tmp_path, sr=16000)

        # ── Phát hiện ────────────────────────────────────────────────────
        with st.spinner("Hội đồng AI đang thẩm định…"):
            result = detector.predict_audio(tmp_path)

        os.remove(tmp_path)

        # ── XAI (tuỳ chọn) ───────────────────────────────────────────────
        xai_results = None
        if run_xai and result.get("success"):
            with st.spinner("🔬 Đang tính SHAP — lần đầu có thể mất 15–40 giây…"):
                xai_results = explainer.explain(y_audio, sr_audio)

        # ── Hiển thị ─────────────────────────────────────────────────────
        if not result.get("success"):
            st.error(f"❌ Lỗi: {result.get('error', 'Không xác định')}")
            st.stop()

        prob    = result["confidence_ai"]
        is_fake = result["is_fake"]
        details = result["details"]
        
        # Determine model names based on number of outputs
        if len(details) == 1:
            if mode.startswith("Deep"):
                model_names = ["AASIST (Deep Learning)"]
            else:
                model_names = ["SVM + Wav2Vec2"]
        else:
            model_names = ["LFCC+SVM", "Wav2Vec+MLP", "MFCC+SVM", "XGBoost", "MFCC+MLP"]

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_detect, tab_xai = st.tabs(["Kết quả Phát hiện", "Giải thích XAI"])

        # ==================================================================
        # Tab 1: Detection
        # ==================================================================
        with tab_detect:
            # Tăng nhẹ cột 1 để chữ không bị rớt dòng
            c1, c_space, c2 = st.columns([1.6, 0.2, 1.2])

            with c1:
                if is_fake:
                    st.error("### 🚨 CẢNH BÁO: GIỌNG AI / DEEPFAKE")
                else:
                    st.success("### ✅ AN TOÀN: GIỌNG NGƯỜI THẬT")

                st.metric("Xác suất Deepfake", f"{prob*100:.2f}%",
                          delta=f"{' Trên' if is_fake else ' Dưới'} ngưỡng 50%")

            with c2:
                fig_gauge = plot_confidence_gauge(prob)
                st.pyplot(fig_gauge, use_container_width=False)

            st.divider()
            
            # Show model details based on mode
            if mode.startswith("Deep"):
                st.markdown("#### 🧠 Thông tin mô hình")
                model_info = detector.get_model_info()
                st.markdown(f"""
                <div class="result-card">
                <b>Mô hình:</b> {model_info['name']}<br>
                <b>Mô tả:</b> {model_info['description']}<br>
                <b>Thiết bị:</b> {model_info['device']}<br>
                <b>Phương pháp:</b> Deep Learning (AASIST - Anti-Spoofing with Automatic Speaker Verification)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("#### 📋 Biểu quyết chi tiết của các mô hình")
                cols = st.columns(min(len(details), 5))
                for col, name, val in zip(cols, model_names, details):
                    delta_text = "🔴 Fake" if val >= 0.5 else "🟢 Real"
                    col.metric(label=name, value=f"{val*100:.1f}%", delta=delta_text)

        # ==================================================================
        # Tab 2: XAI
        # ==================================================================
        with tab_xai:
            if xai_results is None:
                st.info("💡 Bật toggle **'Bật phân tích XAI'** trước khi nhấn Phân tích để xem giải thích.")
                st.stop()
            
            # AASIST XAI: Gradient-based saliency
            if mode.startswith("Deep"):
                if xai_results.get("success"):
                    st.markdown("### 🧠 Giải thích đặc trưng — AASIST (Gradient-based Saliency)")
                    st.markdown("""
                    <div class="note-box">
                    ✅ <b>AASIST dùng Gradient-based Saliency</b> — tính độ nhạy cảm của output
                    model deep learning đối với từng sample trong waveform.<br>
                    🔴 Vùng đỏ (cao) = những vùng âm thanh quan trọng với quyết định AI<br>
                    🟢 Vùng xanh (thấp) = những vùng ít ảnh hưởng đến kết quả
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Xác suất AI: {prob*100:.2f}%** — Mức độ: {'Cao 🔴' if prob >= 0.7 else 'Trung bình 🟡' if prob >= 0.4 else 'Thấp 🟢'}")
                    
                    # Plot saliency
                    saliency = xai_results.get("saliency", np.array([]))
                    if len(saliency) > 0:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(12, 4))
                        time_steps = np.arange(len(saliency))
                        ax.fill_between(time_steps, saliency, alpha=0.7, color='#EF4444')
                        ax.plot(time_steps, saliency, color='#DC2626', linewidth=1.5)
                        ax.set_xlabel('Time Step', color='#E2E8F0')
                        ax.set_ylabel('Saliency (Importance)', color='#E2E8F0')
                        ax.set_title('Audio Saliency Map — Vùng đỏ = Quan trọng với kết quả AI', 
                                   color='#E2E8F0', fontsize=12, fontweight='bold')
                        ax.set_facecolor('#1E293B')
                        fig.patch.set_facecolor('#0F172A')
                        ax.grid(True, alpha=0.2, color='#334155')
                        ax.tick_params(colors='#94A3B8')
                        st.pyplot(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Max Saliency", f"{saliency.max():.3f}" if len(saliency) > 0 else "N/A")
                    with col2:
                        st.metric("Mean Saliency", f"{saliency.mean():.3f}" if len(saliency) > 0 else "N/A")
                    with col3:
                        st.metric("Phương pháp", "Gradient-based")
                else:
                    st.warning(f"⚠️ Không thể tính XAI: {xai_results.get('error', 'Unknown error')}")
            
            # Ensemble/Single model XAI: SHAP
            else:
                summary = xai_results.get("_ensemble_summary", {})

                if summary.get("model_weights"):
                    st.markdown("### 🧭 Tổng quan XAI")
                    left_sum, right_sum = st.columns([1, 1])
                    with left_sum:
                        st.markdown("**Nhận xét tự động:**")
                        for note in summary.get("notes", []):
                            st.markdown(f"- {note}")
                    with right_sum:
                        fig_weights = plot_ensemble_weights(summary)
                        st.pyplot(fig_weights, use_container_width=True)

                # ── Ensemble overview — XÁC SUẤT THỰC của từng model ────────────
                st.markdown("### ⚖️ Phiếu bầu / xác suất đầu ra")
                if len(details) > 1:
                    st.caption("Mỗi model độc lập dự đoán xác suất AI — kết quả cuối là trung bình cộng.")
                else:
                    st.caption("Chế độ single-model: hiển thị xác suất từ mô hình Wav2Vec2 + SVM.")

                vote_cols = st.columns(len(details))
                for col, name, val in zip(vote_cols, model_names, details):
                    verdict_label = "🔴 AI" if val >= 0.5 else "🟢 Real"
                    bg     = "#3B1A1A" if val >= 0.5 else "#1A3B1A"
                    border = "#F87171" if val >= 0.5 else "#4ADE80"
                    col.markdown(f"""
                    <div style="background:{bg};border-left:4px solid {border};
                                border-radius:8px;padding:0.7rem 0.8rem;text-align:center;">
                    <div style="font-size:0.8rem;color:#E2E8F0;margin-bottom:4px;font-weight:500;">{name}</div>
                    <div style="font-size:1.5rem;font-weight:700;
                                color:{'#F87171' if val>=0.5 else '#4ADE80'}">
                        {val*100:.1f}%
                    </div>
                    <div style="font-size:0.85rem;margin-top:4px;color:#F8FAFC;font-weight:500;">{verdict_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                avg        = sum(details) / len(details)
                votes_fake = sum(1 for v in details if v >= 0.5)
                denom = len(details)
                vote_phrase = f"{votes_fake}/{denom} model vote AI" if denom > 1 else "Single model output"
                st.markdown(f"""
                <div style="background:#1E293B;border-radius:10px;padding:0.9rem 1.4rem;
                            margin-top:0.8rem;border:1px solid #334155;text-align:center;">
                <code>({' + '.join(f'{v:.2f}' for v in details)}) ÷ {denom} = <b>{avg:.3f}</b></code>
                &nbsp;→&nbsp;
                <b style="color:{'#EF4444' if avg>=0.5 else '#22C55E'}">
                {'🚨 AI (≥ 0.5)' if avg >= 0.5 else '✅ Real (< 0.5)'}
                </b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <b>{vote_phrase}</b>
                </div>
                """, unsafe_allow_html=True)

                st.divider()

                # ── XGBoost SHAP
                xai_model_key = "XGBoost" if "XGBoost" in xai_results else ("Wav2Vec2" if "Wav2Vec2" in xai_results else None)
                if xai_model_key == "XGBoost":
                    st.markdown("### 🌲 Giải thích đặc trưng — XGBoost (TreeSHAP)")
                    st.markdown("""
                    <div class="note-box">
                    ✅ <b>XGBoost dùng TreeSHAP</b> — tính toán đúng đóng góp của từng feature MFCC.<br>
                    🔴 Bar đỏ = feature đẩy về phía AI &nbsp;|&nbsp;
                    🟢 Bar xanh = feature đẩy về phía Real &nbsp;|&nbsp;
                    Dashed line = điểm gốc (base value)
                    </div>
                    """, unsafe_allow_html=True)
                elif xai_model_key == "Wav2Vec2":
                    st.markdown("### 🎧 Giải thích đặc trưng — Wav2Vec2 (KernelSHAP)")
                    st.markdown("""
                    <div class="note-box">
                    ✅ <b>Wav2Vec2 dùng KernelSHAP</b> — các chiều embedding được gom theo cụm 64 chiều.<br>
                    🔴 Bar đỏ = chiều/nhóm đẩy về phía AI &nbsp;|&nbsp;
                    🟢 Bar xanh = chiều/nhóm đẩy về phía Real
                    </div>
                    """, unsafe_allow_html=True)

                if xai_model_key:
                    xgb_res = xai_results.get(xai_model_key)
                    sv_sum   = float(np.sum(xgb_res["shap_values"]))
                    base     = xgb_res["base_value"]
                    logit    = base + sv_sum
                    final_p  = float(1.0 / (1.0 + np.exp(-logit)))
                    color    = "#EF4444" if final_p >= 0.5 else "#22C55E"
                    verdict  = "AI / Deepfake 🚨" if final_p >= 0.5 else "Giọng thật ✅"

                    st.markdown(f"""
                    <div style="background:#1E293B;border-radius:10px;padding:0.7rem 1.2rem;
                                border-left:4px solid {color};margin-bottom:0.8rem;">
                    <code>sigmoid({base:.3f} + {sv_sum:+.3f}) = sigmoid({logit:.3f}) = <b>{final_p:.1%}</b></code>
                    &nbsp;→&nbsp; <b style="color:{color}">{verdict}</b>
                    </div>
                    """, unsafe_allow_html=True)

                    col_wf, col_tbl = st.columns([2, 1])
                    with col_wf:
                        fig_wf = plot_waterfall(xgb_res, xai_model_key, top_k=15)
                        st.pyplot(fig_wf, use_container_width=True)

                    with col_tbl:
                        st.markdown("**Top 10 features:**")
                        for t in xgb_res["top_k"][:10]:
                            bar_color = "🔴" if t["shap_value"] > 0 else "🟢"
                            st.markdown(
                                f"`{t['feature']}`  \n"
                                f"{bar_color} `{t['shap_value']:+.4f}`"
                            )
                else:
                    st.warning("Không tìm thấy kết quả SHAP phù hợp cho mẫu này.")