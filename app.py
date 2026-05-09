import streamlit as st
import os
import tempfile
import librosa
import numpy as np
from src.data_processing.ensemble_system import VietGuardEnsemble
from src.xai.shap_explainer import AudioXAIExplainer, extract_all_features
from src.xai.visualizer import (
    plot_waterfall,
    plot_ensemble_weights,
    plot_confidence_gauge,
    plot_mfcc_group_importance,
    plot_cross_model_heatmap,
)

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

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    ens = VietGuardEnsemble(models_dir="models_saved")
    xai = AudioXAIExplainer(ens, n_background=10)  # 10 bg samples: nhanh, đủ tốt
    return ens, xai

with st.spinner("⏳ Đang khởi động hệ thống AI…"):
    detector, explainer = load_system()

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
            with st.spinner("🔬 Đang tính SHAP — lần đầu ~15s, lần sau <5s…"):
                feats       = extract_all_features(detector, y_audio, sr_audio)
                xai_results = explainer.explain(feats)

        # ── Hiển thị ─────────────────────────────────────────────────────
        if not result.get("success"):
            st.error(f"❌ Lỗi: {result.get('error', 'Không xác định')}")
            st.stop()

        prob    = result["confidence_ai"]
        is_fake = result["is_fake"]
        details = result["details"]
        model_names = ["LFCC+SVM", "Wav2Vec+MLP", "MFCC+SVM", "XGBoost", "MFCC+MLP"]

        # ── Tabs ─────────────────────────────────────────────────────────
        tab_detect, tab_xai = st.tabs(["Kết quả Phát hiện", "Giải thích XAI (SHAP)"])

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
            st.markdown("#### 📋 Biểu quyết chi tiết của 5 mô hình")
            cols = st.columns(5)
            for col, name, val in zip(cols, model_names, details):
                delta_text = "🔴 Fake" if val >= 0.5 else "🟢 Real"
                col.metric(label=name, value=f"{val*100:.1f}%", delta=delta_text)

        # ==================================================================
        # Tab 2: XAI
        # ==================================================================
        with tab_xai:
            if xai_results is None:
                st.info("💡 Bật toggle **'Bật phân tích XAI'** trước khi nhấn Phân tích để xem giải thích SHAP.")
                st.stop()

            summary = xai_results.get("_ensemble_summary", {})

            # ── Ensemble overview — XÁC SUẤT THỰC của từng model ────────────
            st.markdown("### ⚖️ Phiếu bầu của từng Model")
            st.caption("Mỗi model độc lập dự đoán xác suất AI — kết quả cuối là trung bình cộng.")

            vote_cols = st.columns(5)
            model_names_ordered = ["LFCC+SVM", "Wav2Vec+MLP", "MFCC+SVM", "XGBoost", "MFCC+MLP"]
            for col, name, val in zip(vote_cols, model_names_ordered, details):
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
            st.markdown(f"""
            <div style="background:#1E293B;border-radius:10px;padding:0.9rem 1.4rem;
                        margin-top:0.8rem;border:1px solid #334155;text-align:center;">
            <code>({' + '.join(f'{v:.2f}' for v in details)}) ÷ 5 = <b>{avg:.3f}</b></code>
            &nbsp;→&nbsp;
            <b style="color:{'#EF4444' if avg>=0.5 else '#22C55E'}">
            {'🚨 AI (≥ 0.5)' if avg >= 0.5 else '✅ Real (< 0.5)'}
            </b>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>{votes_fake}/5 model vote AI</b>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # ── XGBoost SHAP — phần duy nhất đáng tin cậy ────────────────────
            st.markdown("### 🌲 Giải thích đặc trưng — XGBoost (TreeSHAP)")
            st.markdown("""
            <div class="note-box">
            ✅ <b>XGBoost dùng TreeSHAP</b> — phương pháp SHAP chính xác nhất, tính toán
            đúng đóng góp của từng feature MFCC vào quyết định của model XGBoost.<br>
            🔴 Bar đỏ = feature đẩy về phía AI &nbsp;|&nbsp;
            🟢 Bar xanh = feature đẩy về phía Real &nbsp;|&nbsp;
            Dashed line = điểm gốc (base value)
            </div>
            """, unsafe_allow_html=True)

            xgb_res = xai_results.get("XGBoost")
            if xgb_res:
                sv_sum   = float(np.sum(xgb_res["shap_values"]))
                base     = xgb_res["base_value"]
                logit    = base + sv_sum          # log-odds (raw SHAP output)
                # Sigmoid để chuyển về probability: khớp với predict_proba
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
                    fig_wf = plot_waterfall(xgb_res, "XGBoost", top_k=15)
                    st.pyplot(fig_wf, use_container_width=True)

                with col_tbl:
                    st.markdown("**Top 10 features ảnh hưởng nhất:**")
                    for t in xgb_res["top_k"][:10]:
                        bar_color = "🔴" if t["shap_value"] > 0 else "🟢"
                        st.markdown(
                            f"`{t['feature']}`  \n"
                            f"{bar_color} `{t['shap_value']:+.4f}`"
                        )
            else:
                st.warning("Không tìm thấy kết quả XGBoost SHAP.")