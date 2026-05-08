import streamlit as st
import os
import tempfile
from src.data_processing.ensemble_system import VietGuardEnsemble

# --- CẤU HÌNH TRANG WEB (Load ngay lập tức) ---
st.set_page_config(page_title="Viet-Guard: Anti-Deepfake", page_icon="🛡️", layout="centered")

st.title("🛡️ VIET-GUARD")
st.subheader("Hệ thống phát hiện Giọng nói giả mạo tiếng Việt")
st.markdown("Dự án được xây dựng bởi Nhóm 17. Tích hợp 5 mô hình AI: SVM, XGBoost, MLP và Wav2Vec2.")

# --- HIỆN LOADING KHI BẮT ĐẦU LOAD AI ---
with st.spinner('⏳ Đang khởi động Hội đồng AI (Lần đầu tiên sẽ mất khoảng 15-30 giây)...'):
    @st.cache_resource
    def load_system():
        return VietGuardEnsemble(models_dir='models_saved')

    # Gọi hàm load model
    detector = load_system()

st.success("✅ Hệ thống đã sẵn sàng!")

# --- GIAO DIỆN TẢI FILE ---
st.markdown("### 📤 Tải lên file ghi âm cần kiểm tra (.wav, .mp3)")
uploaded_file = st.file_uploader("", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("🔍 Bắt đầu Phân tích", use_container_width=True):
        with st.spinner('🤖 Hội đồng AI đang hội ý để đánh giá file âm thanh của bạn...'):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            result = detector.predict_audio(tmp_path)
            os.remove(tmp_path)

            if result.get("success", False):
                st.markdown("---")
                prob = result["confidence_ai"] * 100
                
                if result["is_fake"]:
                    st.error(f"🚨 CẢNH BÁO: ĐÂY LÀ GIỌNG AI (DEEPFAKE) 🚨")
                    st.markdown(f"**Độ tự tin của hệ thống: {prob:.2f}%**")
                else:
                    st.success(f"✅ AN TOÀN: ĐÂY LÀ GIỌNG NGƯỜI THẬT ✅")
                    st.markdown(f"**Tỉ lệ nghi ngờ AI: {prob:.2f}%**")
                
                st.markdown("#### 📊 Chi tiết biểu quyết của 5 mô hình:")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                models =["LFCC+SVM", "Wav2Vec+MLP", "MFCC+SVM", "XGBoost", "MFCC+MLP"]
                details = result["details"]
                
                for col, name, val in zip([col1, col2, col3, col4, col5], models, details):
                    col.metric(label=name, value=f"{val*100:.1f}% AI")
                    
            else:
                st.error(f"Đã xảy ra lỗi khi phân tích: {result.get('error', 'Lỗi không xác định')}")