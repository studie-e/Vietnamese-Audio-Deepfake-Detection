import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Hệ thống Cảnh báo Deepfake", page_icon="🛡️", layout="wide")

# Nạp Mô hình và Thang đo (Dùng cache để không bị load lại mỗi lần bấm nút)
@st.cache_resource
def load_ai_system():
    # 1. Load Mô hình Mạng Nơ-ron (MLP)
    model = joblib.load("models_saved/best_mlp.pkl")
    
    # 2. Khôi phục lại Thang đo (Scaler) từ dữ liệu gốc
    X_data = np.load("data/fetures_model/MLP/X_super_data.npy")
    scaler = StandardScaler()
    scaler.fit(X_data)
    
    return model, scaler

mlp_model, scaler = load_ai_system()

# Hàm trích xuất 56 đặc trưng y hệt như lúc train
def extract_single_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    bandwidth_mean = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    return np.hstack([mfccs_mean, chroma_mean, centroid_mean, bandwidth_mean, rolloff_mean, zcr_mean])

# --- 2. GIAO DIỆN LỊCH SỬ (SIDEBAR) ---
with st.sidebar:
    st.title("🕒 Lịch sử quét")
    st.write("---")
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if len(st.session_state.history) == 0:
        st.info("Chưa có dữ liệu quét.")
    else:
        # Hiện 5 file gần nhất
        for item in reversed(st.session_state.history[-5:]): 
            if item['result'] == 1:
                st.error(f"🚨 {item['name']}")
            else:
                st.success(f"✅ {item['name']}")

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🛡️ Cổng An ninh: Nhận diện Giọng nói Deepfake")
st.markdown("*Hệ thống sử dụng Mạng Nơ-ron Nhân tạo (MLP) để phân tích 56 đặc trưng âm thanh chuyên sâu và phát hiện giả mạo.*")
st.write("---")

# === PHẦN TRÊN: CHIA 2 CỘT (INPUT | KẾT QUẢ) ===
col_input, col_result = st.columns(2, gap="large")

audio_bytes = None
file_name = ""
analyze_button = False

# CỘT TRÁI: NHẬP LIỆU & NÚT BẤM
with col_input:
    tab1, tab2 = st.tabs(["📂 Tải tệp lên", "🎙️ Ghi âm trực tiếp"])

    with tab1:
        uploaded_file = st.file_uploader("Tải tệp (.wav, .mp3)", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue() 
            file_name = uploaded_file.name

    with tab2:
        st.info("💡 Mẹo: Hãy thử đọc một đoạn văn bản bất kỳ vào mic để kiểm tra.")
        recorded_audio = st.audio_input("Nhấn vào biểu tượng Micro để ghi âm")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue() 
            file_name = "Ghi_am_truc_tiep.wav"

    if audio_bytes is not None:
        st.write("🎧 **Nghe lại tệp âm thanh:**")
        st.audio(audio_bytes, format='audio/wav') 
        analyze_button = st.button("🚀 BẮT ĐẦU QUÉT AI", use_container_width=True, type="primary")

# CỘT PHẢI: XỬ LÝ & IN KẾT QUẢ
if audio_bytes is not None and analyze_button:
    
    # 1. Lưu file tạm
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", file_name)
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    with col_result:
        # Hiệu ứng loading chạy ở bên phải
        with st.status("🔍 Hệ thống AI đang tiến hành phân tích...", expanded=True) as status:
            st.write("⏳ Đang bóc tách khoảng lặng và làm sạch nhiễu...")
            time.sleep(1)
            st.write("⏳ Đang trích xuất 56 đặc trưng Super Vector...")
            features = extract_single_audio_features(temp_path)
            time.sleep(1)
            st.write("🧠 Đang đưa qua Mạng Nơ-ron (MLP) đa tầng phân tích...")
            features_scaled = scaler.transform([features])
            prediction = mlp_model.predict(features_scaled)[0]
            probabilities = mlp_model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities) * 100
            time.sleep(1)
            status.update(label="✅ Đã phân tích xong!", state="complete", expanded=False)

        # In kết quả Đỏ/Xanh ở bên phải
        if prediction == 1:
            st.error(f"🚨 CẢNH BÁO GIẢ MẠO (Độ tin cậy: {confidence:.2f}%)")
            st.warning("⚠️ **Hệ thống phát hiện dấu vết nhân tạo (AI Voice Cloning). Tuyệt đối không chuyển tiền.**")
        else:
            st.success(f"✅ AN TOÀN - GIỌNG NGƯỜI THẬT (Độ tin cậy: {confidence:.2f}%)")
            st.info("🟢 Hệ thống không phát hiện sự can thiệp của AI. Tệp âm thanh có độ tự nhiên cao.")

        # Lưu lịch sử quét
        st.session_state.history.append({"name": file_name, "result": prediction})

    # === PHẦN DƯỚI: FULL CHIỀU NGANG VẼ BIỂU ĐỒ ===
    st.write("---")
    st.subheader("📊 Trực quan hóa đặc trưng âm thanh (EDA)")
    
    # Chia lại 2 cột TO ở bên dưới để vẽ biểu đồ
    col_chart1, col_chart2 = st.columns(2)
    y, sr = librosa.load(temp_path, sr=16000)
    
    with col_chart1:
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax1, color="#4A90E2")
        ax1.set_title("Biểu đồ dạng sóng (Waveform)", fontweight="bold")
        st.pyplot(fig1)
    
    with col_chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax2, cmap='magma')
        ax2.set_title("Biểu đồ phổ Mel (Mel-Spectrogram)", fontweight="bold")
        fig2.colorbar(img, ax=ax2, format='%+2.0f dB')
        st.pyplot(fig2)