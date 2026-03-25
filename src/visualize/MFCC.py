import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load 2 file âm thanh
real_audio, sr = librosa.load('data/clean_data/real/1.wav', sr=16000)
fake_audio, _ = librosa.load('data/clean_data/ai/1.wav', sr=16000)

# 2. Tạo khung vẽ (Figure) có 2 hàng, 2 cột
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

# --- HÀNG 1: BIỂU ĐỒ DẠNG SÓNG (WAVEFORM) ---
librosa.display.waveshow(real_audio, sr=sr, ax=ax[0, 0], color='blue')
ax[0, 0].set(title='Waveform - Giọng Thật (Người)')

librosa.display.waveshow(fake_audio, sr=sr, ax=ax[0, 1], color='red')
ax[0, 1].set(title='Waveform - Giọng Giả (AI Clone)')

# --- HÀNG 2: BIỂU ĐỒ ẢNH PHỔ (SPECTROGRAM) ---
# Chuyển đổi âm thanh sang phổ tần số (Mel-Spectrogram)
S_real = librosa.feature.melspectrogram(y=real_audio, sr=sr)
S_fake = librosa.feature.melspectrogram(y=fake_audio, sr=sr)
S_real_db = librosa.power_to_db(S_real, ref=np.max)
S_fake_db = librosa.power_to_db(S_fake, ref=np.max)

# Vẽ Spectrogram
img1 = librosa.display.specshow(S_real_db, x_axis='time', y_axis='mel', sr=sr, ax=ax[1, 0])
ax[1, 0].set(title='Mel-Spectrogram - Giọng Thật')

img2 = librosa.display.specshow(S_fake_db, x_axis='time', y_axis='mel', sr=sr, ax=ax[1, 1])
ax[1, 1].set(title='Mel-Spectrogram - Giọng Giả')

# Hiển thị và lưu ảnh
plt.tight_layout()
plt.savefig('figures/so_sanh_that_gia_MFCC.png', dpi=300) # Lưu ảnh độ nét cao để chèn Word
plt.show()