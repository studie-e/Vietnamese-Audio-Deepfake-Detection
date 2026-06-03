import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
real_audio, sr = librosa.load('data/clean_data/real/1.wav', sr=16000)
fake_audio, _ = librosa.load('data/clean_data/ai/1.wav', sr=16000)

# ==========================================
# FIX 1: Cắt bỏ khoảng lặng thừa ở đầu và cuối để biểu đồ gọn gàng
# ==========================================
real_audio, _ = librosa.effects.trim(real_audio)
fake_audio, _ = librosa.effects.trim(fake_audio)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

# ==========================================
# HÀNG 1: BIỂU ĐỒ MFCC 
# ==========================================
# Dùng chuẩn hóa (Standardization) để màu sắc MFCC nổi bật hơn
mfcc_real = librosa.feature.mfcc(y=real_audio, sr=sr, n_mfcc=20)
mfcc_fake = librosa.feature.mfcc(y=fake_audio, sr=sr, n_mfcc=20)

librosa.display.specshow(mfcc_real, x_axis='time', ax=ax[0, 0], cmap='viridis') # Đổi màu viridis cho dễ nhìn
ax[0, 0].set(title='MFCC - Giọng Thật (Đã cắt khoảng lặng)')

librosa.display.specshow(mfcc_fake, x_axis='time', ax=ax[0, 1], cmap='viridis')
ax[0, 1].set(title='MFCC - Giọng Giả (Đã cắt khoảng lặng)')

# ==========================================
# HÀNG 2: BIỂU ĐỒ CAO ĐỘ (PITCH / F0)
# ==========================================
f0_real, voiced_flag_real, _ = librosa.pyin(real_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0_fake, voiced_flag_fake, _ = librosa.pyin(fake_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

times_real = librosa.times_like(f0_real)
times_fake = librosa.times_like(f0_fake)

# Dùng marker 'o' để làm nổi bật các điểm cao độ bị đứt đoạn
ax[1, 0].plot(times_real, f0_real, label='F0', color='green', linewidth=2, marker='.', markersize=4)
ax[1, 0].set(title='Đường viền Cao độ (Pitch) - Giọng Thật', xlabel='Time', ylabel='Hz')
ax[1, 0].set_ylim(50, 500) # FIX 2: Nâng trần lên 500Hz để không bị cụt đầu

ax[1, 1].plot(times_fake, f0_fake, label='F0', color='orange', linewidth=2, marker='.', markersize=4)
ax[1, 1].set(title='Đường viền Cao độ (Pitch) - Giọng Giả', xlabel='Time', ylabel='Hz')
ax[1, 1].set_ylim(50, 500) # FIX 2: Nâng trần lên 500Hz

plt.tight_layout()
plt.savefig('figures/Bieu_do_cao_do_F0.png', dpi=300)
plt.show()