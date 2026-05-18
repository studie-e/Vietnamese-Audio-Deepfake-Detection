import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ==========================================
# 1. SINC-CONVOLUTION (Theo SincNet / RawNet2)
# Lớp này học các dải tần số trực tiếp từ raw audio
# ==========================================
class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Đảm bảo kernel_size là số lẻ
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.sample_rate = sample_rate
        
        # Khởi tạo dải tần số theo thang Mel
        mel = np.linspace(self.to_mel(30), self.to_mel(self.sample_rate / 2), self.out_channels + 1)
        hz = self.to_hz(mel)
        
        # Parameters (chỉ học các dải cắt băng tần)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Cửa sổ Hamming
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # Half window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # Trục thời gian (dùng để tính toán hàm sinc)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def to_hz(self, mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def to_mel(self, hz):
        return 2595.0 * np.log10(1 + hz / 700.0)

    def forward(self, waveforms):
        # Đảm bảo tần số luôn dương
        low = self.low_hz_.abs() + 50
        high = low + self.band_hz_.abs()

        band = (high - low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_.to(waveforms.device))
        f_times_t_high = torch.matmul(high, self.n_.to(waveforms.device))

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_.to(waveforms.device) / 2)) * self.window_.to(waveforms.device)
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        # Nối lại để tạo ra bộ lọc hoàn chỉnh
        filters = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        filters = filters / (2 * band[:, None])

        filters = filters.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=1, padding=self.kernel_size // 2)

# ==========================================
# 2. KHỐI RESIDUAL BLOCK (Theo RawNet2)
# ==========================================
class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_block, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Pre-activation SeLU như trong paper
        self.selu = nn.SELU(inplace=True)
        self.maxpool = nn.MaxPool1d(3)

        # Shortcut connection (nếu số channel đổi, cần 1 lớp Conv1x1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.bn1(x)
        out = self.selu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        out = out + residual
        out = self.maxpool(out) # Downsample
        
        return out

class GraphAttentionLayer(nn.Module):
    """
    Mô phỏng cơ chế GAT (Graph Attention) đơn giản:
    Cho phép các node (đặc trưng) tự đánh giá mức độ quan trọng lẫn nhau.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Dùng MultiheadAttention của PyTorch để thay thế cho Self-Attention trên Đồ thị
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(in_dim, out_dim)
        self.selu = nn.SELU()

    def forward(self, x):
        # x: (batch_size, num_nodes, features)
        attn_out, _ = self.attention(x, x, x)
        out = self.selu(self.proj(attn_out))
        return out

class Full_AASIST_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- 1. ENCODER (Giống hệt nãy) ---
        self.sinc_conv = SincConv(out_channels=70, kernel_size=128, sample_rate=16000)
        self.maxpool_1 = nn.MaxPool1d(3)
        self.res_blocks = nn.Sequential(
            Residual_block(in_channels=70, out_channels=32),
            Residual_block(in_channels=32, out_channels=32),
            Residual_block(in_channels=32, out_channels=64),
            Residual_block(in_channels=64, out_channels=64),
            Residual_block(in_channels=64, out_channels=64),
            Residual_block(in_channels=64, out_channels=64),
        )
        self.bn_before_graph = nn.BatchNorm1d(64)
        self.selu = nn.SELU()

        # --- 2. HETEROGENEOUS GRAPH MODULE (Đóng góp của bài báo) ---
        # Ở đây ta chuyển 64 channels thành các node của đồ thị
        self.spectral_attention = GraphAttentionLayer(in_dim=64, out_dim=64)
        self.temporal_attention = GraphAttentionLayer(in_dim=64, out_dim=64)
        
        # Lớp Readout (Gom dữ liệu cuối cùng theo max và avg như Section 3.3)
        self.fc_out = nn.Linear(64 * 2, 2) # *2 vì nối Max và Mean

    def forward(self, x):
        # 1. ENCODER
        x = x.unsqueeze(1)
        x = self.sinc_conv(x)
        x = torch.abs(x) 
        x = self.maxpool_1(x)
        x = self.res_blocks(x) 
        x = self.selu(self.bn_before_graph(x))
        #  x có dạng: (batch, channels=64, time_frames)

        # 2. TẠO ĐỒ THỊ
        # Transpose để chuẩn bị cho Attention: (batch, time_frames, channels)
        x_transposed = x.transpose(1, 2)

        # Mô phỏng G_t (Temporal Graph)
        t_nodes = self.temporal_attention(x_transposed) 
        
        # Mô phỏng G_s (Spectral/Channel Graph)
        s_nodes = self.spectral_attention(x_transposed)

        # 3. MAX GRAPH OPERATION (MGO) - Lấy Max Element-wise giữa 2 nhánh
        combined_nodes = torch.max(t_nodes, s_nodes)

        # 4. READOUT SCHEME
        # Theo bài báo: Dùng node-wise maximum và average
        node_max = torch.max(combined_nodes, dim=1)[0] # Shape: (batch, 64)
        node_avg = torch.mean(combined_nodes, dim=1)   # Shape: (batch, 64)

        # Nối (concatenate) lại
        readout = torch.cat([node_max, node_avg], dim=1) # Shape: (batch, 128)

        # Phân loại 0 (Thật) - 1 (Giả)
        out = self.fc_out(readout)
        return out