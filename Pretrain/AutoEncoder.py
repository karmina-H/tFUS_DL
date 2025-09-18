import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- CBAM for 3D -----
class ChannelAttention3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    # kernel_size=3 로 설정해 작은 특징맵(예: 6×6×6)에서도 안전
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn


class CBAM3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 3):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ----- 3D VAE with CBAM -----
class VAE3DWithAttention(nn.Module):
    """
    입력:  (N, 1, 101, 101, 101)
    잠재:  (N, z_channels, 6, 6, 6)  # conv-잠재 (벡터가 아닌 spatial latent)
    출력:  (N, 1, 101, 101, 101)
    """
    def __init__(self, z_channels: int = 64):
        super().__init__()

        # Encoder (원본 구조 유지)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),     # -> (N, 32, 101,101,101)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                     # -> (N, 32, 50,50,50)

            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),    # -> (N, 64, 50,50,50)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                     # -> (N, 64, 25,25,25)

            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),   # -> (N, 128, 25,25,25)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                     # -> (N, 128, 12,12,12)

            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),  # -> (N, 256, 12,12,12)
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),                     # -> (N, 256, 6,6,6)
        )

        # ----- "중간" Attention: bottleneck에서 CBAM -----
        self.attn = CBAM3D(256, reduction=16, spatial_kernel=3)

        # ----- VAE 파트: μ, logσ² 산출 및 z 샘플링 -----
        # 1x1x1 conv로 채널만 줄여 conv-잠재로 사용
        self.to_mu     = nn.Conv3d(256, z_channels, kernel_size=1)
        self.to_logvar = nn.Conv3d(256, z_channels, kernel_size=1)
        self.from_z    = nn.Conv3d(z_channels, 256, kernel_size=1)

        # Decoder (원본 구조 유지)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),  # 6 -> 12
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),   # 12 -> 24
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),    # 24 -> 48
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),    # 48 -> 96
            nn.ReLU(inplace=True),

            # (I - 1)*stride - 2*padding + kernel_size + output_padding
            # (96 - 1)*1 - 0 + 6 + 0 = 101
            nn.ConvTranspose3d(16, 1, kernel_size=6, stride=1, padding=0),  # 96 -> 101
            # 주의: BCEWithLogitsLoss를 쓸 예정이라 Sigmoid는 넣지 않음
        )

    # ----- VAE 유틸 -----
    def encode(self, x):
        h = self.encoder(x)      # (N,256,6,6,6)
        h = self.attn(h)         # attention in the middle
        mu = self.to_mu(h)       # (N,z,6,6,6)
        logvar = self.to_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.from_z(z)       # (N,256,6,6,6)
        x_logits = self.decoder(h)
        return x_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        # 반환은 (복원 logits, μ, logσ²)
        return x_logits, mu, logvar


# ----- 손실 함수 예시 -----
def vae_loss(x_logits, x_target, mu, logvar, beta: float = 1.0, use_bce: bool = True):
    """
    x_logits: 디코더 출력 (logits)
    x_target: 원본 입력과 동일 크기 (N,1,101,101,101)
    - 재구성 손실: BCEWithLogits(기본) 또는 MSE(옵션)
    - KL divergence: N, C, D, H, W 차원 합->배치 평균
    """
    if use_bce:
        # 안정적 학습을 위해 sum 후 batch 평균 권장 (스케일이 KL과 비슷해짐)
        recon = F.binary_cross_entropy_with_logits(
            x_logits, x_target, reduction='sum'
        ) / x_target.size(0)
    else:
        recon = F.mse_loss(torch.sigmoid(x_logits), x_target, reduction='mean')

    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                                     dim=[1, 2, 3, 4]))
    return recon + beta * kl, recon, kl
