import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1   = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1   = nn.LayerNorm(channels)   # 채널 방향 정규화
        self.relu    = nn.ReLU()
        self.conv2   = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2   = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(p=dropout)    # CNN branch regularization

    def forward(self, x):
        # x: (B, C, L)
        residual = x

        out = self.conv1(x)
        # LayerNorm expects (B, L, C) → transpose, norm, transpose back
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)

        out = out + residual
        return self.relu(out)


def build_cnn_branch():
    return nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),

        ResBlock1D(32, dropout=0.1),

        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),

        ResBlock1D(64, dropout=0.1),

        nn.AdaptiveAvgPool1d(19),
        nn.Flatten()
        # 출력 차원: 64ch × 19 = 1216  (285/19=15, MPS 호환)
    )
