import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


def build_cnn_branch():
    return nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),

        ResBlock1D(32),

        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),

        ResBlock1D(64),

        nn.AdaptiveAvgPool1d(19),
        nn.Flatten()
        # 출력 차원: 64ch × 19 = 1216  (285/19=15, MPS 호환)
    )
