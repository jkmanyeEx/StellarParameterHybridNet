import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class MultiArmCNNBranch(nn.Module):
    def __init__(self, num_arms=3, pool_size=25):
        super().__init__()
        self.num_arms = num_arms
        self.base_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            ResBlock1D(32),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            ResBlock1D(64),

            nn.AdaptiveAvgPool1d(pool_size),
            nn.Flatten()
            # 64ch * 25 = 1600D output per arm
        )

    def forward(self, x):
        # x shape: (batch, num_arms, length)
        batch_size, num_arms, length = x.shape
        x_flat = x.view(batch_size * num_arms, 1, length)
        out_flat = self.base_cnn(x_flat)  # (batch * num_arms, 1600)
        out_dim = out_flat.shape[1]
        return out_flat.view(batch_size, num_arms * out_dim)


def build_cnn_branch():
    return MultiArmCNNBranch(num_arms=3, pool_size=25)
