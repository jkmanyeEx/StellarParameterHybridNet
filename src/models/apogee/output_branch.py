import torch
import torch.nn as nn
from src.utils.apogee.config import NUM_ARMS, OUTPUT_DIM_PER_ARM

# ── 차원 상수 ──────────────────────────────────────────────────────────────────
CNN_OUT_DIM   = NUM_ARMS * OUTPUT_DIM_PER_ARM  # 3 * 1600 = 4800
DENSE_OUT_DIM = 128
FUSION_DIM    = CNN_OUT_DIM + DENSE_OUT_DIM   # 4928


class CrossModalAttention(nn.Module):
    def __init__(self, in_dim=FUSION_DIM):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention_weights(x)


def build_output_branch(in_dim=FUSION_DIM):
    return nn.Sequential(
        nn.LayerNorm(in_dim),

        CrossModalAttention(in_dim=in_dim),

        nn.Linear(in_features=in_dim, out_features=256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(p=0.2),

        nn.Linear(in_features=256, out_features=3)
    )
