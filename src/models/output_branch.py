import torch
import torch.nn as nn

# ── 차원 상수 ──────────────────────────────────────────────────────────────────
# CNN branch:   64ch × 19 (AdaptiveAvgPool1d(19)) = 1216  (285/19=15, MPS 호환)
# Dense branch: 128
# Fusion:       1216 + 128 = 1344
CNN_OUT_DIM   = 1216
DENSE_OUT_DIM = 128
FUSION_DIM    = CNN_OUT_DIM + DENSE_OUT_DIM  # 1344


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


def build_output_branch():
    return nn.Sequential(
        nn.LayerNorm(FUSION_DIM),

        CrossModalAttention(in_dim=FUSION_DIM),

        nn.Linear(in_features=FUSION_DIM, out_features=256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(p=0.2),

        nn.Linear(in_features=256, out_features=3)
    )
