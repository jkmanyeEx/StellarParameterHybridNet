import torch
import torch.nn as nn

def build_feature_branch():
    return nn.Sequential(
        nn.Linear(in_features=30, out_features=128),  # 10선 × 3값 = 30D
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(p=0.1),

        nn.Linear(in_features=128, out_features=128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Dropout(p=0.1),

        nn.Linear(in_features=128, out_features=128),
        nn.LayerNorm(128),
        nn.GELU()
    )
