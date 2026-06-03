import torch
import torch.nn as nn
from src.utils.galah.config import FEATURE_DIM

def build_feature_branch():
    return nn.Sequential(
        nn.Linear(in_features=FEATURE_DIM, out_features=128),  # 15 lines x 3 values = 45D
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
