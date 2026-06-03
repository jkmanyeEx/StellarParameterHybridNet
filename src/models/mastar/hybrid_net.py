import torch
import torch.nn as nn

from .cnn_branch import build_cnn_branch
from .dense_branch import build_feature_branch
from .fusion import run_knowledge_fusion
from .output_branch import build_output_branch, CNN_OUT_DIM, DENSE_OUT_DIM

class StellarParameterHybridNet(nn.Module):
    def __init__(self, use_features=True):
        super(StellarParameterHybridNet, self).__init__()

        self.use_features = use_features
        self.cnn_branch   = build_cnn_branch()

        if self.use_features:
            self.feature_branch = build_feature_branch()
            self.output_branch  = build_output_branch(in_dim=CNN_OUT_DIM + DENSE_OUT_DIM)
        else:
            self.feature_branch = None
            self.output_branch  = build_output_branch(in_dim=CNN_OUT_DIM)

    def forward(self, flux, physical_feat=None):
        x1 = self.cnn_branch(flux)

        if self.use_features and physical_feat is not None:
            x2 = self.feature_branch(physical_feat)
            x_combined = run_knowledge_fusion(x1, x2)
        else:
            x_combined = x1

        return self.output_branch(x_combined)
