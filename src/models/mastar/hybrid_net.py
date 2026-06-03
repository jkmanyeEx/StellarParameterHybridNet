import torch
import torch.nn as nn

from .cnn_branch import build_cnn_branch
from .dense_branch import build_feature_branch
from .fusion import run_knowledge_fusion
from .output_branch import build_output_branch

class StellarParameterHybridNet(nn.Module):
    def __init__(self):
        super(StellarParameterHybridNet, self).__init__()

        self.cnn_branch = build_cnn_branch()
        self.feature_branch = build_feature_branch()
        self.output_branch = build_output_branch()

    def forward(self, flux, physical_feat):
        x1 = self.cnn_branch(flux)
        x2 = self.feature_branch(physical_feat)
        x_combined = run_knowledge_fusion(x1, x2)
        output = self.output_branch(x_combined)

        return output
