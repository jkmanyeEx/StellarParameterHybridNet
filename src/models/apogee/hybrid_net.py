import torch
import torch.nn as nn

from src.models.apogee.cnn_branch import build_cnn_branch
from src.models.apogee.dense_branch import build_feature_branch
from src.models.apogee.fusion import run_knowledge_fusion
from src.models.apogee.output_branch import build_output_branch

class StellarParameterHybridNet(nn.Module):
    def __init__(self, use_features=True):
        super(StellarParameterHybridNet, self).__init__()
        self.use_features = use_features
        self.cnn_branch = build_cnn_branch()
        if self.use_features:
            self.feature_branch = build_feature_branch()
            self.output_branch = build_output_branch(in_dim=4800 + 128)
        else:
            self.feature_branch = None
            self.output_branch = build_output_branch(in_dim=4800)

    def forward(self, flux, physical_feat=None):
        x1 = self.cnn_branch(flux)
        if self.use_features and physical_feat is not None:
            x2 = self.feature_branch(physical_feat)
            x_combined = run_knowledge_fusion(x1, x2)
            output = self.output_branch(x_combined)
        else:
            output = self.output_branch(x1)

        return output
