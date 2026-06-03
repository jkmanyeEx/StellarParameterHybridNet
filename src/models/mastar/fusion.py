import torch
import torch.nn as nn

def run_knowledge_fusion(cnn_output, feature_output):
    combined_tensor = torch.cat((cnn_output, feature_output), dim=1)
    return combined_tensor
