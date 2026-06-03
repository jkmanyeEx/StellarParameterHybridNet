import torch.nn as nn
import torch.optim as optim
from .config import LEARNING_RATE

# NOTE: engine.py constructs the optimizer directly so that it can pass
# weight_decay and later save/restore optimizer state in checkpoints.
# This helper is kept for quick experimentation only — do not use it
# in the main training loop without also updating engine.py.

def build_loss_and_optimizer(model, weight_decay=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=weight_decay)
    return criterion, optimizer
