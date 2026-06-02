import torch.nn as nn
import torch.optim as optim
from .config import LEARNING_RATE

def build_loss_and_optimizer(model):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return criterion, optimizer
