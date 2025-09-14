import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(config.h_dim, config.h_dim * 4),
      nn.GELU(),
      nn.Linear(config.h_dim * 4, config.h_dim),
      nn.Dropout()
    )
  
  def forward(self, x):
    return self.net(x)