import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
  def __init__(self, feature_in, feature_out):
    super().__init__()
    self.ln = nn.Linear(feature_in, feature_out)

  # Todo: add activation function  
  def forward(self, x):
    return self.ln(x)