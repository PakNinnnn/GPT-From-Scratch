import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.attn = MultiHeadAttention(config)
    self.ffn = FeedForwardNetwork(config)
    
    self.ln1 = nn.LayerNorm(config.h_dim)
    self.ln2 = nn.LayerNorm(config.h_dim)
  
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    
    return x
  