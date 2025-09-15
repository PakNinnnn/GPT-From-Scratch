import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadAttention
from ffn import FeedForwardNetwork
from MoE.sparseMoE import SparseMoE
from MoE.basicMoE import BasicMoE
from MoE.deepseekMoE import DeepSeekMoE


class Block(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.attn = MultiHeadAttention(config)
    
    if config.moe == "none":
      self.ffn = FeedForwardNetwork(config)
    elif config.moe == "sparse":
      self.ffn = SparseMoE(config)
    elif config.moe == "deepseek":
      self.ffn = DeepSeekMoE(config)
    else:
      self.ffn = BasicMoE(config)
    
    self.ln1 = nn.LayerNorm(config.h_dim)
    self.ln2 = nn.LayerNorm(config.h_dim)
    
    self.config = config
  
  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x_layernorm = self.ln2(x)
    
    if self.config.moe != "none":
      ffn_out, router_logits = self.ffn(x_layernorm)
    else:
      ffn_out = self.ffn(x)
      router_logits = None
        
    x = x + ffn_out
    
    return x, router_logits
  