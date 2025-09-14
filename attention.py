import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Single head attention
class SingleHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Compute QKV embeddings
    self.query = nn.Linear(config.h_dim, config.head_size)
    self.key = nn.Linear(config.h_dim, config.head_size)
    self.value = nn.Linear(config.h_dim, config.head_size)
    self.head_size = config.head_size
    
    # Attention mask
    self.register_buffer(
      "attention_mask",
      torch.tril(
        torch.ones(config.block_size, config.block_size)
      )
    )
    
    self.dropout = nn.Dropout(config.dropout)
    
  def forward(self, x):
    batch_size, seq_len, h_dim = x.size()
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
  
    # qk^T
    weight = q @ k.transpose(-2, -1)
    weight = weight.masked_fill(
      # Zero padding 
      self.attention_mask[:seq_len, :seq_len] == 0,
      float('-inf')
    )
    
    weight = F.softmax(weight / math.sqrt(self.head_size), dim=-1) 
    weight = self.dropout(weight)
    
    output = weight @ v
    
    return output
    
# Multi-head attention
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Concate multiple single head attention
    self.heads = nn.ModuleList(
      [
        SingleHeadAttention(config)
        for _ in range(config.n_head)
      ]
    )
    
    self.proj = nn.Linear(config.h_dim, config.h_dim)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, x):
    output = torch.cat(
      [h(x) for h in self.heads],
      dim=-1
    )
    
    output = self.proj(output)
    output = self.dropout(output)
    
    return output