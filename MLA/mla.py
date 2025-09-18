import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from misc import *
from utils import DeepseekConfig

class MLA(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    self.attn_dropout = config.attention_dropout
    self.h_dim = config.h_dim
    self.num_head = config.num_head
    self.v_head_dim = config.v_head_dim
    
    self.out_project = nn.Linear(self.num_head * self.v_head_dim, self.h_dim, bias=False)
    
    # Downward projection
    self.q_lora_rank = config.q_lora_rank # 7168 -> 1536, 1:4.7
    self.kv_lora_rank = config.kv_lora_rank 
    
    self.q_down_project = nn.Linear(self.h_dim, self.q_lora_rank, bias=config.attention_bias)
    self.kv_down_project = nn.Linear(self.h_dim, self.kv_lora_rank + config.qk_rope_head_dim, bias=config.attention_bias)
    
    self.q_down_norm = RMSNorm(self.q_lora_rank)
    self.kv_down_norm = RMSNorm(self.kv_lora_rank)
    
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_rope_head_dim = config.qk_rope_head_dim
    
    # Upward projection
    self.q_head_dim = self.qk_nope_head_dim + config.qk_rope_head_dim # k_head_dim = q_head_dim
    self.q_up_project = nn.Linear(self.q_lora_rank, self.num_head * self.q_head_dim, bias=config.attention_bias)
    # Need split rope and nope
    
    self.kv_up_project = nn.Linear(self.kv_lora_rank, self.num_head * (self.q_head_dim - config.qk_rope_head_dim + self.v_head_dim), bias=config.attention_bias)
    
    # ROPE
    self.rotary_emb = DeepseekV2RotaryEmbedding(
      config.qk_rope_head_dim,
      config.max_position_embeddings,
      config.rope_theta
    )
  
  def forward(self, hidden_states, position_idx, attention_mask=None):
    batch_size, seq_len, hidden_dim = hidden_states.size()
    
    q = self.q_down_project(hidden_states)
    q = self.q_down_norm(q)
    q = self.q_up_project(q)
    q = q.view(batch_size, seq_len, self.num_head, self.q_head_dim).transpose(1, 2)
    # q : (batch_size, num_head, seq_len, head_dim)
    
    q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    
    kv = self.kv_down_project(hidden_states)
    kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    # k_rope: (batch_size, 1, seq_len, head_dim)
    kv = self.kv_down_norm(kv)
    kv = self.kv_up_project(kv)
    kv = kv.view(
          batch_size, 
          seq_len, 
          self.num_head, 
          self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
    k_nope, v_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    
    kv_seq_len = v_states.shape[-2]
    cos, sin = self.rotary_emb(v_states, seq_len=kv_seq_len)
    q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin, position_idx)    
    
    q_states = torch.concat([q_nope, q_rope], dim=-1)
    k_states = torch.concat([k_nope, k_rope.expand(-1, self.num_head, -1, -1)], dim=-1) # shape: (batch_size, num_head, q_len, head_dim)
    
    attn_weights = q_states @ k_states.transpose(2, 3)
    attn_weights = attn_weights / math.sqrt(self.q_head_dim)
    
    if attention_mask is not None:
      # Assume causal mask
      attn_weights = torch.masked_fill(
        attn_weights, attention_mask==0, float('-inf')
      )
    
    attn_weights = F.softmax(attn_weights, dim=-1).to(q_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
    
    output = attn_weights @ v_states  # shape: (batch_size, num_head, seq_len, v_head_dim)
    output = output.transpose(1, 2).reshape(batch_size, seq_len, -1) # (batch_size, seq_len. num_head * v_head_dim)
    output = self.out_project(output)
    
    return output, attn_weights
  
  
if __name__ == "__main__":
  config = DeepseekConfig(
      h_dim=7168,
      num_head=16,
      max_position_embeddings=1024,
      rope_theta=128000,
      
      attention_dropout=0.1,
      attention_bias=False,
      
      q_lora_rank=1536,
      qk_rope_head_dim=64,
      kv_lora_rank=512,
      
      v_head_dim=128,
      qk_nope_head_dim=128,
  )
  
  mla = MLA(config)
  x = torch.randn(2, 1024, 7168)
  position_idx = torch.arange(
      config.max_position_embeddings,
  ).unsqueeze(0).expand(
      x.size(0), -1
  ) # (batch_size, seq_len)
  attn_output, attn_weights = mla(x, position_idx=position_idx)
  print(attn_output.shape)
  print(attn_weights.shape)
  