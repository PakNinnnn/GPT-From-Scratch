import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# All implementation based on DeepSeek

class RMSNorm(nn.Module):
  def __init__(self, hidden_size, eps=1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.var_epsilon = eps
  
  def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.var_epsilon)
    
    return self.weight * hidden_states.to(input_dtype)

class DeepseekV2RotaryEmbedding(nn.Module):
  def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
    super().__init__()

    self.dim = dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    inv_freq = 1.0 / (
        self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    # 较小索引位置对应较低频率
    # 较大的索引位置有较高的频率
    
    # Build here to make `torch.jit.trace` work.
    self._set_cos_sin_cache(
        seq_len=max_position_embeddings,
        device=self.inv_freq.device,
        dtype=torch.get_default_dtype(),
    )
    self.max_seq_len_cached = None

  def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    t = torch.arange(
        self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
    )

    freqs = torch.outer(t, self.inv_freq.to(t.device))
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

  def forward(self, x, seq_len=None):
    # x: [bs, num_attention_heads, seq_len, head_size]
    if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
        self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

    return (
        self.cos_cached[:seq_len].to(dtype=x.dtype),
        self.sin_cached[:seq_len].to(dtype=x.dtype),
    )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
  cos = cos[position_ids].unsqueeze(unsqueeze_dim)
  sin = sin[position_ids].unsqueeze(unsqueeze_dim)

  b, h, s, d = q.shape
  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  b, h, s, d = k.shape
  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed