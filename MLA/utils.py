
from dataclasses import dataclass

@dataclass
class DeepseekConfig:
  hidden_size: int
  num_heads: int
  max_seq_len: int
  rope_theta: float # frequency
  
  attention_dropout: float
  attention_bias: bool
  
  q_lora_rank: int
  qk_rope_head_dim: int
  kv_lora_rank: int
  
  v_head_dim: int
  qk_rope_head_dim: int