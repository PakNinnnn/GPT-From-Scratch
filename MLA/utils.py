
from dataclasses import dataclass

@dataclass
class DeepseekConfig:
  h_dim: int
  num_head: int
  max_position_embeddings: int
  rope_theta: float # frequency
  
  attention_dropout: float
  attention_bias: bool
  
  q_lora_rank: int
  qk_nope_head_dim: int
  kv_lora_rank: int
  
  v_head_dim: int
  qk_rope_head_dim: int