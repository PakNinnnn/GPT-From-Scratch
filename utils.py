from dataclasses import dataclass

@dataclass
class modelConfig:
  block_size: int = 512
  batch_size: int = 4
  
  n_layer: int = 12
  n_head: int = 12
  n_embd: int = 768 # embedding dimension, size of vector
  h_dim: int = 768
  head_size: int = n_embd // n_head
  
  dropout: float = 0.1

  vocab_size: int = 50257  # GPT-2 vocab size
  
  moe: str = "deepseek" # {"none", "sparse", "vanilla", "deepseek"}
  
  num_expert: int = 16
  top_k: int = 4
  num_shared: int = 2

