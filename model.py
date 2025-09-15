import torch
import torch.nn as nn
import torch.nn.functional as F

from block import Block

class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    
    # input -> embeddings -> position embedding -> blocks -> FC -> softmax
    
    self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
    self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
    self.blocks = nn.Sequential(
      *[Block(config) for _ in range(config.n_layer)]
    )
    self.ln = nn.LayerNorm(config.h_dim)
    self.lm_head = nn.Linear(config.h_dim, config.vocab_size, bias=False)
    
    # Tie weighting to reduce parameters
    self.token_embedding.weight = self.lm_head.weight
    
    self.config = config
    
    print(f"Using MoE: {self.config.moe}")
    
  def _init_weight(self, module):
    # Initialize to Gaussian distribution
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0, std=0.02)

  def forward(self, idx, targets=None):
    # idx: input token ids
    # tragets: target token ids       
    batch, seq_len = idx.size()
    
    token_embd = self.token_embedding(idx)
    pos_embd = self.position_embedding(
      torch.arange(seq_len, device=idx.device)
    )
    
    x = token_embd + pos_embd
    
    x = self.blocks(x)
    
    x = self.ln(x)
    logits = self.lm_head(x)
    
    if targets is None:
      loss = None
    else:
      batch, seq_len, vocab_size = logits.size()
      logits = logits.view(batch * seq_len, vocab_size)
      targets = targets.view(batch * seq_len)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
  
  def generator(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits, _ = self(idx_cond)
      
      # logits = [batch_size, seq_len, vocab_size]
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      
      # Random sampling
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
      
    return idx