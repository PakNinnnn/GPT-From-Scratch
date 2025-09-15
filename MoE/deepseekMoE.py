import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.sparseMoE import SparseMoE, Expert
from utils import modelConfig

# DeepSeek MoE implementation
# SparseMoE + shared experts

class DeepSeekMoE(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.routed_expert_moe = SparseMoE(config)
    self.shared_expert = nn.ModuleList(
      [
        Expert(config.h_dim, config.h_dim) for _ in range(config.num_shared)
      ] 
    )  
  
  def forward(self, x):
    batch_size, seq_len, h_dim = x.size()
    
    # Shared experts
    shared_expert_output_lst = [expert(x) for expert in self.shared_expert] # (num_shared, batch_size, seq_len, h_dim)
    shared_expert_output = torch.stack(shared_expert_output_lst, dim=0)
    shared_expert_output = shared_expert_output.sum(dim=0) # (batch_size, seq_len, h_dim)
    
    # Routed experts
    sparse_moe_out, router_logits = self.routed_expert_moe(x)
    
    output = shared_expert_output + sparse_moe_out # Elementwise addition (no weighting)
    
    return output, router_logits
    
    
if __name__ == "__main__":
  x = torch.rand(2, 4, 768)  
  deepseekMoE = DeepSeekMoE(modelConfig)
  output = deepseekMoE(x)
  print(output[0].shape)
  