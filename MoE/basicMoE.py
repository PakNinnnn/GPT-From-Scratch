import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.expert import Expert

class BasicMoE(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Vanilla implementation, can try others like attention
    self.gating = nn.Linear(config.h_dim, config.num_expert)
    self.experts = nn.ModuleList(
      Expert(
        config.h_dim, config.h_dim
      ) for _ in range(config.num_expert)
    )
    self.num_expert = config.num_expert
  
  def forward(self, x):
    # input -> gating -> experts => aggregate
    
    batch_size, seq_len, h_dim = x.size()
    
    x = x.view(-1, h_dim)
    
    experts_weights = self.gating(x) # out = (batch * seq_len, config.num_experts)
    experts_out_lst= [expert(x) for expert in self.experts] # out = (batch * seq_len, config.h_dim)
    expert_out_lst_resized = [expert_out.unsqueeze(1) for expert_out in experts_out_lst]
    expert_output = torch.concat(expert_out_lst_resized, dim=1) # (batch * seq_len, config.num_experts, config.h_dim)
    
    expert_weights = F.softmax(experts_weights, dim=1) # (batch * seq_len, config.num_experts) -> (batch * seq_len, 1, config.num_experts)
    experts_weights_resized = experts_weights.unsqueeze(1)
    
    output = experts_weights_resized @ expert_output # (batch * seq_len, 1, config.h_dim)
    output_resized = output.squeeze(1) # (batch * seq_len, config.h_dim)   
    
    return output_resized.view(batch_size, seq_len, h_dim)
    
if __name__ == "__main__":
  class MoEConfig():
    def __init__(self, h_dim, num_expert, top_k, num_shared_expert=2):
      self.h_dim = h_dim
      self.num_expert = num_expert
      self.top_k = top_k
      self.num_shared_expert = num_shared_expert
  
  config = MoEConfig(16, 2, 2)   
  x = torch.rand(2, 4, 16) 
  moe = BasicMoE(config)
  output = moe(x)
  print(output.shape)