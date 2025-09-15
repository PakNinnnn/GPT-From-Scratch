import torch
import torch.nn as nn
import torch.nn.functional as F

from expert import Expert

class BasicMoE(nn.Module):
  def __init__(self, feature_in, feature_out, num_experts):
    super().__init__()
    # Vanilla implementation, can try others like attention
    self.gating = nn.Linear(feature_in, num_experts)
    self.experts = nn.ModuleList(
      Expert(
        feature_in, feature_out
      ) for _ in range(num_experts)
    )
  
  def forward(self, x):
    # input -> gating -> experts => aggregate
    experts_weights = self.gating(x) # out = (batch, num_experts)
    experts_out_lst= [expert(x) for expert in self.experts] # out = (batch, feature_out)
    expert_out_lst_resized = [expert_out.unsqueeze(1) for expert_out in experts_out_lst]
    expert_output = torch.concat(expert_out_lst_resized, dim=1) # (batch, num_experts, feature_out)
    
    expert_weights = F.softmax(experts_weights, dim=1) # (batch, num_experts) -> (batch, 1, num_experts)
    experts_weights_resized = experts_weights.unsqueeze(1)
    
    output = experts_weights_resized @ expert_output # (batch, 1, feature_out)
    return output.squeeze(1) # (batch, feature_out)   
    
if __name__ == "__main__":
  x = torch.rand(8, 512)
  moe = BasicMoE(512, 128, 4)
  output = moe(x)
  print(output.shape)