import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import modelConfig
from MoE.deepseekMoE import DeepSeekMoE

def load_balancing_loss(router_logits, num_experts, k):
  # Total loss = auxiliary loss + z loss
  
  router_probs = F.softmax(router_logits, dim=-1)
  
  _, selected_experts = torch.topk(router_probs, k=k, dim=-1)
  
  mask = F.one_hot(selected_experts, num_experts).float()
  
  expected_load = torch.ones_like(router_probs) / num_experts # [1/num_expert, 1/num_expert, ..., 1/num_expert]
  actual_load = mask.mean(dim=0) # (num_expert)
  
  aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts
  
  # Penalize very large router logits
  z_loss = torch.mean(torch.square(router_logits))

  total_loss = aux_loss + z_loss * 0.001
  
  return total_loss

if __name__ == "__main__":
  model = DeepSeekMoE(modelConfig)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  model.train()
  for batch in range(100):
    x = torch.randn(modelConfig.batch_size, modelConfig.seq_len, modelConfig.h_dim)
    target = torch.randn(modelConfig.batch_size, modelConfig.seq_len, modelConfig.h_dim)
    
    output, router_logits = model(x)
    
    mse_loss = F.mse_loss(output, target)
    aux_loss = load_balancing_loss(router_logits, modelConfig.num_expert, modelConfig.top_k)
    total_loss = mse_loss + 0.01 * aux_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if batch % 10 == 0:
      print(f"Batch: {batch}, loss: {total_loss.item():.4f}, MSE: {mse_loss.item():.4f}, aux: {aux_loss.item():.4f}")