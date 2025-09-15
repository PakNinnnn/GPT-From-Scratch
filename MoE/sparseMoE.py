import torch
import torch.nn as nn
import torch.nn.functional as F

from MoE.expert import Expert

# Reference from Mistral MoE

class MoEConfig():
  def __init__(self, h_dim, num_expert, top_k, num_shared_expert=2):
    self.h_dim = h_dim
    self.num_expert = num_expert
    self.top_k = top_k
    self.num_shared_expert = num_shared_expert

class Router(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.gating = nn.Linear(config.h_dim, config.num_expert)
    self.num_expert = config.num_expert
    self.top_k = config.top_k
  
  def forward(self, x):
    router_logits = self.gating(x) # out: (batch * seq_len, num_expert)
    router_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    router_weights, selected_experts_idx = torch.topk(
      router_probs,
      self.top_k,
      dim=-1
    ) # weight = idx = (batch * seq_len, top_k)
    
    # We need to recalculate weight distribution as router_weights sum != 0
    router_weights = F.softmax(router_weights, dim=1)
    router_weights = router_weights.to(x.dtype)  
    
    expert_mask = F.one_hot(
      selected_experts_idx,
      num_classes=self.num_expert
    ) # output: (batch * seq_len, top_k, num_expert)
    # Why? F.one_hot convert each selected_experts_idx to a one-hot vector with dim num_expert
    # Thus expert_mask means, for each input, we have a (top_k, ) list with each item showing one-hot index of the selected expert
    
    expert_mask = expert_mask.permute(2, 1, 0) # -> (num_expert, top_k, batch * seq_len)
    # Why permute? For later multiplying with expert_output
    
    return router_logits, router_weights, selected_experts_idx, expert_mask

class SparseMoE(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.top_k = config.top_k
    self.num_expert = config.num_expert
    self.num_shared_experts = config.h_dim
    
    self.experts = nn.ModuleList(
      Expert(
        config.h_dim, config.h_dim
      ) for _ in range(config.num_expert)
    )
    
    self.router = Router(config)
  
  def forward(self, x):
    # x: (batch, seq_len, h_dim)
    batch_size, seq_len, h_dim = x.size()
    
    hidden_states = x.view(-1, h_dim)
    
    router_logits, router_weights, selected_experts_idx, expert_mask = self.router(hidden_states)
    
    final_hidden_states = torch.zeros((batch_size * seq_len, h_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    
    # Loop through each expert
    # Put all tokens who selected this expert to `final_hidden_states`
    # e.g. expert 0 may have 100 tokens
    # total number of token = batch * seq_len
    
    for expert_idx in range(self.num_expert):
      expert_layer = self.experts[expert_idx]
      
      # expert_mask: (num_expert, top_k, batch * seq_len)
      current_expert_mask = expert_mask[expert_idx] # (top_k, batch * seq_len)
      
      router_weights_idx, top_x = torch.where(current_expert_mask)
      # router_weights_idx = the top_k position (which "slot" this expert was selected for) => to select expert weight
      # e.g. router_weights_idx = {0, 1} if top_k = 2      
      # top_x = the token index (in batch * seq_len) where this expert was selected => to select hidden_states
      
      current_state = hidden_states.unsqueeze(0) # (1, batch * seq_len, h_dim)
      current_state = current_state[:, top_x, :].reshape(-1, h_dim) # (selected_token_number, h_dim)
      # Why output dim has selected_token_number? top_x = a list of token index routed to this expert
      # And we extract using top_x, so it refers to the total number of token routed to this expert 
      
      current_state = expert_layer(current_state)
      
      current_token_router_weight = router_weights[top_x, router_weights_idx] # (selected_token_number, )
      # Why? router weights is of shape (batch * seq_len, top_k), that is, for each token, the top_k selected expert's weight
      # So router_weights[top_k, router_weights_idx] extract all the selected tokens, for each of their router_weights
      # , we only extract the weight's for the selected experts, so output is a 1-d vector of each token's weight
      
      current_token_router_weight = current_token_router_weight.unsqueeze(-1)
      
      current_hidden_states = current_state * current_token_router_weight # (selected_token_num, h_dim) * (selected_token_num, 1)
      
      # Difference between index_add and index_add_??
      final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
      
    final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, h_dim)
    
    return final_hidden_states, router_logits # for computing loss
    
if __name__ == "__main__":
  x = torch.rand(2, 4, 16) 
  config = MoEConfig(16, 2, 2)
  sparseMoE = SparseMoE(config)
  output = sparseMoE(x)
  print(output[0].shape, output[1].shape)