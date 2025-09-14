from model import GPT
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import modelConfig
import os
import tiktoken


def load_model(checkpoint_path, device):
  config = modelConfig()
  model = GPT(config).to(device)
  
  if os.path.isfile(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully")
  else:
    print(f"No checkpoint found at {checkpoint_path}. Initializing model with random weights.")
  
  model.eval()
  return model


def generate_text(model, text, max_new_tokens, device):
  # Tokenize the input text
  tokenizer = tiktoken.get_encoding("gpt2")
  input_ids = tokenizer.encode(text)
  idx = torch.tensor([input_ids], dtype=torch.long).to(device)

  with torch.no_grad():
    for _ in range(max_new_tokens):
      # Get the model's predictions
      logits, _ = model(idx)
      
      # Focus on the last time step
      logits = logits[:, -1, :]  # Shape: (1, vocab_size)
      
      # Apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1)  # Shape: (1, vocab_size)
      
      # Sample from the distribution or take the most likely token
      next_token_id = torch.multinomial(probs, num_samples=1)  # Shape: (1, 1)
      
      # Append the predicted token to the input sequence
      idx = torch.cat((idx, next_token_id), dim=1)  # Shape: (1, sequence_length + 1)
  
  # Decode
  generated_text = tokenizer.decode(idx[0].tolist())

  return generated_text

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = load_model("checkpoint.pth", device)
  
  # Example input text
  input_text = "Once upon a time"
  
  # Generate text
  generated_text = generate_text(model, input_text, max_new_tokens=50, device=device)

  # Decode the generated token IDs back to text
  print("Generated Text:")
  print(generated_text)