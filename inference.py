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
  
  tokenizer = tiktoken.get_encoding("gpt2")
  input_ids = tokenizer.encode(text)
  idx = torch.tensor([input_ids], dtype=torch.long).to(device)

  output_idx = model.generator(idx, max_new_tokens)

  generated_text = tokenizer.decode(output_idx[0].tolist())

  return generated_text

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = load_model("checkpoints/model_epoch0.pt", device)
  
  # Example input text
  input_text = "Once upon a time"
  
  # Generate text
  generated_text = generate_text(model, input_text, max_new_tokens=50, device=device)

  # Decode the generated token IDs back to text
  print("Generated Text:")
  print(generated_text)