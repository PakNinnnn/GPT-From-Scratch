import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tiktoken


class MyDataset(Dataset):
  def __init__(self, ds, block_size=512):
    self.enc = tiktoken.get_encoding("gpt2")
    self.block_size = block_size
    self.max_lines = 500000    
    self.encoded_data = []
    
    # Specify the pecial symbol that used to separate samples
    self.eos_token = self.enc.encode(
      "<|endoftext|>",
      allowed_special={"<|endoftext|>"}
    )[0]
    
    import json
    
    raw_text = []
    # ds should be a HuggingFace dataset dict, e.g. ds['train']
    split = ds['train'] if 'train' in ds else ds
    for i, sample in enumerate(split):
      if i >= self.max_lines:
        break
      # For BookCorpus, text is under "text" key
      text = sample.get("text", "").strip()
      if text:
        raw_text.append(text)
    
    # Concate ALL samples with eos token to separate
    encoded = []
    for text in raw_text:
      encoded_text = self.enc.encode(text)
      encoded.extend(encoded_text + [self.eos_token])
    
    # Trim to block_size
    for i in range(0, len(encoded), self.block_size):
      chunk = encoded[i:i+self.block_size+1] # Each 513 tokens
      if len(chunk) < self.block_size + 1:
        chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
      
      self.encoded_data.append(chunk)       
    
  def __len__(self):
    return len(self.encoded_data)

  def __getitem__(self, idx):
    chunk = self.encoded_data[idx]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    
    return x, y
  
  def encode(self, text):
    return self.enc.encode(text)

  def decode(self, ids):
    return self.enc.decode(ids)


if __name__ == "__main__":
  from datasets import load_dataset

  ds = load_dataset("rojagtap/bookcorpus")
  dataset = MyDataset(ds)
  
  # print total samples
  print(f"Total samples: {len(dataset)}")