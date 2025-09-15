import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm

from model import GPT
from utils import modelConfig
from dataset import MyDataset

from datasets import load_dataset

ds = load_dataset("rojagtap/bookcorpus")

def train(model, optimizer, scheduler, train_loader, epoch, device):
  model.train() 
  
  total_loss = 0
  for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
    x, y = x.to(device), y.to(device)
    
    logits, loss = model(x, targets=y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    total_loss += loss.item()
    
    if batch_idx % 100 == 0:
      print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
  return total_loss / len(train_loader)

def eval(model, val_loader, device):
  model.eval()
  
  total_val_loss = 0
  
  with torch.no_grad():
    for x, y in tqdm(val_loader, desc="Validation"):
      x, y = x.to(device), y.to(device)
      
      logits, val_loss = model(x, targets=y)
      total_val_loss += val_loss.item()
  
  print(f'Validation Loss: {total_val_loss / len(val_loader):.4f}')
  return total_val_loss / len(val_loader)

    
if __name__ == "__main__":
  train_dataset = MyDataset(ds, max_samples=1000)
  # Each sample: [12, 512]
  
  train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
  train_loader = DataLoader(train_dataset, batch_size=modelConfig.batch_size, shuffle=True)    
  val_loader = DataLoader(val_dataset, batch_size=modelConfig.batch_size, shuffle=False)

  # Print # of training and validation samples
  print(f"Training samples: {len(train_dataset)}")
  print(f"Validation samples: {len(val_dataset)}")
  
  model = GPT(modelConfig)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = model.to(device)
  
  print(f"Using device: {device}")

  total_param = sum(p.numel() for p in model.parameters())
  print(f"Total parameters: {total_param / 1e6} M")

  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
  total_epochs = 1
  
  for epoch in range(total_epochs):
    print(f"Epoch {epoch} ----------------")
    train_loss = train(model, optimizer, scheduler, train_loader, epoch, device)
    val_loss = eval(model, val_loader, device)
    
    
    checkpoint = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "scheduler_state_dict": scheduler.state_dict(),
      "val_loss": val_loss
    }

    torch.save(checkpoint, f"checkpoints/model_epoch{epoch}.pt")
    print(f"Saved model checkpoint at checkpoints/model_epoch{epoch}.pt")
    