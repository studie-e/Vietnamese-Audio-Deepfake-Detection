import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from models.baseline import Full_AASIST_Model
from pathlib import Path
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

metadata_path = Path(__file__).parent.parent / "vispoofdb" / "data" / "clean_data" / "metadata.csv"

# Load một dataset nhỏ trước
print("[STEP 1] Loading dataset...")
train_dataset = AudioDataset(str(metadata_path), split="train")

print("[STEP 2] Creating DataLoader (batch_size=2)...")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

print("[STEP 3] Get first batch...")
t0 = time.time()
for x, y in train_loader:
    print(f"  Batch loaded! Shape: {x.shape}, Labels: {y}")
    load_time = time.time() - t0
    print(f"  Load time: {load_time:.2f}s")
    
    print("\n[STEP 4] Move batch to device...")
    t0 = time.time()
    x = x.to(device)
    y = y.to(device)
    move_time = time.time() - t0
    print(f"  Move time: {move_time:.2f}s")
    
    print("\n[STEP 5] Initialize model...")
    t0 = time.time()
    model = Full_AASIST_Model().to(device)
    init_time = time.time() - t0
    print(f"  Init time: {init_time:.2f}s")
    
    print("\n[STEP 6] Forward pass...")
    t0 = time.time()
    outputs = model(x)
    forward_time = time.time() - t0
    print(f"  Forward pass time: {forward_time:.2f}s")
    print(f"  Output shape: {outputs.shape}")
    
    print("\n[STEP 7] Backward pass...")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, y)
    t0 = time.time()
    loss.backward()
    backward_time = time.time() - t0
    print(f"  Backward pass time: {backward_time:.2f}s")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Load batch from disk:  {load_time:.2f}s")
    print(f"Move to device:        {move_time:.2f}s")
    print(f"Model init:            {init_time:.2f}s")
    print(f"Forward pass:          {forward_time:.2f}s")
    print(f"Backward pass:         {backward_time:.2f}s")
    print(f"Total:                 {load_time+move_time+init_time+forward_time+backward_time:.2f}s")
    
    break  # Chỉ test 1 batch
