import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from models.baseline import Full_AASIST_Model
from pathlib import Path
import time
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# DATA PATH
# =========================
metadata_path = Path(__file__).parent.parent / "vispoofdb" / "data" / "clean_data" / "metadata.csv"

if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata không tìm thấy: {metadata_path}")

print(f"Using metadata: {metadata_path}")

# =========================
# DATASET
# =========================
print("[1/6] Loading train dataset...", end=" ", flush=True)
t0 = time.time()
train_dataset = AudioDataset(str(metadata_path), split="train")
print(f"✓ ({time.time()-t0:.1f}s)")

print("[2/6] Loading test_seen dataset...", end=" ", flush=True)
t0 = time.time()
test_seen_dataset = AudioDataset(str(metadata_path), split="test_seen")
print(f"✓ ({time.time()-t0:.1f}s)")

print("[3/6] Loading test_unseen dataset...", end=" ", flush=True)
t0 = time.time()
test_unseen_dataset = AudioDataset(str(metadata_path), split="test_unseen")
print(f"✓ ({time.time()-t0:.1f}s)")

print("[4/6] Creating DataLoaders...", end=" ", flush=True)
t0 = time.time()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_seen_loader = DataLoader(test_seen_dataset, batch_size=16, shuffle=False)
test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=16, shuffle=False)
print(f"✓ ({time.time()-t0:.1f}s)")

print("[5/6] Initializing model...", end=" ", flush=True)
t0 = time.time()
model = Full_AASIST_Model().to(device)
print(f"✓ ({time.time()-t0:.1f}s)")

print("[6/6] Setting up optimizer...", end=" ", flush=True)
t0 = time.time()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
print(f"✓ ({time.time()-t0:.1f}s)")

print("\n" + "="*60)
print("TRAINING WITH 2 EPOCHS (DEBUG MODE)")
print("="*60)

EPOCHS = 2
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n[Epoch {epoch+1}/{EPOCHS}]")
    model.train()
    total_loss = 0
    batch_count = 0

    print("  Training:", end=" ", flush=True)
    t_epoch = time.time()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        
        if (i + 1) % 100 == 0:
            print(f"{i+1}", end=" ", flush=True)

    train_loss = total_loss / len(train_loader)
    epoch_time = time.time() - t_epoch
    print(f" ✓ ({epoch_time:.1f}s, Loss: {train_loss:.4f})")

    # --- VALIDATION (test_seen) ---
    print("  Validating:", end=" ", flush=True)
    t_val = time.time()
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(test_seen_loader):
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            
            if (i + 1) % 50 == 0:
                print(f"{i+1}", end=" ", flush=True)

    val_acc = accuracy_score(all_labels, all_preds)
    val_time = time.time() - t_val
    print(f" ✓ ({val_time:.1f}s, Acc: {val_acc:.4f})")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "aasist_best_model.pth")
        print("  => Kỷ lục mới! Đã lưu model.")

print("\n" + "="*60)
print("DEBUG TRAINING HOÀN TẤT!")
print("="*60)
print(f"Best Test-Seen Accuracy: {best_val_acc:.4f}")
print("\nNếu chạy được tới đây, có thể:")
print("- Train.py full (20 epoch) sẽ mất nhiều giờ trên CPU")
print("- Có thể chạy bằng GPU để nhanh hơn")
