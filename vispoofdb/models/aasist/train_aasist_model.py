"""
train_aasist_model.py
=====================
Huấn luyện AASIST trực tiếp từ vispoofdb/models/aasist/.

Cách chạy (từ thư mục gốc dự án):
    python vispoofdb/models/aasist/train_aasist_model.py

Hoặc qua wrapper:
    python vispoofdb/models/train_aasist.py

Kết quả lưu tại:
    vispoofdb/models_saved/aasist_best_model.pth
"""

import sys
import os
from pathlib import Path

# Thêm thư mục hiện tại vào sys.path để import dataset và models
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from models.baseline import Full_AASIST_Model
from sklearn.metrics import accuracy_score

# Fix encoding Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = THIS_DIR.parents[2]          # gốc dự án
METADATA_PATH  = BASE_DIR / "vispoofdb" / "data" / "clean_data" / "metadata.csv"
SAVE_MODEL_DIR = BASE_DIR / "vispoofdb" / "models_saved"
SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = SAVE_MODEL_DIR / "aasist_best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra tiên quyết
# ─────────────────────────────────────────────────────────────────────────────
if not METADATA_PATH.exists():
    print(f"[ERROR] Metadata không tìm thấy: {METADATA_PATH}")
    print("        Hãy chạy vispoofdb_generate_metadata.py trước!")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 65)
print("  AASIST — Anti-Spoofing with Integrated Spectral & Temporal")
print(f"  Device: {device.upper()}")
print(f"  Metadata: {METADATA_PATH}")
print(f"  Save to:  {MODEL_SAVE_PATH}")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading datasets...")
train_dataset        = AudioDataset(str(METADATA_PATH), split="train")
test_seen_dataset    = AudioDataset(str(METADATA_PATH), split="test_seen")
test_unseen_dataset  = AudioDataset(str(METADATA_PATH), split="test_unseen")

train_loader       = DataLoader(train_dataset,       batch_size=16, shuffle=True,  num_workers=0)
test_seen_loader   = DataLoader(test_seen_dataset,   batch_size=16, shuffle=False, num_workers=0)
test_unseen_loader = DataLoader(test_unseen_dataset, batch_size=16, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────────────────────────────────────
# Model, Loss, Optimizer
# ─────────────────────────────────────────────────────────────────────────────
model     = Full_AASIST_Model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS       = 20
best_val_acc = 0.0

print(f"\nBắt đầu training {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss    = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # --- Validate (test_seen) ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_seen_loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss:.4f} | Test-Seen Acc: {val_acc:.4f}", end="")

    # Lưu nếu phá kỷ lục
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
        print(f"  => Ky luc moi! Da luu model.")
    else:
        print()

print(f"\nTraining hoan tat! Best Test-Seen Acc: {best_val_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Đánh giá cuối: test_unseen
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  DANH GIA TREN TEST_UNSEEN (giong AI hoan toan moi)")
print("=" * 65)

# Load lại model tốt nhất để đánh giá
model.load_state_dict(torch.load(str(MODEL_SAVE_PATH), map_location=device))
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_unseen_loader:
        x = x.to(device)
        preds = torch.argmax(model(x), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

test_unseen_acc = accuracy_score(all_labels, all_preds)

print(f"\n  [TEST_UNSEEN]  AASIST")
print(f"  Accuracy: {test_unseen_acc * 100:.2f}%")
print(f"\n  Model da luu tai: {MODEL_SAVE_PATH}")
print("=" * 65)


if __name__ == "__main__":
    pass  # Logic chạy ở module level
