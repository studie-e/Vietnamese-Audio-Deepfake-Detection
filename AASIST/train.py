import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from models.baseline import Full_AASIST_Model
from pathlib import Path

from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# DATA PATH
# =========================
# Đường dẫn từ AASIST/train.py lên 2 level, rồi vào vispoofdb/data/clean_data/
metadata_path = Path(__file__).parent.parent / "vispoofdb" / "data" / "clean_data" / "metadata.csv"

if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata không tìm thấy: {metadata_path}")

print(f"Using metadata: {metadata_path}")

# =========================
# DATASET
# =========================
train_dataset = AudioDataset(str(metadata_path), split="train")
test_seen_dataset = AudioDataset(str(metadata_path), split="test_seen")
test_unseen_dataset = AudioDataset(str(metadata_path), split="test_unseen")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_seen_loader = DataLoader(
    test_seen_dataset,
    batch_size=16,
    shuffle=False
)

test_unseen_loader = DataLoader(
    test_unseen_dataset,
    batch_size=16,
    shuffle=False
)

# =========================
# MODEL
# =========================

model = Full_AASIST_Model().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# =========================
# TRAIN LOOP THÔNG MINH (LƯU BEST MODEL)
# =========================
EPOCHS = 20
best_val_acc = 0.0 # Biến theo dõi kỷ lục

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # --- VALIDATION (test_seen) ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_seen_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test-Seen Acc: {val_acc:.4f}", end="")

    # Chỉ lưu nếu phá kỷ lục
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "aasist_best_model.pth")
        print(" => Kỷ lục mới! Đã lưu model.")
    else:
        print()

print(f"\nTraining hoàn tất! Test-Seen Accuracy cao nhất đạt được: {best_val_acc:.4f}")

# =========================
# EVALUATE ON TEST_UNSEEN
# =========================
print("\n" + "="*60)
print("EVALUATE ON TEST_UNSEEN (Completely New Data)")
print("="*60)

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_unseen_loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

test_unseen_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest-Unseen Accuracy: {test_unseen_acc:.4f}")