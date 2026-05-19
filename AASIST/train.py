import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from models.baseline import Full_AASIST_Model

from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =========================
# DATASET
# =========================

train_dataset = AudioDataset("dataset/train")
val_dataset = AudioDataset("dataset/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
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

    # --- VALIDATION ---
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}", end="")

    # Chỉ lưu nếu phá kỷ lục
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "aasist_best_model.pth")
        print(" => Kỷ lục mới! Đã lưu model.")
    else:
        print()

print(f"\nTraining hoàn tất! Val Accuracy cao nhất đạt được: {best_val_acc:.4f}")