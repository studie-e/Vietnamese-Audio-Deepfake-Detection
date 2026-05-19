from dataset import AudioDataset
from torch.utils.data import DataLoader

dataset = AudioDataset("dataset/train")

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

for x, y in loader:

    print("Waveform shape:", x.shape)
    print("Labels:", y)

    break