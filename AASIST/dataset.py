from torch.utils.data import Dataset
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_LENGTH = 80000


class AudioDataset(Dataset):
    """
    Load audio files từ vispoofdb/data/clean_data/ theo metadata.csv
    Filter theo split: train / test_seen / test_unseen
    """

    def __init__(self, metadata_path: str, split: str = "train"):
        """
        Args:
            metadata_path: Đường dẫn tới metadata.csv
            split: "train", "test_seen", hoặc "test_unseen"
        """
        self.split = split
        self.files = []

        # Đọc metadata
        df = pd.read_csv(metadata_path)

        # Filter theo split
        df = df[df["split"] == split]

        # Đường dẫn base (thư mục chứa metadata)
        base_dir = Path(metadata_path).parent

        # Load file paths và labels
        for _, row in df.iterrows():
            file_path = base_dir / row["file_path"]
            label = 0 if row["label"] == "real" else 1

            if file_path.exists():
                self.files.append((str(file_path), label))

        print(f"[{split}] Loaded {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path, label = self.files[idx]

        # load audio
        waveform, sr = librosa.load(
            path,
            sr=16000,
            mono=True
        )

        waveform = torch.tensor(
            waveform,
            dtype=torch.float32
        )

        # normalize length
        if waveform.shape[0] < TARGET_LENGTH:

            pad_size = TARGET_LENGTH - waveform.shape[0]

            waveform = F.pad(
                waveform,
                (0, pad_size)
            )

        else:
            waveform = waveform[:TARGET_LENGTH]

        return waveform, torch.tensor(label)