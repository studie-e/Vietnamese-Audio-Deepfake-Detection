from torch.utils.data import Dataset
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

TARGET_LENGTH = 80000


class AudioDataset(Dataset):

    def __init__(self, root_dir):

        self.files = []

        bonafide_files = list(
            Path(root_dir, "bonafide").glob("*.wav")
        )

        spoof_files = list(
            Path(root_dir, "spoof").glob("*.wav")
        )

        for file in bonafide_files:
            self.files.append((str(file), 0))

        for file in spoof_files:
            self.files.append((str(file), 1))

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