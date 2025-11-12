import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class NPZTimeSeriesDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.samples = []
        self.labels = []
        self.label_to_idx = {label: i for i, label in enumerate(data.keys())}

        for label, arr in data.items():
            if label == 'E' or label == 'F':
                continue

            self.samples.append(arr)  # (N, 132, 3)
            self.labels.append(np.full(len(arr), self.label_to_idx[label]))

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)  # (132, 3)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def create_dataloaders(npz_path, batch_size=32, val_ratio=0.10, test_ratio=0.20, seed=42):
    dataset = NPZTimeSeriesDataset(npz_path)
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    torch.manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (B, 32, 66)

            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (B, 64, 33)

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)

            nn.Flatten(),             # -> (B, 128)
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


model = Model(num_classes=4)
model.train()
train_loader, val_loader, test_loader = create_dataloaders("./ABCDEF.npz",)

