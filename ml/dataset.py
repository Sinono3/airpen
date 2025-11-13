import numpy as np
import torch
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

