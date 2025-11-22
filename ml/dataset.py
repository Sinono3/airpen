import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class AccelGyroDataset(Dataset):
    def __init__(self, npz_path, used_labels=['A', 'B', 'C', 'D'], normalize=True, ignore_gyro=True):
        data = np.load(npz_path)
        self.samples = []
        self.labels = []
        self.label_to_idx = {label: i for i, label in enumerate(data.keys())}

        for label, arr in data.items():
            if label not in used_labels:
                print(f"Warning: {label} class will not be included in the dataset")
                continue

            self.samples.append(arr)  # (N, 6, 132)
            self.labels.append(np.full(len(arr), self.label_to_idx[label]))

        self.samples = np.concatenate(self.samples, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.samples = torch.from_numpy(self.samples).to(torch.float32)
        self.labels = torch.from_numpy(self.labels).to(torch.long)

        # If ignore_gyro==True, only use the first three values, corresponding to accelerometer
        slice_idx = 3 if ignore_gyro else 6
        self.samples = self.samples[:, :slice_idx, :]

        # Normalize by mean and stddev
        if normalize:
            self.samples = (self.samples - self.samples.mean(dim=(0, 2), keepdim=True)) / self.samples.std(dim=(0,2), keepdim=True)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def create_dataloaders(npz_path, batch_size=32, val_ratio=0.10, test_ratio=0.20, seed=42, num_workers=2):
    dataset = AccelGyroDataset(npz_path)

    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        generator=generator,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader

