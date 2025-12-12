import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AccelGyroDataset(Dataset):
    def __init__(self, x, y, normalize="sample", ignore_gyro=True, transforms=None):
        self.samples = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).long()
        self.eps = 1e-10
        self.normalize = normalize

        if ignore_gyro:
            self.samples = self.samples[:, :3, :]

        if normalize == "dataset":
            self.samples = (self.samples - self.samples.mean(dim=(0, 2), keepdim=True)) / (
                self.samples.std(dim=(0, 2), keepdim=True) + self.eps
            )
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.normalize == 'sample':
            sample = (sample - sample.mean(dim=1, keepdim=True)) / (sample.std(dim=1, keepdim=True) + self.eps)
        return sample, self.labels[idx]


def create_dataloaders(
    npz_path,
    batch_size=32,
    num_workers=2,
    seed=42,
    transforms=None,
    normalize="sample",
    ignore_gyro=True,
):
    data = np.load(npz_path)
    required = ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in split dataset {npz_path}")

    train_ds = AccelGyroDataset(data["train_x"], data["train_y"], normalize, ignore_gyro, transforms)
    val_ds = AccelGyroDataset(data["val_x"], data["val_y"], normalize, ignore_gyro, transforms)
    test_ds = AccelGyroDataset(data["test_x"], data["test_y"], normalize, ignore_gyro, transforms)

    generator = torch.Generator().manual_seed(seed)
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
