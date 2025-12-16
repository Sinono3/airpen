import einops
import numpy as np
import processing
import torch
from torch.utils.data import DataLoader, Dataset


class AccelGyroDataset(Dataset):
    def __init__(self, x, y, mode='pca', normalize=None, smooth=True, augment=False):
        self.samples = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).long()
        self.eps = 1e-10
        self.normalize = normalize
        self.augment = augment
        self.smooth = smooth
        self.mode = mode

        if normalize == "dataset":
            self.samples = (self.samples - self.samples.mean(dim=(0, 2), keepdim=True)) / (
                self.samples.std(dim=(0, 2), keepdim=True) + self.eps
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].clone()
        
        # gravity = torch.tensor(processing.GRAVITY)
        # sample = processing.remove_gravity_constant(sample)
        # if self.augment:
        #     sample[:3] = processing.random_rotation(sample[:3], axis=gravity, angle_rad_std=0.02)
        # sample[:3] = processing.align_to_plane(sample[:3], gravity)

        # if self.smooth:
        #     sample = processing.smooth(sample)
        if self.normalize == 'sample':
            sample = (sample - sample.mean(dim=1, keepdim=True)) / (sample.std(dim=1, keepdim=True) + self.eps)

        match self.mode:
            case "acc":
                return sample[:3], self.labels[idx]
            case "acc+gyr":
                return sample, self.labels[idx]
            case "pca":
                acc = sample[:3]
                # pca = processing.canonical_transform_robust(acc)
                pca = processing.pca_transform_3_handedness(acc)             
                pca = processing.align_to_first_movement(pca)             

                # crop AUGMENTATION
                pca = processing.integrate_crop_resample_diff(
                    pca, crop_min=0.5, crop_max=1.0
                )
                # print(pca.shape)

                # random:
                # pca = pca[torch.randperm(3), :]
                return pca, self.labels[idx]
            case _:
                raise NotImplementedError()


def create_dataloaders(
    npz_path,
    batch_size=32,
    num_workers=2,
    seed=42,
    normalize=None,
):
    data = np.load(npz_path)
    required = ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in split dataset {npz_path}")

    dataset_args = dict(normalize=normalize, mode='pca', smooth=True)
    train_ds = AccelGyroDataset(data["train_x"], data["train_y"], augment=True, **dataset_args)
    val_ds = AccelGyroDataset(data["val_x"], data["val_y"], augment=False, **dataset_args)
    test_ds = AccelGyroDataset(data["test_x"], data["test_y"], augment=False, **dataset_args)

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
