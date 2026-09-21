"""
PyTorch Dataset and Data Module for Spatiotemporal Wildfire Forecasting.

Includes chronological train/val/test splitting to avoid temporal data leakage.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from src.data.dataset_generator import WildfireDataCubeGenerator


class WildfireDataset(Dataset):
    """
    PyTorch Dataset wrapping 5-day spatiotemporal data cubes X and 24h/48h/72h target Y.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X  # [N, T=5, C=7, H=128, W=128]
        self.Y = Y  # [N, 3, H=128, W=128]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def get_wildfire_dataloaders(
    batch_size: int = 4,
    seq_len: int = 5,
    total_days: int = 60,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Creates chronologically split Train, Val, and Test DataLoaders.
    Chronological splitting ensures no future leakage into past validation/test sets.
    """
    generator = WildfireDataCubeGenerator(seed=seed)
    X, Y = generator.create_dataset_samples(sequence_length=seq_len, num_days=total_days)

    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))

    # Chronological slices
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    train_ds = WildfireDataset(X_train, Y_train)
    val_ds = WildfireDataset(X_val, Y_val)
    test_ds = WildfireDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'shapes': {
            'X_shape': list(X.shape),
            'Y_shape': list(Y.shape),
            'train_len': len(train_ds),
            'val_len': len(val_ds),
            'test_len': len(test_ds)
        }
    }


if __name__ == "__main__":
    loaders = get_wildfire_dataloaders()
    print("DataLoader summary:", loaders['shapes'])
    for bx, by in loaders['train']:
        print(f"Batch X: {bx.shape}, Batch Y: {by.shape}")
        break
