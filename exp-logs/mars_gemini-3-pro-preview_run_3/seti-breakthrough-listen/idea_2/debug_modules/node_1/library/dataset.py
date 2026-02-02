import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from library.config import Config


class SETIDataset(Dataset):
    """
    Custom Dataset for loading SETI spectrograms.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'target' columns.
            transform (albumentations.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.file_paths = df["file_path"].values
        self.targets = df["target"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        file_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load spectrogram
        # Original shape: (6, 273, 256) -> (Channels, Frequency, Time)
        image = np.load(file_path).astype(np.float32)

        # Transpose to (H, W, C) for Albumentations compatibility
        # New shape: (273, 256, 6)
        image = np.transpose(image, (1, 2, 0))

        # Apply transforms (Resize, Augmentations)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Transpose back to (C, H, W) for PyTorch
        # New shape: (6, 224, 224)
        image = np.transpose(image, (2, 0, 1))

        # Convert to Tensor
        image = torch.tensor(image, dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, target


def get_transforms(phase="train"):
    """
    Returns the appropriate Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ]
        )
    else:
        # Validation and Test only require resizing
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
            ]
        )


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Performs Mixup augmentation on the batch.

    Args:
        x (torch.Tensor): Input batch of images.
        y (torch.Tensor): Input batch of targets.
        alpha (float): Mixup alpha parameter.
        device (str/torch.device): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Targets for first image set.
        y_b (torch.Tensor): Targets for second image set.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        pred: Model predictions.
        y_a: Targets A.
        y_b: Targets B.
        lam: Lambda coefficient.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataloaders(
    train_csv=Config.TRAIN_CSV,
    val_csv=Config.VAL_CSV,
    test_csv=Config.TEST_CSV,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        train_csv (str): Path to train metadata CSV.
        val_csv (str): Path to validation metadata CSV.
        test_csv (str): Path to test metadata CSV.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, uses a small subset of data.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Handle Debug Mode
    if debug:
        print(f"Debug mode enabled. Sampling {debug_sample_size} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        # We generally keep the test set intact or sample it similarly for a quick dry run
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)

    # Instantiate Datasets
    train_dataset = SETIDataset(train_df, transform=get_transforms("train"))
    val_dataset = SETIDataset(val_df, transform=get_transforms("valid"))
    test_dataset = SETIDataset(test_df, transform=get_transforms("test"))

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
