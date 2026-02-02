import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library import config, utils


class IcebergDataset(Dataset):
    def __init__(self, images, inc_angles, targets=None, transform=None, stats=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            inc_angles (np.ndarray): Shape (N,)
            targets (np.ndarray, optional): Shape (N,)
            transform (albumentations.Compose, optional): Augmentation pipeline
            stats (dict): Global stats for normalization {'min_ch0': val, 'max_ch0': val, ...}
        """
        self.images = images
        self.inc_angles = inc_angles
        self.targets = targets
        self.transform = transform
        self.stats = stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image: (3, 75, 75)
        img = self.images[idx].copy()

        # Global Min-Max Normalization per channel
        # Formula: (x - min) / (max - min)
        # No clipping is applied, allowing values > 1.0 or < 0.0
        if self.stats:
            for c in range(3):
                min_val = self.stats[f"min_ch{c}"]
                max_val = self.stats[f"max_ch{c}"]
                denom = max_val - min_val
                if denom == 0:
                    denom = 1.0
                img[c, :, :] = (img[c, :, :] - min_val) / denom

        # Convert to HWC for Albumentations
        img = np.transpose(img, (1, 2, 0))  # (75, 75, 3)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented[
                "image"
            ]  # This is now a tensor (3, 75, 75) due to ToTensorV2
        else:
            # Fallback if no transform provided (shouldn't happen with get_transforms)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()

        # Process metadata
        angle = self.inc_angles[idx]
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.targets is not None:
            target = self.targets[idx]
            target = torch.tensor(target, dtype=torch.float32)
            return img, angle, target
        else:
            return img, angle


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Rotational Invariance: 0, 90, 180, 270 degrees
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return A.Compose([ToTensorV2()])


def get_dataloaders(fold_idx, load_cached_data=True):
    """
    Creates train and validation dataloaders for a specific fold.

    Args:
        fold_idx (int): The current fold index (0 to NUM_FOLDS-1).
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load all data
    data = utils.load_and_process_data(load_cached_data=load_cached_data)

    all_images = data["train_images"]
    all_targets = data["train_targets"]
    all_inc_angles = data["train_inc_angles"]
    all_ids = data["train_ids"]
    stats = data["stats"]

    # Filter data to keep only samples in metadata/train.csv
    train_meta = pd.read_csv(config.TRAIN_META_FILE)
    allowed_ids = set(train_meta["id"].values)

    mask = np.isin(all_ids, list(allowed_ids))

    all_images = all_images[mask]
    all_targets = all_targets[mask]
    all_inc_angles = all_inc_angles[mask]

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Get indices for the requested fold
    # We iterate to find the specific fold indices
    fold_generator = skf.split(all_images, all_targets)
    train_idx, val_idx = next(x for i, x in enumerate(fold_generator) if i == fold_idx)

    # Create subsets
    train_images = all_images[train_idx]
    train_targets = all_targets[train_idx]
    train_angles = all_inc_angles[train_idx]

    val_images = all_images[val_idx]
    val_targets = all_targets[val_idx]
    val_angles = all_inc_angles[val_idx]

    # Create Datasets
    train_dataset = IcebergDataset(
        images=train_images,
        inc_angles=train_angles,
        targets=train_targets,
        transform=get_transforms(mode="train"),
        stats=stats,
    )

    val_dataset = IcebergDataset(
        images=val_images,
        inc_angles=val_angles,
        targets=val_targets,
        transform=get_transforms(mode="val"),
        stats=stats,
    )

    # Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates the test dataloader.

    Args:
        load_cached_data (bool): Whether to use cached processed data.

    Returns:
        DataLoader: The test data loader.
    """
    data = utils.load_and_process_data(load_cached_data=load_cached_data)

    test_images = data["test_images"]
    test_inc_angles = data["test_inc_angles"]
    stats = data["stats"]
    test_ids = data["test_ids"]  # Not used in dataset, but available if needed

    test_dataset = IcebergDataset(
        images=test_images,
        inc_angles=test_inc_angles,
        targets=None,
        transform=get_transforms(mode="test"),
        stats=stats,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
