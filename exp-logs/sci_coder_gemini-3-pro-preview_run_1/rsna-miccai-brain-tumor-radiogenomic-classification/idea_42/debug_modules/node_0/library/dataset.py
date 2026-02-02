import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library import config, preprocessing


class MGMTDataset(Dataset):
    """
    PyTorch Dataset for MGMT Promoter Methylation Prediction.
    Wraps preprocessed numpy arrays and applies Albumentations transforms.
    """

    def __init__(self, images, targets=None, transform=None, is_test=False):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            targets (np.ndarray, optional): Array of labels or IDs.
            transform (A.Compose, optional): Albumentations augmentation pipeline.
            is_test (bool): Flag to indicate if this is the test set (returns IDs instead of labels).
        """
        self.images = images
        self.targets = targets
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, 3) - float32 [0, 1]
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.tensor(image).permute(2, 0, 1)

        # Handle Targets
        if self.targets is not None:
            target = self.targets[idx]
            if self.is_test:
                # For test set, return the Subject ID (LongTensor)
                return image, torch.tensor(target, dtype=torch.long)
            else:
                # For train/val, return the Label (FloatTensor for BCE Loss)
                return image, torch.tensor(target, dtype=torch.float32)
        else:
            # Should not happen in this pipeline, but safe fallback
            return image, torch.tensor(0.0, dtype=torch.float32)


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Strategy:
    - Train: Spatially-Constrained Augmentation (No Translation/Scaling).
    - Val/Test: Normalization only.
    """
    # ImageNet normalization statistics
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                # Spatial Augmentations (Preserving Center Anchoring)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # Non-Rigid Deformations
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                # Normalization & Tensor Conversion
                # max_pixel_value=1.0 because input is float32 [0, 1]
                A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Validation/Test: Normalize to match training distribution
                A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Main function to initialize datasets and dataloaders.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Use config defaults if not specified
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # 1. Load Data from Preprocessing Module
    # Returns: ((train_X, train_y), (val_X, val_y), (test_X, test_ids))
    (train_data, val_data, test_data) = preprocessing.prepare_datasets(
        load_cached_data=load_cached_data
    )

    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_ids = test_data

    # 2. Instantiate Datasets
    train_dataset = MGMTDataset(
        train_X, train_y, transform=get_transforms("train"), is_test=False
    )

    val_dataset = MGMTDataset(
        val_X, val_y, transform=get_transforms("val"), is_test=False
    )

    test_dataset = MGMTDataset(
        test_X, test_ids, transform=get_transforms("test"), is_test=True
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batch to stabilize BatchNorm
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
