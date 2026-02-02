import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from library.config import Config
from library.utils import read_image


class SteganalysisDataset(Dataset):
    """
    Dataset class for ALASKA2 Steganalysis.
    Reads images, extracts Y channel, and applies augmentations.
    """

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
            transform (albumentations.Compose): Albumentations transforms to apply.
        """
        self.df = df
        self.transform = transform
        # Pre-convert paths to absolute paths to save time in __getitem__
        self.image_paths = [
            os.path.join(Config.INPUT_DIR, p) for p in df["image_path"].values
        ]
        self.labels = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read image (returns Y channel as numpy array, shape (H, W))
        image = read_image(self.image_paths[idx])

        # Albumentations expects HWC or HW.
        # Since we have HW, we can pass it directly.
        # However, for consistency with transforms that might expect channels,
        # we often keep it as is. Albumentations handles 2D images fine.

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Ensure image is float32 and normalized to [0, 1]
        # ToTensorV2 converts to tensor but preserves dtype if not specified.
        # If the transform pipeline didn't normalize, we do it here.
        # Note: If ToTensorV2 was used, image is a Tensor.
        if isinstance(image, torch.Tensor):
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
        else:
            # If still numpy
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image)
            # Add channel dimension if missing (H, W) -> (1, H, W)
            if image.ndim == 2:
                image = image.unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label


def get_transforms(mode="train"):
    """
    Returns the appropriate Albumentations transforms for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Dihedral Group Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Convert to Tensor (H, W) -> (C, H, W) implicitly handled by logic in __getitem__
                # or explicitly here. We'll rely on manual tensor conversion in __getitem__
                # to ensure correct channel dimension for grayscale.
            ]
        )
    else:
        # Validation/Test: No geometric augmentations
        return A.Compose([])


def get_dataloaders(debug=False):
    """
    Creates and returns the training and validation DataLoaders.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"Debug mode: Train size={len(train_df)}, Val size={len(val_df)}")

    # 2. Implement WeightedRandomSampler for Class Balancing
    # We want to oversample the minority class (Cover, Label 0)
    # Target: Cover images should appear with frequency proportional to Config.COVER_BATCH_RATIO

    # Extract labels from training dataframe
    y_train = train_df["label"].values

    # Count classes
    class_counts = np.bincount(y_train.astype(int))
    # class_counts[0] is Cover count, class_counts[1] is Stego count

    # Calculate weights for each class
    # Weight = 1.0 / count
    class_weights = 1.0 / class_counts

    # Assign a weight to each sample corresponding to its class
    sample_weights = class_weights[y_train.astype(int)]

    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights).double()

    # Create Sampler
    # num_samples can be set to len(train_df) or larger/smaller.
    # Usually len(train_df) is standard.
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_df), replacement=True
    )

    # 3. Create Datasets
    train_dataset = SteganalysisDataset(
        df=train_df, transform=get_transforms(mode="train")
    )

    val_dataset = SteganalysisDataset(df=val_df, transform=get_transforms(mode="val"))

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,  # Sampler is mutually exclusive with shuffle
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates and returns the test DataLoader.
    """
    # Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        print("Warning: Test metadata not found. Returning None.")
        return None

    test_df = pd.read_csv(Config.TEST_METADATA)

    test_dataset = SteganalysisDataset(
        df=test_df, transform=get_transforms(mode="test")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
