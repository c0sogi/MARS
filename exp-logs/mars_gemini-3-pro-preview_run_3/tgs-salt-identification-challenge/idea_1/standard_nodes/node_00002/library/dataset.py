import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.utils import MinMaxNormalizer

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Constants
INPUT_DIR = "./input"
IMG_ORIG_SIZE = 101
IMG_TARGET_SIZE = 128


def get_transforms(phase):
    """
    Returns the augmentation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        albumentations.Compose: The transform pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.PadIfNeeded(
                    min_height=IMG_TARGET_SIZE,
                    min_width=IMG_TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only padding is required
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_TARGET_SIZE,
                    min_width=IMG_TARGET_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


class SaltDataset(Dataset):
    def __init__(
        self, metadata_df, transform=None, depth_normalizer=None, mode="train"
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Dataframe containing image paths and metadata.
            transform (albumentations.Compose): Augmentation pipeline.
            depth_normalizer (MinMaxNormalizer): Fitted normalizer for depth values.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.metadata = metadata_df
        self.transform = transform
        self.mode = mode
        self.depth_normalizer = depth_normalizer

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["id"]

        # 1. Load Image
        img_path = os.path.join(INPUT_DIR, row["image_path"])
        # Load as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for safety, though metadata check ensures existence
            image = np.zeros((IMG_ORIG_SIZE, IMG_ORIG_SIZE), dtype=np.uint8)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # 2. Process Depth
        z = row["z"]
        if self.depth_normalizer:
            z_norm = self.depth_normalizer.transform(np.array([z]))[0]
        else:
            z_norm = 0.5  # Fallback default

        # Create depth channel (constant value map)
        depth_channel = np.full_like(image, z_norm)

        # 3. Construct 3-Channel Input
        # Channel 0: Image
        # Channel 1: Image (duplicated)
        # Channel 2: Depth
        input_volume = np.dstack([image, image, depth_channel])

        # 4. Load Mask (if training/validation)
        mask = None
        if self.mode in ["train", "valid"]:
            mask_path = os.path.join(INPUT_DIR, row["mask_path"])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                mask_img = np.zeros((IMG_ORIG_SIZE, IMG_ORIG_SIZE), dtype=np.uint8)

            # Normalize mask to 0 or 1
            mask = (mask_img > 127).astype(np.float32)

        # 5. Apply Transforms (Padding + Augmentation)
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=input_volume, mask=mask)
                input_volume = augmented["image"]
                mask = augmented["mask"]
                # Ensure mask has channel dimension (1, H, W)
                mask = mask.unsqueeze(0)
            else:
                augmented = self.transform(image=input_volume)
                input_volume = augmented["image"]

        # Return based on mode
        if self.mode in ["train", "valid"]:
            return input_volume, mask, image_id
        else:
            return input_volume, image_id


def get_dataloaders(batch_size=32, num_workers=2):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_meta = pd.read_csv("./metadata/train_metadata.csv")
    val_meta = pd.read_csv("./metadata/val_metadata.csv")
    test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Fit Depth Normalizer on Training Data
    # We use training data min/max to fit, but apply to all.
    # Alternatively, since we know the physics range or have all data,
    # fitting on train is the standard ML practice to avoid leakage.
    depth_normalizer = MinMaxNormalizer()
    depth_normalizer.fit(train_meta["z"].values)

    # Create Datasets
    train_dataset = SaltDataset(
        train_meta,
        transform=get_transforms("train"),
        depth_normalizer=depth_normalizer,
        mode="train",
    )

    val_dataset = SaltDataset(
        val_meta,
        transform=get_transforms("valid"),
        depth_normalizer=depth_normalizer,
        mode="valid",
    )

    test_dataset = SaltDataset(
        test_meta,
        transform=get_transforms("test"),
        depth_normalizer=depth_normalizer,
        mode="test",
    )

    # Create DataLoaders
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
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
