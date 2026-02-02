import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    Reads images from disk and processes attributes into multi-hot vectors.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, file_path, attribute_ids).
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Transformations to apply to the image.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.root_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # file_path in metadata is relative to ./input (e.g., "train/xxx.png")
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Load image using OpenCV
        image = cv2.imread(image_path)

        # Handle missing or corrupt images gracefully
        if image is None:
            # Return a black image of size 256x256 (before crop) to prevent crashing
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            # Convert BGR (OpenCV default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            # Prepare multi-hot target vector
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

            attr_ids = row.get("attribute_ids", "")

            # Handle NaN values safely
            if pd.isna(attr_ids):
                attr_ids = ""
            else:
                attr_ids = str(attr_ids)

            if attr_ids.strip():
                # Parse space-separated IDs (e.g., "0 1 2")
                labels = [int(x) for x in attr_ids.split()]
                target[labels] = 1.0

            return image, target
        else:
            # Test mode: return image and image ID for submission mapping
            return image, row["id"]


def get_transforms(mode="train"):
    """
    Creates the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    # ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = [
        # Resize to 256x256 as per task description
        A.Resize(height=256, width=256),
        # Center crop to target size (224x224)
        A.CenterCrop(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
    ]

    if mode == "train":
        # Augmentation: Random Horizontal Flip
        transforms.append(A.HorizontalFlip(p=0.5))

    transforms.extend(
        [
            # Normalize pixel values
            A.Normalize(mean=mean, std=std),
            # Convert to PyTorch Tensor (C, H, W)
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def get_dataloaders(debug=False):
    """
    Initializes Datasets and DataLoaders for train, val, and test splits.

    Args:
        debug (bool): If True, uses a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load metadata CSVs
    # Ensure attribute_ids are read as strings to preserve formatting
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH, dtype={"attribute_ids": str})
    val_df = pd.read_csv(Config.VAL_METADATA_PATH, dtype={"attribute_ids": str})
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset data if in debug mode
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Get transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")

    # Create Datasets
    train_dataset = ArtworkDataset(train_df, mode="train", transform=train_transform)
    val_dataset = ArtworkDataset(val_df, mode="val", transform=val_transform)
    test_dataset = ArtworkDataset(test_df, mode="test", transform=test_transform)

    # Create DataLoaders
    # Pin memory speeds up host-to-device transfer
    # Num workers enables parallel data loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
