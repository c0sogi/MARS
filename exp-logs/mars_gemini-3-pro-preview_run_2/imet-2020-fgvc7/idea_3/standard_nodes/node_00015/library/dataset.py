import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Define normalization constants (ImageNet defaults)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    Reads images from disk and processes labels from metadata.
    """

    def __init__(self, df, mode="train", transform=None, input_dir=Config.input_dir):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (id, attribute_ids, file_path).
            mode (str): 'train', 'val', or 'test'. Determines if labels are processed.
            transform (A.Compose): Albumentations transforms to apply.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.mode = mode
        self.transform = transform
        self.input_dir = input_dir
        self.num_classes = Config.num_classes

        # Pre-process file paths
        # The metadata file_path is relative (e.g., "train/abc.png")
        self.file_paths = self.df["file_path"].values

        # Pre-process labels for train/val
        if self.mode != "test":
            self.labels = self.df["attribute_ids"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        rel_path = self.file_paths[idx]
        img_path = os.path.join(self.input_dir, rel_path)

        # Load image
        # cv2 loads in BGR, convert to RGB
        image = cv2.imread(img_path)

        if image is None:
            # Fallback for missing/corrupt images
            # Create a black image to prevent crashing
            image = np.zeros((Config.image_size, Config.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform
            t = A.Compose(
                [
                    A.Resize(Config.image_size, Config.image_size),
                    A.Normalize(mean=MEAN, std=STD),
                    ToTensorV2(),
                ]
            )
            image = t(image=image)["image"]

        # Handle Labels
        if self.mode != "test":
            label_str = self.labels[idx]
            target = torch.zeros(self.num_classes, dtype=torch.float32)

            if isinstance(label_str, str) and len(label_str) > 0:
                # Parse space-separated IDs
                try:
                    indices = [int(x) for x in label_str.split()]
                    target[indices] = 1.0
                except ValueError:
                    pass  # Handle potential parsing errors gracefully

            return image, target
        else:
            # For test, return image and the ID (useful for submission mapping)
            image_id = self.df.iloc[idx]["id"]
            return image, image_id


def get_transforms(mode="train", image_size=Config.image_size):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train' or 'val'/'test'.
        image_size (int): Target image size.

    Returns:
        A.Compose: Transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Random Resized Crop as specified: scale 0.75-1.0
                A.RandomResizedCrop(
                    size=(image_size, image_size), scale=(0.75, 1.0), p=1.0
                ),
                # Random Horizontal Flip
                A.HorizontalFlip(p=0.5),
                # Normalize
                A.Normalize(mean=MEAN, std=STD),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Val / Test
        return A.Compose(
            [
                # Deterministic Resize to target size
                A.Resize(height=image_size, width=image_size),
                # Normalize
                A.Normalize(mean=MEAN, std=STD),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=Config.batch_size, num_workers=Config.num_workers, debug=Config.debug
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)
    test_df = pd.read_csv(Config.test_metadata_path)

    if debug:
        print("DEBUG MODE: Subsetting data...")
        train_df = train_df.iloc[:1000]
        val_df = val_df.iloc[:500]
        test_df = test_df.iloc[:100]

    # Create Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # Create Datasets
    train_dataset = ArtworkDataset(train_df, mode="train", transform=train_transform)
    val_dataset = ArtworkDataset(val_df, mode="val", transform=val_transform)
    test_dataset = ArtworkDataset(test_df, mode="test", transform=val_transform)

    # Create DataLoaders
    # Pin memory enables faster data transfer to CUDA
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
