import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.ColorJitter(
                    brightness=Config.AUG_COLOR_JITTER_BRIGHTNESS,
                    contrast=Config.AUG_COLOR_JITTER_CONTRAST,
                    saturation=Config.AUG_COLOR_JITTER_SATURATION,
                    hue=Config.AUG_COLOR_JITTER_HUE,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class ArtworkDataset(Dataset):
    """
    PyTorch Dataset for Artwork Attribute Labeling.
    Reads images via OpenCV and processes multi-label targets.
    """

    def __init__(self, df, mode="train", transforms=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            transforms (A.Compose): Albumentations transforms.
        """
        self.df = df
        self.mode = mode
        self.transforms = transforms

        # Pre-compute paths and ids for faster access
        self.file_paths = df["file_path"].values
        self.ids = df["id"].values

        # Process labels for train/val
        if self.mode != "test":
            # Ensure attribute_ids are strings and handle NaNs
            self.labels = df["attribute_ids"].fillna("").astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full image path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing/corrupt images: create a black image
            # This prevents crashing during training
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            # Return image and ID for submission mapping
            return image, self.ids[idx]
        else:
            # Parse labels into multi-hot vector
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            label_str = self.labels[idx]

            if len(label_str) > 0:
                # Split string "0 1 2" -> [0, 1, 2]
                label_indices = [int(x) for x in label_str.split()]
                target[label_indices] = 1.0

            return image, target


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(
        Config.TRAIN_METADATA, dtype={"id": str, "attribute_ids": str}
    )
    val_df = pd.read_csv(Config.VAL_METADATA, dtype={"id": str, "attribute_ids": str})
    test_df = pd.read_csv(Config.TEST_METADATA, dtype={"id": str})

    # Debug Mode: Subset data
    if debug:
        train_df = train_df.head(2000)
        val_df = val_df.head(500)
        test_df = test_df.head(500)
        print(
            f"[DEBUG] Subsetting data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    # Initialize Datasets
    train_dataset = ArtworkDataset(
        train_df, mode="train", transforms=get_transforms(mode="train")
    )

    val_dataset = ArtworkDataset(
        val_df, mode="val", transforms=get_transforms(mode="val")
    )

    test_dataset = ArtworkDataset(
        test_df, mode="test", transforms=get_transforms(mode="test")
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
