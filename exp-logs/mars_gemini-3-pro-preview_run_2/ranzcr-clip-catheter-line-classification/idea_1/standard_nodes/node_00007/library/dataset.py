import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter and Line Detection.
    Reads images from disk and applies transformations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (paths, labels, IDs).
            transforms (albumentations.Compose): Transformations to apply.
            mode (str): 'train', 'val', or 'test'. Determines return values.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-extract paths and IDs to avoid DataFrame overhead in __getitem__
        self.file_paths = df["file_path"].tolist()
        self.uids = df["StudyInstanceUID"].tolist()

        # Extract labels for supervised modes
        if self.mode in ["train", "val"]:
            self.labels = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct absolute file path
        # Config.INPUT_DIR is "./input", file_paths are relative like "train/xxx.jpg"
        img_path = os.path.join(Config.INPUT_DIR, self.file_paths[idx])

        # Load image using OpenCV
        image = cv2.imread(img_path)

        # Safety check for missing or corrupt images
        if image is None:
            # Return a black image of correct size to prevent crashing
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            label = torch.tensor(self.labels[idx])
            return image, label
        else:
            # For test mode, return ID to map predictions later
            return image, self.uids[idx]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for the specified mode.

    Args:
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic resizing.
    """
    if mode == "train":
        return A.Compose(
            [
                # Cite solution_lesson_node_00006: Apply intensity-based augmentations before geometric ones
                A.CLAHE(p=0.5),
                # Cite solution_lesson_node_00001: Use LongestMaxSize and PadIfNeeded to preserve aspect ratio
                A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=Config.IMAGE_SIZE,
                    min_width=Config.IMAGE_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Cite solution_lesson_node_00001: Use LongestMaxSize and PadIfNeeded to preserve aspect ratio
                A.LongestMaxSize(max_size=Config.IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=Config.IMAGE_SIZE,
                    min_width=Config.IMAGE_SIZE,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_df=None,
    val_df=None,
    test_df=None,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        train_df (pd.DataFrame): Training metadata. Loads from disk if None.
        val_df (pd.DataFrame): Validation metadata. Loads from disk if None.
        test_df (pd.DataFrame): Test metadata. Loads from disk if None.
        batch_size (int): Batch size for loaders.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # Load DataFrames from metadata CSVs if not provided
    if train_df is None and os.path.exists(Config.TRAIN_METADATA_PATH):
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if val_df is None and os.path.exists(Config.VAL_METADATA_PATH):
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    if test_df is None and os.path.exists(Config.TEST_METADATA_PATH):
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply debug subsampling
    if debug:
        print("DEBUG Mode: Subsampling datasets...")
        if train_df is not None:
            train_df = train_df.head(batch_size * 2)
        if val_df is not None:
            val_df = val_df.head(batch_size)
        if test_df is not None:
            test_df = test_df.head(batch_size)

    loaders = {}

    # Create Train DataLoader
    if train_df is not None:
        train_ds = CatheterDataset(
            train_df, transforms=get_transforms(mode="train"), mode="train"
        )
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    # Create Validation DataLoader
    if val_df is not None:
        val_ds = CatheterDataset(
            val_df, transforms=get_transforms(mode="val"), mode="val"
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    # Create Test DataLoader
    if test_df is not None:
        test_ds = CatheterDataset(
            test_df, transforms=get_transforms(mode="test"), mode="test"
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    return loaders
