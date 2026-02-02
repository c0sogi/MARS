import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformations for the specific data type.

    Args:
        data_type (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The composition of transformations.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                # Minimal rotation and shift as per Idea description
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMAGE_SIZE * 0.05),
                    max_width=int(Config.IMAGE_SIZE * 0.05),
                    fill_value=0,
                    p=0.2,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class CatheterDataset(Dataset):
    """
    PyTorch Dataset for Catheter Detection.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.file_paths = df["file_path"].values

        # Pre-extract labels for train/val modes to avoid overhead in __getitem__
        if self.mode in ["train", "valid"]:
            self.labels = df[Config.TARGET_COLS].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full file path
        # file_path in metadata is relative to input dir (e.g., "train/uid.jpg")
        rel_path = self.file_paths[idx]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image using OpenCV
        # IMREAD_UNCHANGED allows us to inspect channels before conversion
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        # Fallback for missing images (though metadata validation should prevent this)
        if image is None:
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)

        # Convert to RGB
        # If image is grayscale (H, W), convert to (H, W, 3)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # If image is (H, W, C), convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode in ["train", "valid"]:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            # Test mode only returns image
            return image


def load_df(split="train"):
    """
    Loads the metadata dataframe for the specified split.
    """
    if split == "train":
        path = Config.TRAIN_METADATA
    elif split == "valid":
        path = Config.VAL_METADATA
    elif split == "test":
        path = Config.TEST_METADATA
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)
    return df


def get_dataloader(
    split, batch_size=None, shuffle=None, num_workers=None, debug_size=None
):
    """
    Creates a DataLoader for the specified split.

    Args:
        split (str): 'train', 'valid', or 'test'.
        batch_size (int): Batch size. Defaults to Config.BATCH_SIZE.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.
        num_workers (int): Number of workers. Defaults to Config.NUM_WORKERS.
        debug_size (int): If provided, limits the dataset size for debugging.

    Returns:
        DataLoader: The configured PyTorch DataLoader.
    """
    # Set defaults based on Config if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if shuffle is None:
        shuffle = split == "train"

    # Load Data
    df = load_df(split)

    # Debugging: Limit size
    if debug_size is not None:
        df = df.iloc[:debug_size]
    elif Config.DEBUG:
        limit = 100
        if split == "train" and Config.MAX_TRAIN_SAMPLES:
            limit = Config.MAX_TRAIN_SAMPLES
        elif split == "valid" and Config.MAX_VAL_SAMPLES:
            limit = Config.MAX_VAL_SAMPLES
        df = df.iloc[:limit]

    # Define Transforms
    transforms = get_transforms(data_type=split)

    # Create Dataset
    dataset = CatheterDataset(df, transforms=transforms, mode=split)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),  # Drop last incomplete batch during training
    )

    return loader
