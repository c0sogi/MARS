import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import get_label_mappings


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline based on the data split.

    Args:
        data (str): One of 'train', 'valid', 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    image_size = Config.IMAGE_SIZE
    mean = Config.MEAN
    std = Config.STD

    if data == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.08, 1.0)),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    elif data in ["valid", "test"]:
        # Deterministic processing: Resize to slightly larger then CenterCrop
        resize_dim = int(image_size * 1.14)
        return A.Compose(
            [
                A.Resize(size=(resize_dim, resize_dim)),
                A.CenterCrop(size=(image_size, image_size)),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class PlantDataset(Dataset):
    """
    PyTorch Dataset for Plant Image Classification.
    Loads images via OpenCV and applies Albumentations transforms.
    """

    def __init__(self, df, transforms=None, mode="train", label2id=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (file_path, category_id, etc.).
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'. Determines return values.
            label2id (dict): Mapping from raw category_id to contiguous index.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Pre-extract paths and labels to avoid dataframe overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        if self.mode != "test":
            self.labels = self.df["category_id"].values
            if label2id is not None:
                self.labels = np.array([label2id[c] for c in self.labels])
        else:
            self.image_ids = self.df["image_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        file_path = os.path.join(self.input_dir, self.file_paths[idx])

        # Load image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Handle missing/corrupt files gracefully by returning a zero tensor
            # (Though metadata generation should have filtered these)
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = ToTensorV2()(image=image)["image"]

        if self.mode == "test":
            # For test, return image and image_id (for submission)
            return image, self.image_ids[idx]
        else:
            # For train/valid, return image and label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return image, label


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=None,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders for train, val, or test splits.

    Args:
        split (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.
        num_workers (int): Number of worker threads.
        debug (bool): If True, loads a small subset of data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Determine file path and mode
    if split == "train":
        csv_path = Config.TRAIN_CSV
        mode = "train"
        default_shuffle = True
    elif split == "val":
        csv_path = Config.VAL_CSV
        mode = "valid"
        default_shuffle = False
    elif split == "test":
        csv_path = Config.TEST_CSV
        mode = "test"
        default_shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    # Set shuffle default if not provided
    if shuffle is None:
        shuffle = default_shuffle

    # Load Metadata
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debug Mode: Sample subset
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE).copy()
        print(f"[DEBUG] Loaded {len(df)} samples for split '{split}'")

    # Get mappings (always use full training set for consistency)
    label2id, _ = get_label_mappings()

    # Create Dataset
    transforms = get_transforms(data=mode)
    dataset = PlantDataset(df, transforms=transforms, mode=mode, label2id=label2id)

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
