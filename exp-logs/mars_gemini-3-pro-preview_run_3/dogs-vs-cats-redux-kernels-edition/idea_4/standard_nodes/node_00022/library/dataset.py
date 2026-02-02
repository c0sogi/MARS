import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train", image_size=Config.IMAGE_SIZE):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' for augmentation, 'valid' or 'test' for deterministic resizing.
        image_size (int): Target spatial dimension.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Scale (0.8, 1.0) prevents zooming in too close, preserving context
                A.RandomResizedCrop(
                    height=image_size, width=image_size, scale=(0.8, 1.0), p=1.0
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Deterministic resize for validation and testing
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_data(mode="train", load_cached_data=True):
    """
    Loads the metadata dataframe. Implements caching to parquet format.

    Args:
        mode (str): 'train' (combines train+val metadata) or 'test'.
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        pd.DataFrame: The loaded metadata.
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    if mode == "train":
        cache_file = os.path.join(cache_dir, "full_train_data.parquet")

        # 1. Try to load cache
        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # 2. Process from scratch (Merge train and val for K-Fold)
        if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
            Config.VAL_METADATA_PATH
        ):
            raise FileNotFoundError("Metadata CSVs not found in ./metadata")

        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Combine to allow 5-Fold CV on full data
        full_df = pd.concat([train_df, val_df], ignore_index=True)

        # 3. Save to cache
        full_df.to_parquet(cache_file)
        return full_df

    elif mode == "test":
        cache_file = os.path.join(cache_dir, "test_data.parquet")

        # 1. Try to load cache
        if load_cached_data and os.path.exists(cache_file):
            return pd.read_parquet(cache_file)

        # 2. Process from scratch
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError("Test metadata CSV not found in ./metadata")

        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # 3. Save to cache
        test_df.to_parquet(cache_file)
        return test_df

    else:
        raise ValueError(f"Unknown mode: {mode}")


class CatDogDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    """

    def __init__(self, df, transforms=None, input_dir=Config.INPUT_DIR):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'filepath'.
            transforms (A.Compose): Albumentations transforms.
            input_dir (str): Root directory for images.
        """
        self.df = df
        self.transforms = transforms
        self.input_dir = input_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        rel_path = row["filepath"]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for corrupt/missing images (unlikely given metadata check)
            # Return a black image to prevent crash
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Retrieve Label (if training data)
        # Config.NUM_CLASSES = 1 implies Binary Classification (BCEWithLogitsLoss)
        # We return float for BCE
        if "label" in row:
            label = torch.tensor(row["label"], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)

        # Retrieve ID (if test data)
        if "id" in row:
            img_id = row["id"]
        else:
            img_id = -1

        return image, label, img_id
