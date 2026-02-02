import os
import cv2
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def load_train_metadata(load_cached_data=True):
    """
    Loads and combines training and validation metadata for 5-Fold CV.
    Implements caching using Parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "full_train_metadata.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass  # Fallback to re-creation if cache is corrupt

    # 2. Create from scratch
    # Load separate splits provided by the system
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError("Metadata CSVs not found in ./metadata")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine for full Cross-Validation
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)

    return full_df


def load_test_metadata(load_cached_data=True):
    """
    Loads test metadata. Implements caching using Parquet.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "test_metadata.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            pass

    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError("Test metadata CSV not found")

    df = pd.read_csv(Config.TEST_METADATA_PATH)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_transforms(img_size, mode="train"):
    """
    Generates the augmentation pipeline based on the strategy.

    Args:
        img_size (int): Target resolution (e.g., 224, 256, 288).
        mode (str): 'train' or 'val'/'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Context-Preserving Augmentation: Scale (0.8, 1.0)
                A.RandomResizedCrop(
                    height=img_size,
                    width=img_size,
                    scale=(0.8, 1.0),
                    interpolation=cv2.INTER_CUBIC,
                ),
                # Color Jitter with intensity >= 0.2
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic Resize with Bicubic Interpolation
        return A.Compose(
            [
                A.Resize(
                    height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class DogCatDataset(Dataset):
    """
    PyTorch Dataset for Dog vs Cat classification.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'filepath' and 'label' (or 'id').
            transform (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train' (returns img, label) or 'test' (returns img, id).
        """
        self.df = df
        self.transform = transform
        self.mode = mode
        self.input_dir = Config.INPUT_DIR

        # Pre-check columns
        if self.mode == "train":
            if "label" not in self.df.columns:
                raise ValueError(
                    "DataFrame must contain 'label' column for training mode."
                )
            self.labels = self.df["label"].values
        else:
            if "id" not in self.df.columns:
                raise ValueError("DataFrame must contain 'id' column for test mode.")
            self.ids = self.df["id"].values

        self.filepaths = self.df["filepath"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.filepaths[idx]
        full_path = os.path.join(self.input_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for corrupt images (though metadata check passed)
            # Return a black image to prevent crashing
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.mode == "train":
            label = self.labels[idx]
            # Return float label for BCEWithLogitsLoss
            return image, torch.tensor(label, dtype=torch.float32)
        else:
            id_val = self.ids[idx]
            return image, id_val
