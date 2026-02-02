import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(data: str):
    """
    Returns the Albumentations composition based on the data mode (train/valid).

    Args:
        data (str): 'train' or 'valid'.

    Returns:
        A.Compose: The composition of transforms.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=Config.IMG_SIZE // 20,
                    max_width=Config.IMG_SIZE // 20,
                    min_holes=5,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


def load_dataset_dataframe(csv_path, cache_path, load_cached_data=True):
    """
    Loads the dataset metadata, using caching to speed up subsequent runs.

    Args:
        csv_path (str): Path to the source CSV file.
        cache_path (str): Path to the parquet cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}. Reloading from CSV.")

    # Load from CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Failed to save cache to {cache_path}: {e}")

    return df


class AppleDataset(Dataset):
    def __init__(self, df, transforms=None):
        """
        PyTorch Dataset for Apple Disease Detection.

        Args:
            df (pd.DataFrame): DataFrame containing 'file_path' and 'labels'.
            transforms (albumentations.Compose): Image augmentations.
        """
        self.df = df
        self.transforms = transforms
        self.file_paths = df["file_path"].values
        self.labels = df["labels"].values

        # Pre-compute label indices for speed
        # Config.CLASSES is sorted alphabetically
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(Config.CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 1. Load Image
        # file_path in metadata is relative to INPUT_DIR (e.g., "train_images/xyz.jpg")
        rel_path = self.file_paths[index]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        image = cv2.imread(full_path)
        if image is None:
            # Fallback for corrupt images or missing paths
            # Create a black image to prevent training crash
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Process Labels
        label_str = self.labels[index]
        # Create multi-hot vector
        target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        if isinstance(label_str, str):
            current_labels = label_str.split()
            for lbl in current_labels:
                if lbl in self.class_to_idx:
                    target[self.class_to_idx[lbl]] = 1.0

        return image, target
