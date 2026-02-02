import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(mode: str, img_size: int):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        img_size (int): The target image resolution (e.g., 384 or 480).

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Scaled CoarseDropout as per strategy
                # Max hole size 100x100, Min hole size 16x16
                A.CoarseDropout(
                    max_holes=8,
                    max_height=100,
                    max_width=100,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_data(mode: str, load_cached_data: bool = True):
    """
    Loads metadata for the specified mode. Implements caching using Parquet.

    Processing Logic:
    - Reads metadata CSV.
    - Constructs full file paths.
    - For Train/Val: Generates binary targets for 'target_rust' and 'target_scab'.
      - target_rust = 1 if (rust == 1) OR (multiple_diseases == 1)
      - target_scab = 1 if (scab == 1) OR (multiple_diseases == 1)

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: Processed metadata.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Recomputing...")

    # 2. Compute from scratch
    if mode == "train":
        csv_path = Config.TRAIN_METADATA_PATH
    elif mode == "val":
        csv_path = Config.VAL_METADATA_PATH
    elif mode == "test":
        csv_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Construct full image paths
    # Metadata contains 'file_path' relative to input dir (e.g., images/Train_0.jpg)
    df["full_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # Process Targets for Train/Val
    if mode in ["train", "val"]:
        # Logic: Multi-Label Decomposition
        # Rust Target: Present in 'rust' class AND 'multiple_diseases' class
        df["target_rust"] = ((df["rust"] == 1) | (df["multiple_diseases"] == 1)).astype(
            int
        )

        # Scab Target: Present in 'scab' class AND 'multiple_diseases' class
        df["target_scab"] = ((df["scab"] == 1) | (df["multiple_diseases"] == 1)).astype(
            int
        )

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class AppleDataset(Dataset):
    """
    Dataset class for Apple Disease Detection.
    """

    def __init__(self, df: pd.DataFrame, img_size: int, mode: str = "train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata and paths.
            img_size (int): Target image size for resizing.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transforms = get_transforms(mode, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["full_path"]

        # Load Image
        image = cv2.imread(path)
        if image is None:
            # Fallback for missing images (though metadata check should prevent this)
            # Create a black image to avoid crashing
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Return logic based on mode
        if self.mode == "test":
            return image, row["image_id"]
        else:
            # Return targets: [Is_Rust, Is_Scab]
            targets = torch.tensor(
                [row["target_rust"], row["target_scab"]], dtype=torch.float32
            )
            return image, targets
