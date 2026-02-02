import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import seed_everything


def get_transforms(img_size=224, mode="train"):
    """
    Returns the image transformations for the dataset.

    Args:
        img_size (int): The target size for resizing (height and width).
        mode (str): 'train' or 'valid'/'test'. Currently uses the same
                    deterministic transforms for all modes as TTA is handled
                    externally or via specific TTA pipelines.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )


class PetDataset(Dataset):
    """
    PyTorch Dataset for loading Pet Pawpularity data.

    Returns:
        image (torch.Tensor): Preprocessed image tensor (C, H, W).
        metadata (torch.Tensor): Scaled dense metadata vector.
        target (torch.Tensor): Pawpularity score (or 0.0 for test).
        img_id (str): The unique Pet Profile ID.
    """

    def __init__(self, csv_path, transform=None, debug=False):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            transform (A.Compose, optional): Albumentations transforms.
            debug (bool): If True, restricts dataset to a small subset for debugging.
        """
        self.csv_path = csv_path
        self.transform = transform
        self.debug = debug

        # Load Metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Handle Debug Mode
        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Pre-compute paths
        # file_path in metadata is relative (e.g., "train/id.jpg")
        # We join it with INPUT_DIR
        self.df["full_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(Config.INPUT_DIR, x)
        )

        # Store basic columns
        self.ids = self.df["Id"].values
        self.image_paths = self.df["full_path"].values

        # Metadata Features
        self.meta_cols = Config.METADATA_COLS
        # Ensure all meta columns exist, fill with 0 if missing (robustness)
        for col in self.meta_cols:
            if col not in self.df.columns:
                self.df[col] = 0

        self.meta_features = self.df[self.meta_cols].values.astype(np.float32)

        # Target
        if "Pawpularity" in self.df.columns:
            self.targets = self.df["Pawpularity"].values.astype(np.float32)
        else:
            # For test set, use placeholder
            self.targets = np.zeros(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_id = self.ids[idx]

        # Load Image
        # cv2 loads in BGR, convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images (should not happen given metadata verification)
            # Create a black image of default size
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback transform if none provided
            t = get_transforms(Config.IMG_SIZE)
            augmented = t(image=image)
            image = augmented["image"]

        # Process Metadata
        # Scale metadata by the configured factor
        meta = self.meta_features[idx] * Config.METADATA_SCALE
        meta = torch.tensor(meta, dtype=torch.float32)

        # Process Target
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, meta, target, img_id
