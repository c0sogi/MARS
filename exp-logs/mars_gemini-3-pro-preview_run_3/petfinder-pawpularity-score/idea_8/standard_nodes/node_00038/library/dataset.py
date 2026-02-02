import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


class PetDataset(Dataset):
    """
    Dataset class for Pet Pawpularity Prediction.
    Handles image loading, preprocessing, and Test-Time Augmentation (TTA).
    """

    def __init__(self, df: pd.DataFrame, mode: str = "train", tta: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (Id, file_path, features, target).
            mode (str): 'train', 'val', or 'test'. Determines if targets are returned.
            tta (bool): If True, returns a stacked tensor of original and flipped images.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tta = tta

        # Define transformations
        # We use standard ImageNet normalization as expected by most pre-trained backbones
        self.transform = A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE[0], width=Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # 1. Load Image
        # file_path in metadata is relative to input dir (e.g., "train/xyz.jpg")
        image_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image in RGB
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for safety, though metadata generation ensures existence
            # Create a black image to prevent crash
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms & TTA
        if self.tta:
            # Original
            aug_orig = self.transform(image=image)["image"]

            # Flipped
            image_flipped = cv2.flip(image, 1)  # 1 indicates horizontal flip
            aug_flip = self.transform(image=image_flipped)["image"]

            # Stack: (2, C, H, W)
            image_tensor = torch.stack([aug_orig, aug_flip], dim=0)
        else:
            # Single image: (C, H, W)
            image_tensor = self.transform(image=image)["image"]

        # 3. Extract Metadata Features
        # Extract the 12 binary columns defined in Config
        meta_features = row[Config.METADATA_COLS].values.astype(np.float32)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)

        # 4. Prepare Output
        sample = {
            "id": row["Id"],
            "image": image_tensor,
            "metadata": meta_tensor,
        }

        # 5. Extract Target (if available)
        if self.mode in ["train", "val"]:
            target = row[Config.TARGET_COL]
            sample["target"] = torch.tensor(target, dtype=torch.float32)

        return sample
