import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Union, Callable

from library.config import Config


def get_transforms(model_key: str, split: str = "valid") -> A.Compose:
    """
    Creates the appropriate Albumentations transform pipeline for a given model and data split.

    Args:
        model_key (str): The key corresponding to the model in Config.MODEL_CONFIGS
                         (e.g., 'swin_large', 'clip_large').
        split (str): The data split, either 'train' or 'valid' (or 'test').
                     'train' enables augmentation (HorizontalFlip).

    Returns:
        A.Compose: The composed Albumentations transformations.
    """
    if model_key not in Config.MODEL_CONFIGS:
        raise ValueError(f"Model key '{model_key}' not found in Config.MODEL_CONFIGS")

    model_cfg = Config.MODEL_CONFIGS[model_key]
    img_size = model_cfg["img_size"]
    mean = model_cfg["mean"]
    std = model_cfg["std"]

    transforms = []

    # Resize to the specific input size of the backbone
    transforms.append(A.Resize(height=img_size, width=img_size))

    # Augmentations for training
    if split == "train":
        transforms.append(A.HorizontalFlip(p=0.5))
        # Note: Additional heavy augmentations are avoided here to preserve
        # the pre-trained distribution for the feature extraction strategy.

    # Normalization and Tensor conversion
    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0))
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


class PawpularityDataset(Dataset):
    """
    PyTorch Dataset for the Pet Pawpularity Prediction task.
    Loads images, processes metadata features, and handles targets.
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[A.Compose] = None,
        return_target: bool = True,
        root_dir: str = Config.INPUT_DIR,
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV file (train.csv, val.csv, or test.csv).
            transform (A.Compose, optional): Albumentations transforms to apply to the image.
            return_target (bool): Whether to return the 'Pawpularity' target value.
                                  Should be False for the test set if target is missing.
            root_dir (str): Root directory containing the images.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.return_target = return_target
        self.root_dir = root_dir

        # Pre-validate existence of required columns
        if self.return_target and "Pawpularity" not in self.df.columns:
            raise ValueError(f"Target column 'Pawpularity' not found in {csv_path}")

        # Metadata feature columns defined in Config
        self.meta_cols = Config.METADATA_COLS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        row = self.df.iloc[idx]

        # 1. Load and Process Image
        # The 'file_path' column contains relative paths like 'train/{id}.jpg'
        img_path = os.path.join(self.root_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback or error handling for missing images
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback to basic tensor conversion if no transform provided
            image_tensor = ToTensorV2()(image=image)["image"]

        # 2. Extract Metadata Features
        # Extract binary flags as a float tensor
        meta_features = row[self.meta_cols].values.astype(np.float32)
        meta_tensor = torch.tensor(meta_features, dtype=torch.float32)

        # 3. Prepare Output Dictionary
        sample = {
            "image": image_tensor,
            "features": meta_tensor,
            "id": str(row["Id"]),
        }

        # 4. Extract Target (if requested)
        if self.return_target:
            target_val = row["Pawpularity"]
            # Return target as a float tensor of shape (1,)
            sample["target"] = torch.tensor([target_val], dtype=torch.float32)

        return sample


def load_dataset_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Helper function to load the dataframe, useful for quick inspection or
    custom splitting logic outside the Dataset class.
    """
    return pd.read_csv(csv_path)
