import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(phase: str):
    """
    Returns the image transformation pipeline using Albumentations.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    # Standard ImageNet normalization statistics
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # InChI recognition is sensitive to orientation, so we avoid
                # geometric augmentations like flips or rotations that would
                # change the chemical structure's meaning or text orientation.
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test phases
        return A.Compose(
            [
                A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for loading chemical images and InChI labels.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Tokenizer, transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'image_id', 'InChI', and 'file_path'.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            transform (A.Compose, optional): Albumentations transformations.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Image
        # Construct full path from relative path in metadata
        file_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, file_path)

        # Load using OpenCV (loads as BGR by default)
        image = cv2.imread(full_path)

        if image is None:
            # Fallback for missing images to prevent crash, though verification passed
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual fallback if no transform is provided
            image = cv2.resize(image, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            # HWC to CHW and normalize to 0-1
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # 3. Process Label
        # Encode the InChI string to a tensor of token indices
        inchi_text = row["InChI"]
        label_tensor = self.tokenizer.encode(inchi_text)

        return image, label_tensor
