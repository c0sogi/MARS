import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.utils import get_atom_vector


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformation pipeline for a specific mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                # RandomBrightnessContrast or other augmentations could be added here
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test transforms (Deterministic)
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )


class ChemicalDataset(Dataset):
    """
    PyTorch Dataset for Chemical Structure Recognition.

    Handles loading of images and generation of stoichiometry targets.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata (must have 'file_path').
                               For train/val, must have 'InChI'.
                               For test, must have 'image_id'.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose, optional): Albumentations transforms.
                                             If None, default transforms are used.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform if transform is not None else get_transforms(mode)
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata file_path is relative to input_dir (e.g., "train/0/0/0/id.png")
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(file_path)
        if image is None:
            # Fallback for corrupt/missing images: return black image
            # This prevents the dataloader from crashing
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            inchi = row["InChI"]
            # Generate atom count target on-the-fly using utility function
            # This vector is the regression target for the model
            target_vector = get_atom_vector(inchi)
            return image, torch.tensor(target_vector, dtype=torch.float32)

        elif self.mode == "test":
            image_id = row["image_id"]
            return image, image_id

        else:
            # Fallback
            return image
