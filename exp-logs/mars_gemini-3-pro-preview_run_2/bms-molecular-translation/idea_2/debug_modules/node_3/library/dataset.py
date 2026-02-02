import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.tokenizer import Tokenizer


def get_transforms(config: Config, mode: str = "train") -> A.Compose:
    """
    Returns the Albumentations transformations for the dataset.

    Args:
        config (Config): Configuration object containing image parameters.
        mode (str): Mode of operation ('train', 'val', 'test').

    Returns:
        A.Compose: Composed albumentations transforms.
    """
    if mode == "train":
        # For training, we can add augmentations here if needed.
        # Currently keeping it simple as per baseline requirements.
        return A.Compose(
            [
                A.Resize(height=config.image_size, width=config.image_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # For validation and testing, just resize and normalize.
        return A.Compose(
            [
                A.Resize(height=config.image_size, width=config.image_size),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )


class InChiDataset(Dataset):
    """
    PyTorch Dataset for InChI chemical structure recognition.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        config: Config,
        transform: A.Compose = None,
        mode: str = "train",
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing image paths and labels.
            tokenizer (Tokenizer): Instance of the Tokenizer class.
            config (Config): Configuration object.
            transform (albumentations.Compose, optional): Transformations to apply.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.transform = (
            transform if transform is not None else get_transforms(config, mode)
        )
        self.mode = mode

        # Pre-calculate full paths to avoid doing it in __getitem__ repeatedly
        # The 'file_path' in metadata is relative to the input directory
        self.file_paths = [
            os.path.join(config.input_dir, fp) for fp in df["file_path"].values
        ]

        # If labels exist, prepare them
        self.labels = None
        if "InChI" in df.columns:
            self.labels = df["InChI"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # 1. Load Image
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        if image is None:
            # Handle missing image gracefully
            # Return a black image of correct size to prevent crashing
            image = np.zeros(
                (self.config.image_size, self.config.image_size, 3), dtype=np.uint8
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 3. Get Label or ID
        if self.labels is not None:
            # Training/Validation mode: Return image and tokenized label
            text = self.labels[idx]
            label_tensor = self.tokenizer.text_to_sequence(text)
            return image, label_tensor
        else:
            # Test mode: Return image and image_id for submission
            image_id = self.df.iloc[idx]["image_id"]
            return image, image_id
