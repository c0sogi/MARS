import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_class_to_idx(df):
    """
    Generates a mapping from hotel_id to class index (0 to N-1).
    This ensures that hotel IDs are mapped to contiguous integers for the model.

    Args:
        df (pd.DataFrame): DataFrame containing 'hotel_id' column.

    Returns:
        dict: Mapping from hotel_id to index.
    """
    unique_ids = sorted(df["hotel_id"].unique())
    return {hotel_id: idx for idx, hotel_id in enumerate(unique_ids)}


def get_transforms(mode="train", img_size=Config.img_size, crop_size=Config.crop_size):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for deterministic preprocessing.
        img_size (int): The size to resize the input image to.
        crop_size (int): The size to crop the image to (input to model).

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.RandomCrop(crop_size, crop_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=crop_size // 8,
                    max_width=crop_size // 8,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.mean, std=Config.std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Deterministic Center Crop
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.CenterCrop(crop_size, crop_size),
                A.Normalize(mean=Config.mean, std=Config.std),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Handles loading images, converting to RGB, applying transforms, and mapping labels.
    """

    def __init__(
        self,
        df,
        transform=None,
        data_root=Config.input_dir,
        mode="train",
        class_to_idx=None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (image, hotel_id, file_path).
            transform (albumentations.Compose): Transformations to apply.
            data_root (str): Root directory for input data.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict, optional): Mapping from hotel_id to class index. Required for 'train'/'val'.
        """
        self.df = df
        self.transform = transform
        self.data_root = data_root
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Validation checks
        if self.mode in ["train", "val"]:
            if "hotel_id" not in self.df.columns:
                raise ValueError(
                    f"DataFrame must contain 'hotel_id' column for mode '{self.mode}'."
                )
            if self.class_to_idx is None:
                # In a real pipeline, we might generate it here, but it's better to pass it in
                # to share the mapping between train and val sets.
                pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]

        # Construct full image path
        full_path = os.path.join(self.data_root, file_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)

        if image is None:
            # Fallback for missing images (should be caught by metadata validation)
            # Create a black image to prevent crashing
            image = np.zeros((Config.img_size, Config.img_size, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations/transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return data based on mode
        if self.mode in ["train", "val"]:
            hotel_id = row["hotel_id"]

            # Map hotel_id to class index
            if self.class_to_idx:
                label = self.class_to_idx.get(hotel_id, -1)
                if label == -1:
                    raise ValueError(
                        f"Hotel ID {hotel_id} not found in provided class_to_idx mapping."
                    )
            else:
                # Fallback if no mapping provided (e.g. raw ID)
                label = hotel_id

            return image, torch.tensor(label, dtype=torch.long)

        elif self.mode == "test":
            # For test, we need the image ID to map predictions back to the submission file
            return image, row["image"]

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
