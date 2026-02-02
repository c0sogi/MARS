import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_label_mapping(df, cache_dir, load_cached_data=True):
    """
    Generates or loads the mapping between hotel_ids and class indices.

    Args:
        df (pd.DataFrame): DataFrame containing the 'hotel_id' column.
        cache_dir (str): Directory to save/load the label encoder.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (class_to_idx, idx_to_class)
            class_to_idx (dict): Mapping from hotel_id to integer index.
            idx_to_class (np.ndarray): Array where index i corresponds to hotel_id.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "label_encoder.npy")

    if load_cached_data and os.path.exists(cache_path):
        unique_classes = np.load(cache_path, allow_pickle=True)
    else:
        # Ensure we sort the classes for deterministic mapping
        unique_classes = np.sort(df["hotel_id"].unique())
        np.save(cache_path, unique_classes)

    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    return class_to_idx, unique_classes


def get_transforms(image_size, mode="train"):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        image_size (tuple): Target (height, width).
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transformation pipeline.
    """
    height, width = image_size

    if mode == "train":
        # Mild Augmentation: Resize slightly larger, Random Crop, Horizontal Flip
        # This adds spatial variance without aggressive distortion
        resize_h = int(height * 1.1)
        resize_w = int(width * 1.1)

        return A.Compose(
            [
                A.Resize(resize_h, resize_w),
                A.RandomCrop(height, width),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test: Deterministic Resize
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    def __init__(
        self, df, image_root_dir, transform=None, mode="train", class_to_idx=None
    ):
        """
        Dataset for Hotel ID Recognition.

        Args:
            df (pd.DataFrame): Metadata DataFrame.
            image_root_dir (str): Root directory for images.
            transform (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
            class_to_idx (dict): Mapping from hotel_id to class index (required for train/val).
        """
        self.df = df.reset_index(drop=True)
        self.image_root_dir = image_root_dir
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Validation
        if self.mode in ["train", "val"] and self.class_to_idx is None:
            raise ValueError("class_to_idx must be provided for train/val modes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full file path
        # Metadata 'file_path' is relative to input_dir
        rel_path = row["file_path"]
        full_path = os.path.join(self.image_root_dir, rel_path)

        # Load Image
        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode in ["train", "val"]:
            hotel_id = row["hotel_id"]
            label = self.class_to_idx[hotel_id]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # Test mode: return image and image ID (filename) for submission mapping
            return image, str(row["image"])
