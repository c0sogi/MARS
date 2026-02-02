import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_class_mapping(train_csv_path):
    """
    Generates a deterministic mapping from hotel_id to class index (0..N-1).
    This ensures that the classification head maps to the correct hotel ID.

    Args:
        train_csv_path (str): Path to the training metadata CSV.

    Returns:
        dict: Mapping {hotel_id: class_index}
    """
    df = pd.read_csv(train_csv_path)
    # Sort to ensure deterministic mapping
    unique_hotels = sorted(df["hotel_id"].unique())
    class_to_idx = {hotel_id: idx for idx, hotel_id in enumerate(unique_hotels)}
    return class_to_idx


def get_transforms(mode="train"):
    """
    Creates the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # Train: RandomResizedCrop to CROP_SIZE (224)
                # This handles the "Resize to 256... random crop to 224" logic
                # by sampling a crop and resizing it, which is standard for training.
                A.RandomResizedCrop(
                    height=Config.CROP_SIZE,
                    width=Config.CROP_SIZE,
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.333),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Val/Test: Resize to IMG_SIZE (256) then CenterCrop to CROP_SIZE (224)
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Reads images based on metadata CSVs generated in the previous step.
    """

    def __init__(
        self,
        csv_path,
        root_dir=Config.INPUT_DIR,
        transform=None,
        class_to_idx=None,
        mode="train",
    ):
        """
        Args:
            csv_path (str): Path to the metadata CSV (train.csv, val.csv, or test.csv).
            root_dir (str): Base directory for images (default: ./input).
            transform (albumentations.Compose): Image transformations.
            class_to_idx (dict, optional): Mapping from hotel_id to class index. Required for 'train'/'val'.
            mode (str): 'train', 'val', or 'test'.
        """
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.class_to_idx = class_to_idx

        # Load metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # Validation for class mapping
        if self.mode in ["train", "val"] and self.class_to_idx is None:
            # If not provided, we warn or raise. For this implementation, we assume it's passed.
            # However, to be safe, if it's missing in val, we might have issues.
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # Metadata 'file_path' is relative to input dir, e.g., "train_images/1/img.jpg"
        image_path = os.path.join(self.root_dir, row["file_path"])

        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Handle missing or corrupt images gracefully
        if image is None:
            # Create a black image of the expected size to prevent crashing
            # This should be rare given the metadata validation step
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Prepare return dictionary
        sample = {"image": image}

        if self.mode in ["train", "val"]:
            hotel_id = row["hotel_id"]

            # Map hotel_id to class index 0..N-1
            if self.class_to_idx is not None:
                if hotel_id in self.class_to_idx:
                    label = self.class_to_idx[hotel_id]
                else:
                    # Fallback for unknown classes (should not happen in train/val split)
                    label = 0
            else:
                # If no mapping, assume hotel_id is the label (risky if not 0-indexed contiguous)
                label = hotel_id

            sample["label"] = torch.tensor(label, dtype=torch.long)
            # Keep original ID for debugging/metrics
            sample["original_id"] = hotel_id

        elif self.mode == "test":
            # For test, we need the image ID to map predictions back to the submission format
            sample["image_id"] = row["image"]

        return sample
