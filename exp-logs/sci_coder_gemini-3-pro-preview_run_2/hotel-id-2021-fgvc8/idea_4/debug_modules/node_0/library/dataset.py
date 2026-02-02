import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import CFG
from library.utils import seed_everything


def get_transforms(mode="train", image_size=CFG.image_size):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
        image_size (int): Target image size.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=image_size // 10,
                    max_width=image_size // 10,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


def process_metadata(load_cached_data=True):
    """
    Loads metadata CSVs, generates or loads label encodings, and maps labels to integers.
    Implements caching using .npy files for label mappings.

    Args:
        load_cached_data (bool): If True, attempts to load class mappings from cache.

    Returns:
        tuple: (train_df, val_df, test_df, hotel_classes, chain_classes)
            - *_df: Pandas DataFrames with 'hotel_id_idx' and 'chain_idx' columns (for train/val).
            - hotel_classes: Numpy array of unique hotel IDs (original values).
            - chain_classes: Numpy array of unique chain IDs (original values).
    """
    # Ensure working directory exists
    os.makedirs(CFG.working_dir, exist_ok=True)

    hotel_classes_path = os.path.join(CFG.working_dir, "hotel_classes.npy")
    chain_classes_path = os.path.join(CFG.working_dir, "chain_classes.npy")

    # Load raw metadata
    train_df = pd.read_csv(CFG.train_metadata_path)
    val_df = pd.read_csv(CFG.val_metadata_path)
    test_df = pd.read_csv(CFG.test_metadata_path)

    # Determine if we can load from cache
    cache_exists = os.path.exists(hotel_classes_path) and os.path.exists(
        chain_classes_path
    )

    if load_cached_data and cache_exists:
        print("Loading cached label encodings...")
        hotel_classes = np.load(hotel_classes_path)
        chain_classes = np.load(chain_classes_path)
    else:
        print("Computing label encodings from training data...")
        # Get unique classes from training data
        # We sort them to ensure deterministic mapping
        hotel_classes = np.sort(train_df["hotel_id"].unique())
        chain_classes = np.sort(train_df["chain"].unique())

        # Save to cache
        np.save(hotel_classes_path, hotel_classes)
        np.save(chain_classes_path, chain_classes)
        print(f"Saved label encodings to {CFG.working_dir}")

    # Create mapping dictionaries
    hotel_to_idx = {hotel: idx for idx, hotel in enumerate(hotel_classes)}
    chain_to_idx = {chain: idx for idx, chain in enumerate(chain_classes)}

    # Apply mappings to Train and Validation DataFrames
    # We use .map().fillna(-1) to handle potential unseen classes safely,
    # though strict splitting should prevent this for hotel_id.

    # Train
    train_df["hotel_id_idx"] = train_df["hotel_id"].map(hotel_to_idx).astype(int)
    train_df["chain_idx"] = train_df["chain"].map(chain_to_idx).astype(int)

    # Val
    val_df["hotel_id_idx"] = val_df["hotel_id"].map(hotel_to_idx)
    val_df["chain_idx"] = val_df["chain"].map(chain_to_idx)

    # Check for NaNs in validation (unseen classes)
    if val_df["hotel_id_idx"].isnull().any():
        print(
            f"Warning: {val_df['hotel_id_idx'].isnull().sum()} images in validation set have unseen hotel IDs."
        )
        val_df = val_df.dropna(subset=["hotel_id_idx"])
        val_df["hotel_id_idx"] = val_df["hotel_id_idx"].astype(int)
    else:
        val_df["hotel_id_idx"] = val_df["hotel_id_idx"].astype(int)

    if val_df["chain_idx"].isnull().any():
        print(
            f"Warning: {val_df['chain_idx'].isnull().sum()} images in validation set have unseen chain IDs."
        )
        val_df["chain_idx"] = val_df["chain_idx"].fillna(-1).astype(int)
    else:
        val_df["chain_idx"] = val_df["chain_idx"].astype(int)

    # Test DataFrame does not have targets, so no mapping needed for columns.

    return train_df, val_df, test_df, hotel_classes, chain_classes


class HotelDataset(Dataset):
    """
    PyTorch Dataset for Hotel Identification.
    Reads images via OpenCV, applies Albumentations, and returns tensors.
    """

    def __init__(self, df, transform=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata.
            transform (A.Compose): Albumentations transform pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.transform = transform
        self.mode = mode

        # Pre-extract paths to list for faster access
        self.file_paths = df["file_path"].values

        if self.mode != "test":
            self.hotel_labels = df["hotel_id_idx"].values
            self.chain_labels = df["chain_idx"].values
        else:
            self.image_names = df["image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct full path
        rel_path = self.file_paths[idx]
        full_path = os.path.join(CFG.input_dir, rel_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Fallback for missing images (though metadata validation should prevent this)
            # Create a black image of expected size
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback if no transform provided (should not happen in pipeline)
            t = ToTensorV2()
            image = t(image=image)["image"]

        if self.mode == "test":
            # For test, return image and image name (for submission)
            image_name = self.image_names[idx]
            return image, image_name
        else:
            # For train/val, return image, hotel_label, chain_label
            hotel_label = torch.tensor(self.hotel_labels[idx], dtype=torch.long)
            chain_label = torch.tensor(self.chain_labels[idx], dtype=torch.long)
            return image, hotel_label, chain_label
