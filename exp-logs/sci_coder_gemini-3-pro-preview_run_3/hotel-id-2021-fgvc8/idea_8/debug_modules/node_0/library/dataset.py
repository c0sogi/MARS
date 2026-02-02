import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from library.config import Config


def get_label_encoder(metadata_path, cache_dir, load_cached_data=True):
    """
    Creates or loads a LabelEncoder for hotel_ids.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        cache_dir (str): Directory to store the cached encoder classes.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        sklearn.preprocessing.LabelEncoder: Fitted label encoder.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "label_encoder_classes.npy")

    encoder = LabelEncoder()

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading label encoder from {cache_file}...")
        try:
            classes = np.load(cache_file, allow_pickle=True)
            encoder.classes_ = classes
            return encoder
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Fitting label encoder from metadata...")
    df = pd.read_csv(metadata_path)
    # Ensure hotel_id is treated consistently (e.g., as integer or string)
    # The raw data has integers, but we'll fit on whatever is in the column.
    unique_ids = df["hotel_id"].unique()
    encoder.fit(unique_ids)

    # Save to cache
    np.save(cache_file, encoder.classes_)
    print(f"Label encoder saved to {cache_file}. Classes: {len(encoder.classes_)}")

    return encoder


def get_transforms(image_size, mode="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        image_size (int): The target input size (e.g., 224).
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # We resize to slightly larger than target, then crop.
    # 256 is standard for 224 input.
    resize_target = int(image_size / 0.875)

    if mode == "train":
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_target),
                A.RandomCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.SmallestMaxSize(max_size=resize_target),
                A.CenterCrop(height=image_size, width=image_size),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class HotelDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        label_encoder=None,
        transform=None,
        is_test=False,
        debug=False,
    ):
        """
        Args:
            csv_file (str): Path to the metadata CSV.
            root_dir (str): Root directory containing image folders (e.g., ./input).
            label_encoder (LabelEncoder, optional): Fitted encoder for mapping hotel_ids.
            transform (A.Compose, optional): Albumentations transforms.
            is_test (bool): If True, returns (image, filename). If False, returns (image, label).
            debug (bool): If True, limits dataset size for debugging.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.label_encoder = label_encoder

        if debug:
            self.df = self.df.sample(
                n=min(len(self.df), Config.debug_sample_size), random_state=Config.seed
            ).reset_index(drop=True)

        # Pre-compute paths to avoid overhead in __getitem__
        # The 'file_path' column in metadata is relative to input dir (e.g., "train_images/0/xyz.jpg")
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            # Encode labels
            if self.label_encoder is None:
                raise ValueError(
                    "LabelEncoder must be provided for training/validation sets."
                )

            # Ensure hotel_id column exists
            if "hotel_id" not in self.df.columns:
                raise ValueError(f"Column 'hotel_id' not found in {csv_file}")

            self.labels = self.label_encoder.transform(self.df["hotel_id"].values)
        else:
            # For test set, we need the image filename (ID) for submission
            self.image_ids = self.df["image"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, rel_path)

        # Load image using OpenCV
        image = cv2.imread(full_path)

        if image is None:
            # Handle missing images gracefully (though metadata check should prevent this)
            # Return a black image or raise error. Raising error is safer to detect pipeline issues.
            raise FileNotFoundError(f"Image not found at {full_path}")

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = ToTensorV2()(image=image)["image"]

        if self.is_test:
            # Return image and filename for submission mapping
            return image, self.image_ids[idx]
        else:
            # Return image and integer label for training
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
