import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config


def get_transforms(image_size, mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode and resolution.

    Args:
        image_size (int): The target resolution (e.g., 256, 384).
        mode (str): 'train' for augmentation, 'val'/'test' for deterministic transforms.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(image_size * 0.1),
                    max_width=int(image_size * 0.1),
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def create_id_map(csv_path):
    """
    Creates a mapping from Whale ID to Integer Label.
    Strictly excludes 'new_whale' to support the known-only training strategy.

    Args:
        csv_path (str): Path to the training metadata CSV.

    Returns:
        dict: Mapping {whale_id: int_label}
    """
    df = pd.read_csv(csv_path)
    # Exclude new_whale
    known_ids = df[df["Id"] != "new_whale"]["Id"].unique()
    # Sort for determinism
    known_ids = sorted(known_ids)
    return {label: idx for idx, label in enumerate(known_ids)}


class WhaleDataset(Dataset):
    def __init__(
        self,
        csv_path,
        subset_name,
        image_size,
        id_map=None,
        mode="train",
        filter_new_whale=False,
        load_cached_data=True,
    ):
        """
        Args:
            csv_path (str): Path to metadata CSV.
            subset_name (str): 'train', 'val', or 'test'. Used for cache naming.
            image_size (int): Target image resolution.
            id_map (dict, optional): Mapping from ID string to int. Required for train/val.
            mode (str): 'train', 'val', or 'test'. Controls transforms and return values.
            filter_new_whale (bool): If True, removes 'new_whale' samples (for training).
            load_cached_data (bool): If True, attempts to load pre-processed .npy files.
        """
        self.csv_path = csv_path
        self.subset_name = subset_name
        self.image_size = image_size
        self.id_map = id_map
        self.mode = mode
        self.filter_new_whale = filter_new_whale

        # Load Metadata
        self.df = pd.read_csv(self.csv_path)

        # Load Images (with Caching Logic)
        self.images = self._load_images(load_cached_data)

        # Apply Filtering (e.g., remove new_whale)
        if self.filter_new_whale:
            self._apply_filter()

        # Initialize Transforms
        self.transforms = get_transforms(self.image_size, self.mode)

    def _load_images(self, load_cached_data):
        """
        Handles loading images from cache or processing them from scratch.
        """
        cache_path = Config.get_cache_path(self.subset_name, self.image_size)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # print(f"Loading cached images from {cache_path}...")
                images = np.load(cache_path)
                if len(images) == len(self.df):
                    return images
                else:
                    # print("Cache size mismatch. Recomputing...")
                    pass
            except Exception as e:
                # print(f"Error loading cache: {e}. Recomputing...")
                pass

        # 2. Compute from scratch
        # print(f"Processing {len(self.df)} images for {self.subset_name} at {self.image_size}x{self.image_size}...")

        # Pre-allocate array
        num_samples = len(self.df)
        images = np.zeros(
            (num_samples, self.image_size, self.image_size, 3), dtype=np.uint8
        )

        for idx, row in self.df.iterrows():
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Read image
            img = cv2.imread(file_path)

            if img is None:
                # Handle missing/corrupt files by using a black image
                # (Should not happen based on EDA, but good for robustness)
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize
                img = cv2.resize(img, (self.image_size, self.image_size))

            images[idx] = img

        # 3. Save to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, images)
            # print(f"Saved cache to {cache_path}")
        except Exception as e:
            # print(f"Failed to save cache: {e}")
            pass

        return images

    def _apply_filter(self):
        """
        Filters the dataframe and image array to exclude 'new_whale'.
        """
        mask = self.df["Id"] != "new_whale"

        # Filter DataFrame
        self.df = self.df[mask].reset_index(drop=True)

        # Filter Images
        self.images = self.images[mask]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image
        img = self.images[idx]

        # Apply Augmentations/Transforms
        augmented = self.transforms(image=img)
        img_tensor = augmented["image"]

        row = self.df.iloc[idx]

        if self.mode == "test":
            # For test, return image and filename (for submission)
            return img_tensor, row["Image"]
        else:
            # For train/val, return image and label index
            label_str = row["Id"]

            # If we are in val mode and encounter a label not in map (e.g. new_whale if not filtered),
            # we need to handle it. However, based on strategy, we filter new_whale.
            # If something unexpected slips through, we default to -1 or error.
            if self.id_map and label_str in self.id_map:
                label_idx = self.id_map[label_str]
            else:
                # Fallback for safety, though shouldn't happen with correct filtering
                label_idx = -1

            return img_tensor, torch.tensor(label_idx, dtype=torch.long)
