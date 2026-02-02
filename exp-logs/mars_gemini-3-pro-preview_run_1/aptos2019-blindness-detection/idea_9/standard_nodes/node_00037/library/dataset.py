import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Standard ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # "Squashing" preprocessing: Resize directly to target size disregarding aspect ratio
    height = Config.image_size
    width = Config.image_size

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width, always_apply=True),
                # Strict Geometric Invariance: Flips and Rotations
                A.HorizontalFlip(p=Config.aug_prob),
                A.VerticalFlip(p=Config.aug_prob),
                A.RandomRotate90(p=Config.aug_prob),
                # No Photometric (Hue/Sat) augmentations to preserve pathological color markers
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic resizing
        return A.Compose(
            [
                A.Resize(height=height, width=width, always_apply=True),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class RetinaDataset(Dataset):
    """
    Dataset class for Diabetic Retinopathy images.
    Implements caching of resized images to RAM/Disk to speed up training.
    """

    def __init__(self, csv_path, mode, load_cached_data=True, transform=None):
        """
        Args:
            csv_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to try loading data from cache.
            transform (A.Compose): Albumentations transforms.
        """
        self.mode = mode
        self.transform = transform
        self.df = pd.read_csv(csv_path)

        # Handle Subsetting for Debugging
        if Config.debug:
            # Use a small fixed subset if debug is True globally
            subset_size = 100
            if len(self.df) > subset_size:
                self.df = self.df.iloc[:subset_size].reset_index(drop=True)
        elif mode == "train" and Config.train_subset_size is not None:
            self.df = self.df.iloc[: Config.train_subset_size].reset_index(drop=True)
        elif mode == "val" and Config.val_subset_size is not None:
            self.df = self.df.iloc[: Config.val_subset_size].reset_index(drop=True)

        # Define Cache Path
        # Differentiate cache names based on debug status to avoid pollution
        suffix = "_debug" if Config.debug else ""
        self.cache_path = os.path.join(
            Config.working_dir, f"cached_images_{mode}{suffix}.npy"
        )

        # Load or Generate Data
        self.images = self._load_data(load_cached_data)

        # Extract labels if available
        if "diagnosis" in self.df.columns:
            self.labels = self.df["diagnosis"].values
        else:
            self.labels = None

    def _load_data(self, load_cached_data):
        """
        Loads images from cache if available, otherwise reads from disk, resizes, and caches.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            try:
                # print(f"Loading cached {self.mode} data from {self.cache_path}...")
                data = np.load(self.cache_path)
                if len(data) == len(self.df):
                    return data
                else:
                    pass
                    # print("Cache size mismatch. Regenerating...")
            except Exception as e:
                pass
                # print(f"Failed to load cache: {e}. Regenerating...")

        # 2. Generate from scratch
        # print(f"Processing {len(self.df)} images for {self.mode} dataset...")
        images_list = []

        for idx, row in self.df.iterrows():
            # Metadata file_path is relative to input_dir (e.g., "train_images/id.png")
            full_path = os.path.join(Config.input_dir, row["file_path"])

            # Read image
            img = cv2.imread(full_path)
            if img is None:
                # Fallback for missing/corrupt images: create black image
                img = np.zeros(
                    (Config.image_size, Config.image_size, 3), dtype=np.uint8
                )
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Pre-resize to save RAM/Disk space (Squashing)
            # We resize here for caching efficiency.
            # Augmentations will run on these pre-resized images.
            img_resized = cv2.resize(img, (Config.image_size, Config.image_size))
            images_list.append(img_resized)

        data = np.array(images_list, dtype=np.uint8)

        # 3. Save to cache
        try:
            np.save(self.cache_path, data)
            # print(f"Saved cache to {self.cache_path}")
        except Exception as e:
            pass
            # print(f"Warning: Could not save cache: {e}")

        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve image from RAM
        image = self.images[idx]

        # Apply transforms
        if self.transform:
            # Albumentations expects kwargs
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Handle Labels
        if self.labels is not None:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)
        else:
            # For test set without labels, return -1 or similar dummy
            return image, torch.tensor(-1, dtype=torch.long)


def create_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")
    test_transform = get_transforms(mode="test")  # Same as val

    # Datasets
    train_dataset = RetinaDataset(
        csv_path=Config.train_csv,
        mode="train",
        load_cached_data=load_cached_data,
        transform=train_transform,
    )

    val_dataset = RetinaDataset(
        csv_path=Config.val_csv,
        mode="val",
        load_cached_data=load_cached_data,
        transform=val_transform,
    )

    test_dataset = RetinaDataset(
        csv_path=Config.test_csv,
        mode="test",
        load_cached_data=load_cached_data,
        transform=test_transform,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
