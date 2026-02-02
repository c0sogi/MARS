import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Identification.
    """

    def __init__(self, images, labels, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            transform (A.Compose): Albumentations transformations.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return float tensor for label to match BCEWithLogitsLoss expectation
        return image, torch.tensor(label, dtype=torch.float32)


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV and images. Implements caching using .npy files.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images_np, labels_np)
    """
    # Define cache paths
    img_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_imgs.npy")
    label_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try to load from cache
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(label_cache_path)
    ):
        try:
            # print(f"Loading {cache_prefix} data from cache...")
            images = np.load(img_cache_path)
            labels = np.load(label_cache_path)
            return images, labels
        except Exception as e:
            print(
                f"Failed to load cache for {cache_prefix}: {e}. Reloading from source."
            )

    # 2. Load from source
    # print(f"Processing {cache_prefix} data from source...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []

    # Pre-allocate if possible or just append list (list append is fast enough for 17k items)
    for _, row in df.iterrows():
        # Metadata file_path is relative to input dir (e.g., "train/id.jpg")
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # Fallback or error. Given dataset quality, we assume existence.
            # Create a black image if missing to prevent crash, or raise error.
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

        # Get label
        label_list.append(row["has_cactus"])

    images = np.array(img_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.float32)

    # 3. Save to cache
    try:
        np.save(img_cache_path, images)
        np.save(label_cache_path, labels)
        # print(f"Saved {cache_prefix} data to cache.")
    except Exception as e:
        print(f"Warning: Could not save cache for {cache_prefix}: {e}")

    return images, labels


def get_transforms(split):
    """
    Returns Albumentations transforms for the specified split.
    """
    if split == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(
                    mean=Config.NORM_MEAN,
                    std=Config.NORM_STD,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Normalize(
                    mean=Config.NORM_MEAN,
                    std=Config.NORM_STD,
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(Config.SEED)

    # 1. Load Data
    train_imgs, train_labels = load_and_cache_data(
        Config.TRAIN_METADATA, "cache_train", load_cached_data
    )
    val_imgs, val_labels = load_and_cache_data(
        Config.VAL_METADATA, "cache_val", load_cached_data
    )
    test_imgs, test_labels = load_and_cache_data(
        Config.TEST_METADATA, "cache_test", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_labels, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(val_imgs, val_labels, transform=get_transforms("val"))

    test_dataset = CactusDataset(
        test_imgs, test_labels, transform=get_transforms("test")
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup stability if batch size is small
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
