import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed

# Constants
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_39"


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Images.
    """

    def __init__(self, images, targets, ids, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            targets (np.ndarray): Array of labels/targets with shape (N,).
            ids (np.ndarray): Array of image IDs (filenames).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.targets = targets
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and target
        image = self.images[idx]  # Shape: (H, W, C), dtype: uint8
        target = self.targets[idx]  # Scalar float

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, target, self.ids[idx]


def load_and_cache_data(metadata_file, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata/images, caches it to disk as .npy, or loads from cache.

    Args:
        metadata_file (str): Name of the metadata CSV file.
        cache_prefix (str): Prefix for the cached files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, targets, ids)
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache paths
    img_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    target_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_targets.npy")
    id_cache_path = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check if cache exists and is requested
    if (
        load_cached_data
        and os.path.exists(img_cache_path)
        and os.path.exists(target_cache_path)
        and os.path.exists(id_cache_path)
    ):
        images = np.load(img_cache_path)
        targets = np.load(target_cache_path)
        ids = np.load(id_cache_path)
    else:
        # Load metadata
        meta_path = os.path.join(METADATA_DIR, metadata_file)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df = pd.read_csv(meta_path)

        # Pre-allocate lists
        image_list = []
        target_list = []
        id_list = []

        for _, row in df.iterrows():
            # Construct full image path
            rel_path = row["file_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Load image
            img = cv2.imread(full_path)
            if img is None:
                # Skip missing images if any (though metadata check passed)
                continue

            # Convert BGR (OpenCV default) to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_list.append(img)
            target_list.append(row["has_cactus"])
            id_list.append(row["id"])

        # Convert to numpy arrays
        images = np.array(image_list, dtype=np.uint8)
        targets = np.array(target_list, dtype=np.float32)
        ids = np.array(id_list)

        # Save to cache
        np.save(img_cache_path, images)
        np.save(target_cache_path, targets)
        np.save(id_cache_path, ids)

    return images, targets, ids


def get_transforms(split="train"):
    """
    Returns the transformation pipeline for the specified split.
    Strictly adheres to 'light augmentation' (flips) and normalization.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts [0, 255] uint8 -> [0.0, 1.0] float32, (H,W,C)->(C,H,W)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        # Validation and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True, seed=42):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for loaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached .npy files.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    set_seed(seed)

    # 1. Load Data (with caching)
    train_imgs, train_targets, train_ids = load_and_cache_data(
        "train_metadata.csv", "train", load_cached_data
    )
    val_imgs, val_targets, val_ids = load_and_cache_data(
        "val_metadata.csv", "val", load_cached_data
    )
    test_imgs, test_targets, test_ids = load_and_cache_data(
        "test_metadata.csv", "test", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_targets, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_targets, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, test_targets, test_ids, transform=get_transforms("test")
    )

    # 3. Create DataLoaders
    # Pin memory speeds up host-to-device transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
