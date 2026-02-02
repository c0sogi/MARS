import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import Config
from library.utils import get_logger, FileSizeScaler

# Initialize logger
logger = get_logger("dataset")


def get_transforms(phase="train"):
    """
    Returns the data transformations for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    # Normalization constants from Config
    mean = Config.NORM_MEAN
    std = Config.NORM_STD

    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts HWC [0,255] to CHW [0.0,1.0]
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )


def load_data_to_ram(metadata_path, prefix, load_cached_data=True):
    """
    Loads images, labels, and file sizes into RAM, with caching.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, file_sizes, ids)
            images: np.ndarray of shape (N, H, W, 3)
            labels: np.ndarray of shape (N,)
            file_sizes: np.ndarray of shape (N,)
            ids: np.ndarray of shape (N,)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    path_imgs = os.path.join(cache_dir, f"{prefix}_imgs.npy")
    path_labels = os.path.join(cache_dir, f"{prefix}_labels.npy")
    path_fsizes = os.path.join(cache_dir, f"{prefix}_fsizes.npy")
    path_ids = os.path.join(cache_dir, f"{prefix}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_imgs)
        and os.path.exists(path_labels)
        and os.path.exists(path_fsizes)
        and os.path.exists(path_ids)
    )

    if load_cached_data and cache_exists:
        logger.info(f"Loading {prefix} data from cache...")
        try:
            images = np.load(path_imgs)
            labels = np.load(path_labels)
            file_sizes = np.load(path_fsizes)
            ids = np.load(path_ids)
            return images, labels, file_sizes, ids
        except Exception as e:
            logger.warning(f"Failed to load cache for {prefix}: {e}. Recomputing...")

    # Compute from scratch
    logger.info(f"Processing {prefix} data from disk...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.empty((n_samples, Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(n_samples, dtype=np.float32)
    file_sizes = np.zeros(n_samples, dtype=np.float32)
    ids = np.empty(n_samples, dtype=object)

    for i, row in df.iterrows():
        # Construct full path
        # metadata file_path is relative to input dir (e.g., train/id.jpg)
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read ID and Label
        ids[i] = row["id"]
        labels[i] = row["has_cactus"]

        # Read File Size
        if os.path.exists(full_path):
            file_sizes[i] = os.path.getsize(full_path)

            # Read Image
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[i] = img
            else:
                # Fallback for corrupt image (should be caught by metadata validation)
                logger.warning(f"Could not read image: {full_path}")
                images[i] = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
                )
        else:
            logger.warning(f"File not found: {full_path}")
            images[i] = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

    # Save to cache
    logger.info(f"Saving {prefix} data to cache...")
    np.save(path_imgs, images)
    np.save(path_labels, labels)
    np.save(path_fsizes, file_sizes)
    np.save(path_ids, ids)

    return images, labels, file_sizes, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, file_sizes, transform=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            file_sizes (np.ndarray): Array of raw file sizes in bytes (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        # Pre-calculate normalized file sizes for regression target
        self.norm_file_sizes = FileSizeScaler.transform(file_sizes)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        quality_target = self.norm_file_sizes[idx]

        if self.transform:
            image = self.transform(image)

        # Return dictionary as expected by the training loop
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "quality_target": torch.tensor(quality_target, dtype=torch.float32),
        }


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Paths to metadata
    train_meta = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    val_meta = os.path.join(Config.METADATA_DIR, "val_metadata.csv")

    # Load data
    train_imgs, train_lbls, train_fsizes, _ = load_data_to_ram(
        train_meta, "train", load_cached_data
    )
    val_imgs, val_lbls, val_fsizes, _ = load_data_to_ram(
        val_meta, "val", load_cached_data
    )

    # Handle DEBUG mode
    if Config.DEBUG:
        logger.info("DEBUG mode enabled: Truncating datasets.")
        limit = 500
        train_imgs = train_imgs[:limit]
        train_lbls = train_lbls[:limit]
        train_fsizes = train_fsizes[:limit]
        val_imgs = val_imgs[:limit]
        val_lbls = val_lbls[:limit]
        val_fsizes = val_fsizes[:limit]

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_fsizes, transform=get_transforms("train")
    )

    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_fsizes, transform=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Creates DataLoader for the test set.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (test_loader, test_ids)
    """
    test_meta = os.path.join(Config.METADATA_DIR, "test_metadata.csv")

    test_imgs, test_lbls, test_fsizes, test_ids = load_data_to_ram(
        test_meta, "test", load_cached_data
    )

    # Handle DEBUG mode
    if Config.DEBUG:
        limit = 100
        test_imgs = test_imgs[:limit]
        test_lbls = test_lbls[:limit]
        test_fsizes = test_fsizes[:limit]
        test_ids = test_ids[:limit]

    test_dataset = CactusDataset(
        test_imgs,
        test_lbls,  # These are placeholders (0.5)
        test_fsizes,
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_ids
