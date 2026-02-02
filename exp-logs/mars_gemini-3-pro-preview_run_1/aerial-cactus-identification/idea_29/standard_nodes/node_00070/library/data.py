import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from library.config import Config

# Computed statistics from Data Analysis
# RGB Mean and Std scaled to [0, 1]
# Mean: R=128.37, G=115.25, B=119.40
# Std : R=38.60, G=35.68, B=39.15
DATASET_MEAN = [0.5034, 0.4520, 0.4683]
DATASET_STD = [0.1514, 0.1399, 0.1535]


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Identification.
    Holds data in RAM as FloatTensors to minimize I/O.
    """

    def __init__(
        self,
        images,
        labels,
        file_sizes,
        phase="train",
        size_stats=None,
    ):
        """
        Args:
            images (torch.Tensor): Tensor of shape (N, C, H, W) in [0, 1].
            labels (torch.Tensor): Tensor of shape (N,).
            file_sizes (torch.Tensor): Tensor of shape (N,) containing raw file sizes in bytes.
            phase (str): 'train', 'val', or 'test'. Controls augmentation.
            size_stats (dict, optional): Dict with 'min' and 'max' for log-size normalization.
        """
        self.images = images
        self.labels = labels
        self.file_sizes = file_sizes
        self.phase = phase
        self.size_stats = size_stats

        # Pre-calculate log sizes
        # We use log1p to handle potential zero sizes (though unlikely) and smooth distribution
        self.log_sizes = torch.log1p(self.file_sizes.float())

        # Normalize log sizes if stats provided
        if self.size_stats:
            min_val = self.size_stats["min"]
            max_val = self.size_stats["max"]
            # Avoid division by zero
            denom = max_val - min_val if max_val > min_val else 1.0
            self.norm_log_sizes = (self.log_sizes - min_val) / denom
        else:
            self.norm_log_sizes = self.log_sizes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data from RAM
        img = self.images[idx]  # (C, H, W)
        label = self.labels[idx]
        log_size = self.norm_log_sizes[idx]

        # Augmentations
        if self.phase == "train":
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                img = TF.hflip(img)
            # Random Vertical Flip
            if torch.rand(1) < 0.5:
                img = TF.vflip(img)

        # Normalization
        # Normalize using dataset statistics
        img = TF.normalize(img, mean=DATASET_MEAN, std=DATASET_STD)

        return {
            "image": img,
            "label": label,
            "log_size": log_size,
            "raw_idx": idx,  # Useful for debugging or tracking
        }


def load_and_cache_data(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from disk, caches it as .npy files, and returns tensors.
    Strictly follows the caching logic requirement.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache file paths
    cache_imgs_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_imgs.npy")
    cache_labels_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    cache_fsizes_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_fsizes.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    data_exists = (
        os.path.exists(cache_imgs_path)
        and os.path.exists(cache_labels_path)
        and os.path.exists(cache_fsizes_path)
    )

    if load_cached_data and data_exists:
        print(f"Loading cached data from {Config.CACHE_DIR} ({cache_prefix})...")
        imgs_np = np.load(cache_imgs_path)
        labels_np = np.load(cache_labels_path)
        fsizes_np = np.load(cache_fsizes_path)
        # IDs are optional for training but good to have
        if os.path.exists(cache_ids_path):
            ids_np = np.load(cache_ids_path)
    else:
        print(f"Processing data from {metadata_path}...")
        df = pd.read_csv(metadata_path)

        if Config.DEBUG:
            print(f"DEBUG MODE: Subsampling {cache_prefix} data...")
            df = df.head(100)

        img_list = []
        label_list = []
        fsize_list = []
        id_list = []

        for _, row in df.iterrows():
            # Construct full path
            # metadata 'file_path' is relative to input dir (e.g. 'train/xxx.jpg')
            full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            if not os.path.exists(full_path):
                continue

            # Read Image
            img = cv2.imread(full_path)
            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get file size
            fsize = os.path.getsize(full_path)

            img_list.append(img)
            label_list.append(row["has_cactus"])
            fsize_list.append(fsize)
            id_list.append(row["id"])

        imgs_np = np.array(img_list, dtype=np.uint8)
        labels_np = np.array(label_list, dtype=np.float32)
        fsizes_np = np.array(fsize_list, dtype=np.float32)
        ids_np = np.array(id_list)

        # Save to cache
        print(f"Saving cache to {Config.CACHE_DIR} ({cache_prefix})...")
        np.save(cache_imgs_path, imgs_np)
        np.save(cache_labels_path, labels_np)
        np.save(cache_fsizes_path, fsizes_np)
        np.save(cache_ids_path, ids_np)

    # Convert to Tensors and move to RAM
    # Images: (N, H, W, C) -> (N, C, H, W) and normalize to [0, 1]
    print(f"Converting {cache_prefix} data to FloatTensors in RAM...")

    # Permute to (N, C, H, W)
    imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0
    labels_tensor = torch.from_numpy(labels_np).float()
    fsizes_tensor = torch.from_numpy(fsizes_np).float()

    return imgs_tensor, labels_tensor, fsizes_tensor, ids_np


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    Calculates normalization stats for file sizes from the training set.
    """
    # 1. Load Data
    train_imgs, train_labels, train_fsizes, _ = load_and_cache_data(
        Config.TRAIN_META_PATH, "train", load_cached_data
    )
    val_imgs, val_labels, val_fsizes, _ = load_and_cache_data(
        Config.VAL_META_PATH, "val", load_cached_data
    )

    # 2. Compute Normalization Stats for Auxiliary Task (Log File Size)
    # We compute this on training data only to avoid leakage
    train_log_sizes = torch.log1p(train_fsizes)
    min_log_size = train_log_sizes.min().item()
    max_log_size = train_log_sizes.max().item()

    size_stats = {"min": min_log_size, "max": max_log_size}
    print(
        f"Auxiliary Task Stats (Log Size) - Min: {min_log_size:.4f}, Max: {max_log_size:.4f}"
    )

    # 3. Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_labels, train_fsizes, phase="train", size_stats=size_stats
    )

    val_dataset = CactusDataset(
        val_imgs, val_labels, val_fsizes, phase="val", size_stats=size_stats
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, size_stats


def get_test_dataloader(size_stats, load_cached_data=True):
    """
    Creates DataLoader for the test set.
    Requires size_stats from the training set for consistent normalization.
    """
    test_imgs, test_labels, test_fsizes, test_ids = load_and_cache_data(
        Config.TEST_META_PATH, "test", load_cached_data
    )

    test_dataset = CactusDataset(
        test_imgs, test_labels, test_fsizes, phase="test", size_stats=size_stats
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return test_loader, test_ids
