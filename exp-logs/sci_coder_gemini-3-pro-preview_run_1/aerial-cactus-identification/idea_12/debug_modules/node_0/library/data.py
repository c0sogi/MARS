import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config


class CactusDataset(Dataset):
    """
    Custom Dataset for Cactus Classification.
    Holds pre-processed images and metadata in RAM.
    """

    def __init__(self, images, metadata, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, C, H, W), float32 normalized images.
            metadata (np.ndarray): Shape (N, 1), float32 normalized file sizes.
            labels (np.ndarray, optional): Shape (N,), integer labels.
            transform (callable, optional): Augmentations to apply.
        """
        self.images = torch.from_numpy(images).float()
        self.metadata = torch.from_numpy(metadata).float()
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        meta = self.metadata[idx]

        # Apply augmentations if provided (e.g., flips)
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, meta, label
        else:
            return img, meta


def _load_and_process_image(path):
    """
    Loads an image, converts to RGB, normalizes, and permutes to (C, H, W).
    """
    full_path = os.path.join(Config.INPUT_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Load BGR
    img = cv2.imread(full_path)
    if img is None:
        raise ValueError(f"Failed to load image: {full_path}")

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Normalize with dataset statistics
    mean = np.array(Config.NORM_MEAN, dtype=np.float32)
    std = np.array(Config.NORM_STD, dtype=np.float32)
    img = (img - mean) / std

    # HWC to CHW
    img = img.transpose(2, 0, 1)
    return img


def _get_file_size(path):
    """Returns file size in bytes."""
    full_path = os.path.join(Config.INPUT_DIR, path)
    return os.path.getsize(full_path)


def prepare_subset(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from metadata CSV, processes images/filesizes, and caches to disk.
    Returns numpy arrays: images, file_sizes, labels (if available), ids.
    """
    # Define cache file paths
    cache_imgs_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_imgs.npy")
    cache_meta_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_filesizes.npy")
    cache_lbls_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    cache_ids_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    files_exist = (
        os.path.exists(cache_imgs_path)
        and os.path.exists(cache_meta_path)
        and os.path.exists(cache_ids_path)
    )
    # Labels might not exist for test set, so we check conditionally later or handle gracefully

    if load_cached_data and files_exist:
        # print(f"Loading cached data for {cache_prefix}...")
        imgs = np.load(cache_imgs_path)
        meta = np.load(cache_meta_path)
        ids = np.load(cache_ids_path)

        lbls = None
        if os.path.exists(cache_lbls_path):
            lbls = np.load(cache_lbls_path)

        return imgs, meta, lbls, ids

    # Process from scratch
    # print(f"Processing data for {cache_prefix}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    meta_list = []
    lbl_list = []
    id_list = []

    for _, row in df.iterrows():
        # Load Image
        img_tensor = _load_and_process_image(row["file_path"])
        img_list.append(img_tensor)

        # Load File Size
        fsize = _get_file_size(row["file_path"])
        meta_list.append(fsize)

        # ID
        id_list.append(row["id"])

        # Label (if exists and not placeholder)
        # Test set has 0.5 placeholder, we treat it as no label or handle separately.
        # But for consistency, we only save labels if it's train/val.
        # The prompt implies test set labels must be predicted.
        if "has_cactus" in row and row["has_cactus"] in [0, 1]:
            lbl_list.append(row["has_cactus"])

    # Convert to numpy
    imgs = np.array(img_list, dtype=np.float32)
    meta = np.array(meta_list, dtype=np.float32).reshape(-1, 1)  # (N, 1)
    ids = np.array(id_list)

    # Save cache
    np.save(cache_imgs_path, imgs)
    np.save(cache_meta_path, meta)
    np.save(cache_ids_path, ids)

    lbls = None
    if len(lbl_list) > 0:
        lbls = np.array(lbl_list, dtype=np.int64)
        np.save(cache_lbls_path, lbls)

    return imgs, meta, lbls, ids


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and dataloaders for Train, Val, and Test.
    Handles file size normalization using Train statistics.
    """

    # 1. Load/Process Data
    # Train
    train_imgs, train_sizes, train_lbls, _ = prepare_subset(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    # Val
    val_imgs, val_sizes, val_lbls, _ = prepare_subset(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    # Test
    test_imgs, test_sizes, _, test_ids = prepare_subset(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # 2. Normalize File Sizes (Metadata)
    # Compute stats on TRAIN only to avoid leakage
    size_mean = np.mean(train_sizes)
    size_std = np.std(train_sizes) + 1e-8  # epsilon for stability

    train_sizes_norm = (train_sizes - size_mean) / size_std
    val_sizes_norm = (val_sizes - size_mean) / size_std
    test_sizes_norm = (test_sizes - size_mean) / size_std

    # 3. Handle Debug Mode
    if Config.DEBUG:
        limit = min(len(train_imgs), Config.DEBUG_SAMPLES)
        train_imgs = train_imgs[:limit]
        train_sizes_norm = train_sizes_norm[:limit]
        train_lbls = train_lbls[:limit]

        limit_val = min(len(val_imgs), Config.DEBUG_SAMPLES)
        val_imgs = val_imgs[:limit_val]
        val_sizes_norm = val_sizes_norm[:limit_val]
        val_lbls = val_lbls[:limit_val]

        limit_test = min(len(test_imgs), Config.DEBUG_SAMPLES)
        test_imgs = test_imgs[:limit_test]
        test_sizes_norm = test_sizes_norm[:limit_test]
        test_ids = test_ids[:limit_test]

    # 4. Define Transforms
    # Since images are already tensors (C, H, W) and normalized,
    # we use transforms that operate on tensors.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # No TTA transforms here; TTA is handled in the inference loop or by wrapping the dataset
    # For standard validation/testing, no transforms are needed (images are pre-normalized).
    eval_transform = None

    # 5. Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_sizes_norm, train_lbls, transform=train_transform
    )
    val_dataset = CactusDataset(
        val_imgs, val_sizes_norm, val_lbls, transform=eval_transform
    )
    test_dataset = CactusDataset(
        test_imgs, test_sizes_norm, labels=None, transform=eval_transform
    )

    # 6. Create Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup/BatchNorm stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ids
