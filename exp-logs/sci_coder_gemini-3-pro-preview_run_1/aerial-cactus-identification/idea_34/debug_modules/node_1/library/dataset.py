import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Classification.
    Handles in-memory floating point tensors and applies on-the-fly geometric augmentations.
    """

    def __init__(self, images, labels=None, augment=False):
        """
        Args:
            images (np.ndarray): Array of shape (N, C, H, W) containing float32 images [0, 1].
            labels (np.ndarray, optional): Array of shape (N,) containing targets.
            augment (bool): Whether to apply random geometric flips.
        """
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy to tensor
        img = torch.from_numpy(self.images[idx])

        # Apply geometric augmentations if requested (Training)
        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, [2])  # Flip width dimension
            # Random Vertical Flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, [1])  # Flip height dimension

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        else:
            return img


def load_subset(
    metadata_path,
    cache_img_path,
    cache_label_path=None,
    cache_id_path=None,
    load_cached=True,
):
    """
    Loads a data subset. Tries to load from .npy cache first; otherwise processes from scratch.

    Args:
        metadata_path (str): Path to the CSV metadata file.
        cache_img_path (str): Path to save/load image cache.
        cache_label_path (str, optional): Path to save/load label cache.
        cache_id_path (str, optional): Path to save/load ID cache.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # 1. Try Loading from Cache
    if load_cached:
        files_exist = os.path.exists(cache_img_path)
        if cache_label_path:
            files_exist = files_exist and os.path.exists(cache_label_path)
        if cache_id_path:
            files_exist = files_exist and os.path.exists(cache_id_path)

        if files_exist:
            print(f"Loading cached data from {os.path.dirname(cache_img_path)}...")
            images = np.load(cache_img_path)
            labels = np.load(cache_label_path) if cache_label_path else None
            ids = np.load(cache_id_path, allow_pickle=True) if cache_id_path else None
            return images, labels, ids

    # 2. Process from Scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    # Iterate through metadata
    for _, row in df.iterrows():
        rel_path = row["file_path"]
        img_id = row["id"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read Image
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize [0, 255] -> [0, 1] and Transpose (H, W, C) -> (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        img_list.append(img)
        id_list.append(img_id)

        # Collect label if required
        if cache_label_path and "has_cactus" in row:
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list)
    labels = np.array(label_list, dtype=np.float32) if cache_label_path else None

    # 3. Save to Cache
    print(f"Saving processed data to {os.path.dirname(cache_img_path)}...")
    os.makedirs(os.path.dirname(cache_img_path), exist_ok=True)
    np.save(cache_img_path, images)
    if cache_label_path:
        np.save(cache_label_path, labels)
    if cache_id_path:
        np.save(cache_id_path, ids)

    return images, labels, ids


def get_loaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
):
    """
    Prepares and returns DataLoaders for Train, Validation, and Test sets.

    Returns:
        train_loader, val_loader, test_loader, test_ids
    """

    # Define local cache paths for Validation set (not explicitly in Config)
    cache_val_imgs = os.path.join(Config.WORK_DIR, "cache_val_imgs.npy")
    cache_val_labels = os.path.join(Config.WORK_DIR, "cache_val_labels.npy")

    # 1. Load Data Subsets
    # Train
    train_imgs, train_labels, _ = load_subset(
        Config.TRAIN_METADATA_PATH,
        Config.CACHE_TRAIN_IMGS,
        Config.CACHE_TRAIN_LABELS,
        load_cached=Config.LOAD_CACHED_DATA,
    )

    # Validation
    val_imgs, val_labels, _ = load_subset(
        Config.VAL_METADATA_PATH,
        cache_val_imgs,
        cache_val_labels,
        load_cached=Config.LOAD_CACHED_DATA,
    )

    # Test (No labels needed for cache, but we need IDs)
    test_imgs, _, test_ids = load_subset(
        Config.TEST_METADATA_PATH,
        Config.CACHE_TEST_IMGS,
        cache_id_path=Config.CACHE_TEST_IDS,
        load_cached=Config.LOAD_CACHED_DATA,
    )

    # 2. Handle Debug Mode (Slice Data)
    if debug:
        print(f"DEBUG Mode: Slicing datasets to {Config.DEBUG_SUBSET_SIZE} samples.")
        train_imgs = train_imgs[: Config.DEBUG_SUBSET_SIZE]
        train_labels = train_labels[: Config.DEBUG_SUBSET_SIZE]
        val_imgs = val_imgs[: Config.DEBUG_SUBSET_SIZE]
        val_labels = val_labels[: Config.DEBUG_SUBSET_SIZE]
        test_imgs = test_imgs[: Config.DEBUG_SUBSET_SIZE]
        test_ids = test_ids[: Config.DEBUG_SUBSET_SIZE]

    # 3. Instantiate Datasets
    # Train: Augment = True
    train_dataset = CactusDataset(train_imgs, train_labels, augment=True)
    # Val/Test: Augment = False
    val_dataset = CactusDataset(val_imgs, val_labels, augment=False)
    test_dataset = CactusDataset(test_imgs, labels=None, augment=False)

    # 4. Create DataLoaders
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

    print(
        f"Data Loaded Successfully: Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})"
    )

    return train_loader, val_loader, test_loader, test_ids
