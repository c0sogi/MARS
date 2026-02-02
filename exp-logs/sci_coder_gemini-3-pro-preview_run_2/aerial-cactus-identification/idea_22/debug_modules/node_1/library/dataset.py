import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library import config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for Cactus Classification.
    Holds images in memory as numpy arrays for efficiency.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of binary labels with shape (N,).
            ids (np.ndarray, optional): Array of image IDs (filenames).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C)
        img = self.images[idx]

        # Apply transformations
        # ToTensor converts numpy (H, W, C) in [0, 255] -> tensor (C, H, W) in [0.0, 1.0]
        if self.transform:
            img = self.transform(img)

        # Retrieve label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)  # Placeholder for test

        # Retrieve ID
        img_id = self.ids[idx] if self.ids is not None else ""

        return img, label, img_id


def get_transforms(split):
    """
    Returns the transformation pipeline for a given data split.

    Args:
        split (str): One of 'train', 'val', 'test'.

    Returns:
        torchvision.transforms.Compose: The transform pipeline.
    """
    transform_list = [
        transforms.ToTensor(),  # Converts [0, 255] -> [0.0, 1.0] and HWC -> CHW
    ]

    if split == "train":
        # Augmentations strictly as per Idea 22
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

    # No resizing or normalization with mean/std is requested in the idea description,
    # just basic scaling to [0, 1] provided by ToTensor.

    return transforms.Compose(transform_list)


def load_data_split(metadata_path, split_name, load_cached_data=True):
    """
    Loads data for a specific split, utilizing caching to speed up subsequent runs.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        split_name (str): Name of the split (e.g., 'train', 'val', 'test') for cache naming.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Ensure cache directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define cache file paths
    img_cache_path = os.path.join(config.WORKING_DIR, f"{split_name}_images.npy")
    lbl_cache_path = os.path.join(config.WORKING_DIR, f"{split_name}_labels.npy")
    ids_cache_path = os.path.join(config.WORKING_DIR, f"{split_name}_ids.npy")

    # Attempt to load from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(ids_cache_path):
            print(f"[{split_name}] Loading data from cache...")
            images = np.load(img_cache_path)
            ids = np.load(ids_cache_path, allow_pickle=True)

            labels = None
            if os.path.exists(lbl_cache_path):
                labels = np.load(lbl_cache_path)

            return images, labels, ids
        else:
            print(f"[{split_name}] Cache not found. Processing from scratch...")
    else:
        print(f"[{split_name}] Ignoring cache. Processing from scratch...")

    # Process from scratch
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    lbl_list = []
    id_list = []

    # Pre-allocate if possible or just append (dataset is small enough for append)
    for _, row in df.iterrows():
        # Resolve full path
        rel_path = row["file_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            print(f"Warning: Could not read image {full_path}. Skipping.")
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        id_list.append(row["id"])

        # Handle labels
        # For test set, 'has_cactus' might be a placeholder (0.5), we can load it
        # but usually we ignore it downstream.
        if "has_cactus" in row:
            lbl_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)

    if lbl_list:
        labels = np.array(lbl_list, dtype=np.float32)
    else:
        labels = None

    # Save to cache
    print(f"[{split_name}] Saving processed data to cache...")
    np.save(img_cache_path, images)
    np.save(ids_cache_path, ids)
    if labels is not None:
        np.save(lbl_cache_path, labels)

    return images, labels, ids


def get_dataloaders(
    batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the dataloaders.
        num_workers (int): Number of subprocesses for data loading.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """

    # 1. Load Data
    train_imgs, train_lbls, train_ids = load_data_split(
        config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = load_data_split(
        config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = load_data_split(
        config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # 2. Create Datasets
    train_dataset = CactusDataset(
        images=train_imgs,
        labels=train_lbls,
        ids=train_ids,
        transform=get_transforms("train"),
    )

    val_dataset = CactusDataset(
        images=val_imgs, labels=val_lbls, ids=val_ids, transform=get_transforms("val")
    )

    # For test, we explicitly ignore labels to avoid confusion,
    # though the loader might return the placeholder values if we passed test_lbls.
    test_dataset = CactusDataset(
        images=test_imgs, labels=None, ids=test_ids, transform=get_transforms("test")
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    print(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}
