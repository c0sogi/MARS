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
    PyTorch Dataset for the Cactus classification task.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of labels with shape (N,).
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
        # Load image (H, W, C)
        image = self.images[idx]

        # Apply transforms
        # ToTensor() converts numpy (H, W, C) [0, 255] -> tensor (C, H, W) [0.0, 1.0]
        if self.transform:
            image = self.transform(image)

        # Prepare return tuple
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            if self.ids is not None:
                return image, label, self.ids[idx]
            return image, label
        else:
            # Test set case
            if self.ids is not None:
                return image, self.ids[idx]
            return image


def get_transforms(split):
    """
    Returns the appropriate transforms for the given split.

    Args:
        split (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=Config.AUG_HFLIP_PROB),
                transforms.RandomVerticalFlip(p=Config.AUG_VFLIP_PROB),
            ]
        )
    else:
        # Val and Test
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


def _load_or_create_cache(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes from scratch and saves to cache.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(img_cache_path)
        and os.path.exists(lbl_cache_path)
        and os.path.exists(ids_cache_path)
    )

    if load_cached_data and cache_exists:
        print(f"Loading {cache_prefix} data from cache...")
        images = np.load(img_cache_path)
        labels = np.load(lbl_cache_path)
        ids = np.load(ids_cache_path)
        return images, labels, ids

    print(f"Processing {cache_prefix} data from scratch...")

    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    # Pre-allocate arrays
    num_samples = len(df)
    images = np.zeros(
        (num_samples, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
    )
    labels = np.zeros(num_samples, dtype=np.float32)
    ids = np.empty(num_samples, dtype=object)

    for i, row in df.iterrows():
        # Construct full path
        # Metadata contains relative path from input dir
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            raise ValueError(f"Failed to load image: {full_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Store in array
        images[i] = img
        labels[i] = row["has_cactus"]
        ids[i] = row["id"]

    # Save to cache
    print(f"Saving {cache_prefix} data to cache...")
    np.save(img_cache_path, images)
    np.save(lbl_cache_path, labels)
    np.save(ids_cache_path, ids)

    return images, labels, ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    train_imgs, train_lbls, train_ids = _load_or_create_cache(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = _load_or_create_cache(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = _load_or_create_cache(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # 2. Handle Debug Subset
    if Config.DEBUG_SUBSET_SIZE is not None:
        subset_size = min(Config.DEBUG_SUBSET_SIZE, len(train_imgs))
        train_imgs = train_imgs[:subset_size]
        train_lbls = train_lbls[:subset_size]
        train_ids = train_ids[:subset_size]

        val_imgs = val_imgs[:subset_size]
        val_lbls = val_lbls[:subset_size]
        val_ids = val_ids[:subset_size]

        test_imgs = test_imgs[:subset_size]
        test_ids = test_ids[:subset_size]
        # test_lbls are placeholders, slice them too to keep consistency
        test_lbls = test_lbls[:subset_size]

        print(f"DEBUG MODE: Reduced dataset to {subset_size} samples.")

    # 3. Create Datasets
    train_dataset = CactusDataset(
        images=train_imgs,
        labels=train_lbls,
        ids=None,  # IDs not needed for training loop
        transform=get_transforms("train"),
    )

    val_dataset = CactusDataset(
        images=val_imgs,
        labels=val_lbls,
        ids=None,  # IDs not needed for val loop usually, but can be added if needed
        transform=get_transforms("val"),
    )

    test_dataset = CactusDataset(
        images=test_imgs,
        labels=None,  # No labels for test
        ids=test_ids,  # IDs needed for submission
        transform=get_transforms("test"),
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

    return train_loader, val_loader, test_loader
