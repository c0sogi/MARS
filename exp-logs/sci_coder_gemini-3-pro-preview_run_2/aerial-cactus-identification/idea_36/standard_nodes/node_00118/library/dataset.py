import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def _load_cached_dataset(metadata_path, cache_prefix, load_cached_data=True):
    """
    Loads dataset arrays (images, labels, ids) with a caching mechanism.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        cache_prefix (str): Prefix for the cached filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids) as numpy arrays.
    """
    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{cache_prefix}_images.npy")
    lbl_cache_path = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(cache_dir, f"{cache_prefix}_ids.npy")

    # 1. IF load_cached_data is True: Try to load
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(lbl_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            try:
                images = np.load(img_cache_path)
                labels = np.load(lbl_cache_path)
                ids = np.load(ids_cache_path)
                return images, labels, ids
            except Exception as e:
                print(f"Failed to load cache for {cache_prefix}: {e}. Recomputing...")
                pass

    # 2. IF loading fails OR load_cached_data is False: Compute
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    images_list = []
    labels_list = []
    ids_list = []

    for _, row in df.iterrows():
        # Construct full path. Metadata file_path is relative to input dir.
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Load image
        img = cv2.imread(full_path)
        if img is None:
            # In a real scenario, we might log this. For now, skip or error.
            # Given metadata verification passed, this shouldn't happen.
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images_list.append(img)
        labels_list.append(row["has_cactus"])
        ids_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=np.float32)
    ids = np.array(ids_list)

    # Save to cache
    try:
        np.save(img_cache_path, images)
        np.save(lbl_cache_path, labels)
        np.save(ids_cache_path, ids)
    except Exception as e:
        print(f"Warning: Failed to save cache for {cache_prefix}: {e}")

    return images, labels, ids


class CactusDataset(Dataset):
    def __init__(self, images, labels, ids, augment=False):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            ids (np.ndarray): Array of IDs (N,).
            augment (bool): Whether to apply training augmentations.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image (H, W, C)
        img = self.images[idx]
        label = self.labels[idx]
        img_id = self.ids[idx]

        # Apply Augmentations (Train only)
        if self.augment:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                img = np.fliplr(img)
            # Random Vertical Flip
            if np.random.rand() < 0.5:
                img = np.flipud(img)

        # Normalize [0, 255] -> [0, 1]
        # This creates a copy and ensures positive strides
        img = img.astype(np.float32) / 255.0

        # Transpose (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        # Convert to Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor, img_id


def seed_worker(worker_id):
    """
    Ensures reproducibility in DataLoader workers by seeding NumPy.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load raw data arrays (from cache or compute)
    train_imgs, train_lbls, train_ids = _load_cached_dataset(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_imgs, val_lbls, val_ids = _load_cached_dataset(
        Config.VAL_METADATA_PATH, "val", load_cached_data
    )
    test_imgs, test_lbls, test_ids = _load_cached_dataset(
        Config.TEST_METADATA_PATH, "test", load_cached_data
    )

    # Instantiate Datasets
    # Train: Augmentation = True
    train_dataset = CactusDataset(train_imgs, train_lbls, train_ids, augment=True)
    # Val/Test: Augmentation = False
    val_dataset = CactusDataset(val_imgs, val_lbls, val_ids, augment=False)
    test_dataset = CactusDataset(test_imgs, test_lbls, test_ids, augment=False)

    # Generator for reproducibility
    g = torch.Generator()
    g.manual_seed(0)  # Seed for the generator used by DataLoader

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True if Config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader
