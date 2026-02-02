import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import seed_everything


def get_transforms(phase: str):
    """
    Returns the Albumentations transformation pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    if phase == "train":
        return A.Compose(
            [
                # Isotropic Invariance: Continuous rotation on the full 96x96 context
                A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_REFLECT_101),
                # Intensity augmentation
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # Contextual Crop: Crop to 64x64 to remove rotation artifacts and focus on ROI
                A.CenterCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                # Normalization
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic Center Crop and Normalize
        return A.Compose(
            [
                A.CenterCrop(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def load_dataset_arrays(
    metadata_path, cache_prefix, load_cached_data=True, debug=False
):
    """
    Loads dataset images and labels into numpy arrays. Uses caching to speed up subsequent loads.

    Args:
        metadata_path (str): Path to the metadata CSV.
        cache_prefix (str): Prefix for the cache files (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, loads a small subset of data.

    Returns:
        tuple: (images, labels, ids)
            images: np.ndarray of shape (N, H, W, C)
            labels: np.ndarray of shape (N,)
            ids: np.ndarray of shape (N,)
    """
    # Define cache paths
    images_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Attempt to load from cache
    if load_cached_data and not debug:
        if (
            os.path.exists(images_cache_path)
            and os.path.exists(labels_cache_path)
            and os.path.exists(ids_cache_path)
        ):
            print(f"Loading {cache_prefix} data from cache...")
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            ids = np.load(ids_cache_path, allow_pickle=True)
            return images, labels, ids
        else:
            print(f"Cache not found for {cache_prefix}. Processing from scratch...")

    # Load metadata
    df = pd.read_csv(metadata_path)

    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"Debug mode: Loaded {len(df)} samples for {cache_prefix}.")

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.zeros(
        (n_samples, Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE, 3), dtype=np.uint8
    )
    labels = np.zeros(n_samples, dtype=np.int64)
    ids = np.array(df["id"].values)

    # Iterate and load images
    print(f"Loading images for {cache_prefix}...")
    valid_indices = []

    for idx, row in df.iterrows():
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            # In case of missing file, skip (though metadata check passed)
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        images[idx] = img
        labels[idx] = row["label"]
        valid_indices.append(idx)

    # Filter valid loads (in case of read errors)
    if len(valid_indices) != n_samples:
        images = images[valid_indices]
        labels = labels[valid_indices]
        ids = ids[valid_indices]

    # Save to cache (only if not in debug mode)
    if not debug:
        print(f"Saving {cache_prefix} data to cache...")
        np.save(images_cache_path, images)
        np.save(labels_cache_path, labels)
        np.save(ids_cache_path, ids)

    return images, labels, ids


class PathologyDataset(Dataset):
    def __init__(self, images, labels, ids, transforms=None):
        """
        Args:
            images (np.ndarray): Array of images (N, H, W, C).
            labels (np.ndarray): Array of labels (N,).
            ids (np.ndarray): Array of IDs (N,).
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return {
            "image": image,
            "label": torch.tensor(
                label, dtype=torch.float32
            ),  # BCEWithLogitsLoss expects float
            "id": self.ids[idx],
        }


def get_loaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # --- Load Data ---
    # Train
    train_images, train_labels, train_ids = load_dataset_arrays(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # Validation
    val_images, val_labels, val_ids = load_dataset_arrays(
        Config.VAL_METADATA_PATH,
        "val",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # Test
    test_images, test_labels, test_ids = load_dataset_arrays(
        Config.TEST_METADATA_PATH,
        "test",
        load_cached_data=load_cached_data,
        debug=Config.DEBUG,
    )

    # --- Create Datasets ---
    train_dataset = PathologyDataset(
        train_images, train_labels, train_ids, transforms=get_transforms("train")
    )

    val_dataset = PathologyDataset(
        val_images, val_labels, val_ids, transforms=get_transforms("val")
    )

    test_dataset = PathologyDataset(
        test_images, test_labels, test_ids, transforms=get_transforms("test")
    )

    # --- Create Loaders ---
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

    return train_loader, val_loader, test_loader
