import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(phase: str):
    """
    Returns the composition of augmentations for a given phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The albumentations transform pipeline.
    """
    # Normalization parameters to map [0, 255] -> [0, 1]
    # Albumentations Normalize subtracts mean and divides by std.
    # (x - 0) / 1 with max_pixel_value=255.0 results in x / 255.0
    normalization = A.Normalize(
        mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0
    )

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                normalization,
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Pure tensor conversion and normalization
        return A.Compose([normalization, ToTensorV2()])


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus Identification task.
    Wraps pre-loaded numpy arrays of images and labels.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            labels (np.ndarray, optional): Array of binary labels with shape (N,).
            ids (np.ndarray or list, optional): List/Array of image IDs (filenames).
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image
        image = self.images[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Construct return tuple
        result = [image]

        if self.labels is not None:
            # Convert label to float tensor for BCE loss
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            result.append(label)

        if self.ids is not None:
            result.append(self.ids[idx])

        # Unpack if single item (though rarely the case here given structure)
        if len(result) == 1:
            return result[0]

        return tuple(result)


def load_data(split: str, load_cached_data: bool = True):
    """
    Loads data for a specific split, handling caching to .npy files.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, labels, ids)
            images: np.ndarray (N, 32, 32, 3) uint8
            labels: np.ndarray (N,) float32 or None
            ids: np.ndarray (N,) string
    """
    # Ensure cache directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define paths based on split
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
        imgs_cache_path = os.path.join(cache_dir, "train_images.npy")
        lbls_cache_path = os.path.join(cache_dir, "train_labels.npy")
    elif split == "val":
        meta_path = Config.VAL_METADATA_PATH
        imgs_cache_path = os.path.join(cache_dir, "val_images.npy")
        lbls_cache_path = os.path.join(cache_dir, "val_labels.npy")
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
        imgs_cache_path = os.path.join(cache_dir, "test_images.npy")
        lbls_cache_path = None  # Test set has no labels
    else:
        raise ValueError(f"Invalid split: {split}")

    # Load Metadata to get IDs and file paths
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    ids = df["id"].values

    # Attempt to load from cache
    if load_cached_data:
        imgs_exist = os.path.exists(imgs_cache_path)
        lbls_exist = (lbls_cache_path is None) or os.path.exists(lbls_cache_path)

        if imgs_exist and lbls_exist:
            print(
                f"[{split.upper()}] Loading images and labels from cache: {cache_dir}"
            )
            images = np.load(imgs_cache_path)
            labels = np.load(lbls_cache_path) if lbls_cache_path else None
            return images, labels, ids

    # Process from scratch
    print(f"[{split.upper()}] Processing images from scratch...")

    num_samples = len(df)
    # Pre-allocate array for efficiency
    images = np.zeros((num_samples, 32, 32, 3), dtype=np.uint8)

    # Load labels if available
    labels = None
    if "has_cactus" in df.columns and split != "test":
        labels = df["has_cactus"].values.astype(np.float32)

    # Iterate and load images
    for i, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Load image in BGR
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {full_path}")

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Assign to array
        images[i] = img

    # Save to cache
    print(f"[{split.upper()}] Saving processed data to cache...")
    np.save(imgs_cache_path, images)
    if labels is not None and lbls_cache_path is not None:
        np.save(lbls_cache_path, labels)

    return images, labels, ids


def get_dataloaders(load_cached_data: bool = True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    train_imgs, train_lbls, train_ids = load_data("train", load_cached_data)
    val_imgs, val_lbls, val_ids = load_data("val", load_cached_data)
    test_imgs, test_lbls, test_ids = load_data("test", load_cached_data)

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

    test_dataset = CactusDataset(
        images=test_imgs,
        labels=None,  # No labels for test inference
        ids=test_ids,
        transform=get_transforms("test"),
    )

    # 3. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
