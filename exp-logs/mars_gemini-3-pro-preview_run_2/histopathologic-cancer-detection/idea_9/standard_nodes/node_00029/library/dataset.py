import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


class InMemoryDataset(Dataset):
    """
    A dataset that stores all images in memory as a NumPy array to eliminate I/O bottlenecks.
    Applies Albumentations transforms on-the-fly.
    """

    def __init__(self, images, labels, ids, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are stored as uint8 (H, W, C)
        image = self.images[idx]
        label = self.labels[idx]
        image_id = self.ids[idx]

        if self.transform:
            # Albumentations expects image in 'image' key
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return image, label, and ID (ID is needed for submission)
        # Label needs to be tensor for training
        return image, torch.tensor(label, dtype=torch.float32), image_id


def get_transforms(phase):
    """
    Defines the Isotropic Augmentation Pipeline.
    """
    if phase == "train":
        return A.Compose(
            [
                # Geometric Completeness: Flips and Continuous Rotation
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate -180 to 180. Border mode constant (0) is safe because we center crop afterwards.
                A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
                # Intensity: ColorJitter on the transformed patch
                # Applied to 100% of images to force stain invariance (Cite solution_lesson_node_00028)
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
                ),
                # Contextual Crop: 64x64px from center
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                # Normalization
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Deterministic Center Crop
        return A.Compose(
            [
                A.CenterCrop(height=Config.CROP_SIZE, width=Config.CROP_SIZE),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def load_dataset_arrays(
    metadata_path, cache_prefix, load_cached_data=True, debug_size=None
):
    """
    Loads dataset into memory with caching mechanism.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    images_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    labels_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")
    ids_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(images_cache_path)
            and os.path.exists(labels_cache_path)
            and os.path.exists(ids_cache_path)
        ):

            print(f"Loading {cache_prefix} data from cache...")
            images = np.load(images_cache_path)
            labels = np.load(labels_cache_path)
            ids = np.load(ids_cache_path)

            # Handle debug slicing on cached data if requested
            if debug_size is not None and len(images) > debug_size:
                print(f"Slicing cached data to {debug_size} samples for debugging.")
                return images[:debug_size], labels[:debug_size], ids[:debug_size]

            return images, labels, ids

    # 2. Compute from scratch
    print(f"Processing {cache_prefix} data from scratch...")
    df = pd.read_csv(metadata_path)

    if debug_size is not None:
        df = df.head(debug_size)
        print(f"Debug mode: Processing only {len(df)} samples.")

    img_list = []
    label_list = []
    id_list = []

    # Determine input directory based on prefix or metadata path context
    # Metadata paths are relative e.g. "train/xxx.tif" or "test/xxx.tif"
    # We join with Config.INPUT_DIR

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            # In case of missing file (though verified), skip or error.
            # We skip to avoid crashing, but print warning if needed.
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        label_list.append(row["label"])
        id_list.append(row["id"])

    images = np.array(img_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.int64)  # Labels are integers 0/1
    ids = np.array(id_list)

    # Save to cache (only if not debugging, to avoid overwriting full cache with partial)
    if debug_size is None:
        print(f"Saving {cache_prefix} data to cache...")
        np.save(images_cache_path, images)
        np.save(labels_cache_path, labels)
        np.save(ids_cache_path, ids)

    return images, labels, ids


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    # Define metadata paths
    train_meta = os.path.join(Config.METADATA_DIR, "train.csv")
    val_meta = os.path.join(Config.METADATA_DIR, "val.csv")
    test_meta = os.path.join(Config.METADATA_DIR, "test.csv")

    # Load Data
    # Note: We use different cache prefixes to distinguish sets
    # If debug is on, we append _debug to cache prefix to avoid corrupting main cache
    suffix = "_debug" if debug_size is not None else ""

    # Train
    train_imgs, train_lbls, train_ids = load_dataset_arrays(
        train_meta, f"train{suffix}", load_cached_data, debug_size
    )

    # Val
    val_imgs, val_lbls, val_ids = load_dataset_arrays(
        val_meta, f"val{suffix}", load_cached_data, debug_size
    )

    # Test
    test_imgs, test_lbls, test_ids = load_dataset_arrays(
        test_meta, f"test{suffix}", load_cached_data, debug_size
    )

    # Create Datasets
    train_dataset = InMemoryDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = InMemoryDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = InMemoryDataset(
        test_imgs, test_lbls, test_ids, transform=get_transforms("test")
    )

    # Create DataLoaders
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
