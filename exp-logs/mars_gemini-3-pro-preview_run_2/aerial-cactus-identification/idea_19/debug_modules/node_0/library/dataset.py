import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class CactusDataset(Dataset):
    """
    PyTorch Dataset for the Cactus identification task.
    Stores images in memory as numpy arrays for fast access.
    """

    def __init__(self, images, labels=None, ids=None, transform=None):
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is HWC, uint8
        image = self.images[idx]

        # Apply augmentations/normalization
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Training/Validation mode: return image and label
        if self.labels is not None:
            # Create float tensor for BCE loss, shape (1,)
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
            return image, label

        # Test mode: return image and ID
        else:
            return image, self.ids[idx]


def get_transforms(phase: str):
    """
    Returns the albumentations transformations for the specific phase.
    Strategy: Light augmentation (Flips) + Normalization [0, 1].
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalize to [0, 1] by dividing by 255 (mean=0, std=1, max_pixel_value=255)
                A.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Normalize only
        return A.Compose(
            [
                A.Normalize(
                    mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        )


def _load_phase_data(phase, metadata_path, load_cached_data=True):
    """
    Internal helper to load data for a specific phase with caching.
    Reads images from disk, converts to RGB, and saves/loads as .npy files.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    images_path = os.path.join(cache_dir, f"{phase}_images.npy")
    labels_path = os.path.join(cache_dir, f"{phase}_labels.npy")
    ids_path = os.path.join(cache_dir, f"{phase}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(images_path)
        and os.path.exists(ids_path)
        and (phase == "test" or os.path.exists(labels_path))
    )

    # 1. Load from cache if requested and available
    if load_cached_data and cache_exists:
        # print(f"Loading {phase} data from cache...")
        images = np.load(images_path)
        ids = np.load(ids_path, allow_pickle=True)
        if phase != "test":
            labels = np.load(labels_path)
        else:
            labels = None
        return images, labels, ids

    # 2. Process from scratch if cache missing or reload requested
    # print(f"Processing {phase} data from source...")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    img_list = []
    label_list = []
    id_list = []

    for _, row in df.iterrows():
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Read image
        img = cv2.imread(full_path)
        if img is None:
            continue

        # Convert BGR (OpenCV default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_list.append(img)
        id_list.append(row["id"])

        if phase != "test":
            label_list.append(row["has_cactus"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    ids = np.array(id_list)

    if phase != "test":
        labels = np.array(label_list, dtype=np.float32)
    else:
        labels = None

    # 3. Save to cache for future use
    np.save(images_path, images)
    np.save(ids_path, ids)
    if labels is not None:
        np.save(labels_path, labels)

    return images, labels, ids


def get_datasets(debug_sample_size=None, load_cached_data=True):
    """
    Loads data and returns CactusDataset objects for train, val, and test.
    Applies debug slicing if specified.
    """
    # Load Train
    train_imgs, train_lbls, train_ids = _load_phase_data(
        "train", Config.TRAIN_METADATA, load_cached_data
    )

    # Load Val
    val_imgs, val_lbls, val_ids = _load_phase_data(
        "val", Config.VAL_METADATA, load_cached_data
    )

    # Load Test
    test_imgs, _, test_ids = _load_phase_data(
        "test", Config.TEST_METADATA, load_cached_data
    )

    # Apply Debug Slicing (if set)
    if debug_sample_size is not None:
        train_imgs = train_imgs[:debug_sample_size]
        train_lbls = train_lbls[:debug_sample_size]
        train_ids = train_ids[:debug_sample_size]

        val_imgs = val_imgs[:debug_sample_size]
        val_lbls = val_lbls[:debug_sample_size]
        val_ids = val_ids[:debug_sample_size]

        test_imgs = test_imgs[:debug_sample_size]
        test_ids = test_ids[:debug_sample_size]

    # Create Datasets
    train_dataset = CactusDataset(
        train_imgs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_dataset = CactusDataset(
        val_imgs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_dataset = CactusDataset(
        test_imgs, None, test_ids, transform=get_transforms("test")
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size, num_workers, debug_sample_size=None, load_cached_data=True
):
    """
    Returns dataloaders for train, val, and test.
    """
    train_ds, val_ds, test_ds = get_datasets(debug_sample_size, load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
