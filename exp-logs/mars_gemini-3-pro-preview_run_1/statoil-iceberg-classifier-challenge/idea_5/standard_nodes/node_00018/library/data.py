import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import seed_everything

# Constants
CACHE_DIR = "./working/idea_5/"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Global Statistics derived from Data Analysis
# Used for fixed normalization to ensure consistency across train/test
IMG_MIN = -45.0
IMG_MAX = 35.0
ANGLE_MEAN = 39.28
ANGLE_STD = 3.84


class IcebergDataset(Dataset):
    """
    Custom Dataset for Iceberg/Ship classification.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 224, 224, 3), float32 in [0, 1]
            angles (np.ndarray): Shape (N,), float32 normalized
            labels (np.ndarray, optional): Shape (N,), binary labels
            ids (np.ndarray, optional): Shape (N,), string IDs
            transform (callable, optional): PyTorch transforms
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply transforms
        # Note: ToTensor() converts numpy (H, W, C) to tensor (C, H, W)
        if self.transform:
            image = self.transform(image)

        # Convert angle to tensor
        angle_t = torch.tensor(angle, dtype=torch.float32)

        # Return tuple based on available data
        if self.labels is not None:
            label_t = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle_t, label_t
        else:
            # For test set, return ID to facilitate submission creation
            id_val = self.ids[idx] if self.ids is not None else ""
            return image, angle_t, id_val


def process_split(metadata_path, cache_prefix, load_cached_data):
    """
    Loads metadata, processes raw images/angles, and caches results as .npy files.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Cache file paths
    img_cache = os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy")
    ang_cache = os.path.join(CACHE_DIR, f"{cache_prefix}_angles.npy")
    lbl_cache = os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy")
    id_cache = os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Determine if this split expects labels
    has_labels = "test" not in cache_prefix

    # 1. Try Loading Cache
    if load_cached_data:
        # Check if all required files exist
        files_exist = (
            os.path.exists(img_cache)
            and os.path.exists(ang_cache)
            and os.path.exists(id_cache)
        )
        if has_labels:
            files_exist = files_exist and os.path.exists(lbl_cache)

        if files_exist:
            print(f"Loading cached data for {cache_prefix}...")
            images = np.load(img_cache)
            angles = np.load(ang_cache)
            ids = np.load(id_cache)
            labels = np.load(lbl_cache) if has_labels else None
            return images, angles, labels, ids

    # 2. Process from Scratch
    print(f"Processing data for {cache_prefix} from raw sources...")

    # Load Metadata
    df = pd.read_csv(metadata_path)

    # Load Raw JSONs referenced in metadata
    # We load them into a dictionary to avoid repeated I/O
    source_files = df["filepath"].unique()
    raw_data_map = {}
    for f_name in source_files:
        full_path = os.path.join(INPUT_DIR, f_name)
        with open(full_path, "r") as f:
            raw_data_map[f_name] = json.load(f)

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.zeros((n_samples, 224, 224, 3), dtype=np.float32)
    angles = np.zeros(n_samples, dtype=np.float32)
    ids = []
    labels = np.zeros(n_samples, dtype=np.float32) if has_labels else None

    for i, row in df.iterrows():
        # Retrieve raw sample
        source_file = row["filepath"]
        idx = row["sample_index"]
        sample = raw_data_map[source_file][idx]

        # --- Image Processing ---
        # 1. Reshape flattened bands
        b1 = np.array(sample["band_1"]).reshape(75, 75)
        b2 = np.array(sample["band_2"]).reshape(75, 75)

        # 2. Create 3rd band (Mean)
        b3 = (b1 + b2) / 2.0

        # 3. Stack to 3 channels (H, W, C)
        img = np.dstack((b1, b2, b3))

        # 4. Upsample to 224x224 (Bicubic)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

        # 5. Global Min-Max Scaling to [0, 1]
        img = (img - IMG_MIN) / (IMG_MAX - IMG_MIN)
        img = np.clip(img, 0.0, 1.0)

        images[i] = img

        # --- Angle Processing ---
        angle = row["inc_angle"]
        # Handle NA: Impute with global mean
        if pd.isna(angle) or angle == "na":
            angle = ANGLE_MEAN
        else:
            angle = float(angle)

        # Normalize (Standard Scaling)
        angles[i] = (angle - ANGLE_MEAN) / ANGLE_STD

        # --- Metadata ---
        ids.append(row["id"])
        if has_labels:
            labels[i] = row["is_iceberg"]

    # 3. Save to Cache
    np.save(img_cache, images)
    np.save(ang_cache, angles)
    np.save(id_cache, np.array(ids))
    if has_labels:
        np.save(lbl_cache, labels)

    return images, angles, labels, np.array(ids)


def get_transforms(phase):
    """
    Returns torchvision transforms for the specified phase.
    """
    if phase == "train":
        return transforms.Compose(
            [
                transforms.ToTensor(),  # Converts (H,W,C) -> (C,H,W)
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(
                    degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
            ]
        )
    else:
        return transforms.Compose([transforms.ToTensor()])


def get_dataloaders(batch_size=32, num_workers=4, load_cached_data=True):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of subprocesses for loading.
        load_cached_data (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything()

    # 1. Process Data
    train_imgs, train_angs, train_lbls, train_ids = process_split(
        os.path.join(METADATA_DIR, "train_metadata.csv"), "train", load_cached_data
    )
    val_imgs, val_angs, val_lbls, val_ids = process_split(
        os.path.join(METADATA_DIR, "val_metadata.csv"), "val", load_cached_data
    )
    test_imgs, test_angs, test_lbls, test_ids = process_split(
        os.path.join(METADATA_DIR, "test_metadata.csv"), "test", load_cached_data
    )

    # 2. Create Datasets
    train_ds = IcebergDataset(
        train_imgs, train_angs, train_lbls, train_ids, transform=get_transforms("train")
    )
    val_ds = IcebergDataset(
        val_imgs, val_angs, val_lbls, val_ids, transform=get_transforms("val")
    )
    test_ds = IcebergDataset(
        test_imgs, test_angs, None, test_ids, transform=get_transforms("test")
    )

    # 3. Create Loaders
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
