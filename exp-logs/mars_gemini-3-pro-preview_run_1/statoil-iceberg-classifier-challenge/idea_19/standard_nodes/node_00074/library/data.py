import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Constants
CACHE_DIR = "./working/idea_19/"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_and_cache_json(filename, cache_prefix, load_cached_data=True):
    """
    Parses a JSON file into numpy arrays and caches them.
    Returns: images (N, 75, 75, 2), angles (N,), ids (N,), labels (N,) [optional]
    """
    ensure_dir(CACHE_DIR)

    # Define cache paths
    paths = {
        "images": os.path.join(CACHE_DIR, f"{cache_prefix}_images.npy"),
        "angles": os.path.join(CACHE_DIR, f"{cache_prefix}_angles.npy"),
        "ids": os.path.join(CACHE_DIR, f"{cache_prefix}_ids.npy"),
        "labels": os.path.join(CACHE_DIR, f"{cache_prefix}_labels.npy"),
    }

    # Check if all required cache files exist
    all_exist = all(os.path.exists(p) for k, p in paths.items() if k != "labels")
    # For train, labels must also exist
    if "train" in cache_prefix and not os.path.exists(paths["labels"]):
        all_exist = False

    if load_cached_data and all_exist:
        print(f"Loading cached data for {cache_prefix}...")
        images = np.load(paths["images"])
        angles = np.load(paths["angles"])
        ids = np.load(paths["ids"], allow_pickle=True)
        labels = np.load(paths["labels"]) if "train" in cache_prefix else None
        return images, angles, ids, labels

    print(f"Processing raw {filename}...")
    file_path = os.path.join(INPUT_DIR, filename)
    with open(file_path, "r") as f:
        data = json.load(f)

    # Pre-allocate arrays
    n_samples = len(data)
    images = np.zeros((n_samples, 75, 75, 2), dtype=np.float32)
    angles = np.zeros(n_samples, dtype=np.float32)
    ids = np.empty(n_samples, dtype=object)
    labels = np.zeros(n_samples, dtype=np.float32) if "train" in cache_prefix else None

    for i, item in enumerate(data):
        # Process Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2

        # Process Angle
        ang = item["inc_angle"]
        if ang == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(ang)

        # Process ID
        ids[i] = item["id"]

        # Process Label
        if labels is not None:
            labels[i] = item["is_iceberg"]

    # Save to cache
    np.save(paths["images"], images)
    np.save(paths["angles"], angles)
    np.save(paths["ids"], ids)
    if labels is not None:
        np.save(paths["labels"], labels)

    return images, angles, ids, labels


def get_transforms(mode="train", img_size=224):
    """
    Returns albumentations transforms for train or validation/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    rotate_limit=20, shift_limit=0.1, scale_limit=0.1, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC), ToTensorV2()]
        )


class IcebergDataset(Dataset):
    def __init__(self, images, angles, ids, labels=None, transform=None, stats=None):
        """
        Args:
            images: (N, 75, 75, 2) numpy array
            angles: (N,) numpy array
            ids: (N,) numpy array
            labels: (N,) numpy array or None
            transform: albumentations transform
            stats: dict containing normalization stats
        """
        self.images = images
        self.angles = angles
        self.ids = ids
        self.labels = labels
        self.transform = transform
        self.stats = stats

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx].copy()  # (75, 75, 2)
        angle = self.angles[idx]

        # 1. Independent Band Normalization (Global Min-Max)
        # Band 1 (HH)
        b1 = img[:, :, 0]
        b1 = (b1 - self.stats["b1_min"]) / (self.stats["b1_max"] - self.stats["b1_min"])

        # Band 2 (HV)
        b2 = img[:, :, 1]
        b2 = (b2 - self.stats["b2_min"]) / (self.stats["b2_max"] - self.stats["b2_min"])

        # 2. Composite Band (Average of Normalized B1 and B2)
        b3 = (b1 + b2) / 2.0

        # Stack to 3 channels (H, W, 3)
        img_processed = np.dstack((b1, b2, b3))

        # 3. Augmentation & Upsampling (handled by transform)
        if self.transform:
            augmented = self.transform(image=img_processed)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform provided (should not happen in this pipeline)
            img_resized = cv2.resize(
                img_processed, (224, 224), interpolation=cv2.INTER_CUBIC
            )
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()

        # 4. Angle Normalization
        # Fill NaN with mean
        if np.isnan(angle):
            angle = self.stats["angle_mean"]

        # Standardize angle
        angle_norm = (angle - self.stats["angle_mean"]) / self.stats["angle_std"]
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # Label
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label, self.ids[idx]
        else:
            return img_tensor, angle_tensor, self.ids[idx]


def compute_stats(images, angles):
    """
    Computes global statistics from the training set.
    """
    # Image stats
    b1 = images[..., 0]
    b2 = images[..., 1]

    stats = {
        "b1_min": np.min(b1),
        "b1_max": np.max(b1),
        "b2_min": np.min(b2),
        "b2_max": np.max(b2),
    }

    # Angle stats (ignoring NaNs)
    valid_angles = angles[~np.isnan(angles)]
    stats["angle_mean"] = np.mean(valid_angles)
    stats["angle_std"] = np.std(valid_angles)

    return stats


def get_datasets(load_cached_data=True):
    """
    Prepares and returns Train, Val, and Test datasets.
    """
    set_seed(42)

    # 1. Load Raw Data (Cached or Parsed)
    # Train.json contains both train and val samples
    all_train_imgs, all_train_angles, all_train_ids, all_train_labels = (
        load_and_cache_json("train.json", "train", load_cached_data)
    )

    test_imgs, test_angles, test_ids, _ = load_and_cache_json(
        "test.json", "test", load_cached_data
    )

    # 2. Load Metadata to split Train/Val
    df_train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    df_val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    df_test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    # Extract indices
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values
    test_indices = df_test_meta["sample_index"].values

    # 3. Subset Data
    # Training Subset
    train_imgs = all_train_imgs[train_indices]
    train_angles = all_train_angles[train_indices]
    train_ids_sub = all_train_ids[train_indices]
    train_labels = all_train_labels[train_indices]

    # Validation Subset
    val_imgs = all_train_imgs[val_indices]
    val_angles = all_train_angles[val_indices]
    val_ids_sub = all_train_ids[val_indices]
    val_labels = all_train_labels[val_indices]

    # Test Subset (Metadata indices ensure order aligns with submission requirements if needed)
    # Note: test.json might be loaded in order, but using metadata ensures consistency
    test_imgs_sub = test_imgs[test_indices]
    test_angles_sub = test_angles[test_indices]
    test_ids_sub = test_ids[test_indices]

    # 4. Compute Global Stats (ON TRAIN SUBSET ONLY)
    stats = compute_stats(train_imgs, train_angles)
    print("Global Stats computed on Training Subset:")
    print(stats)

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        train_imgs,
        train_angles,
        train_ids_sub,
        train_labels,
        transform=get_transforms("train"),
        stats=stats,
    )

    val_dataset = IcebergDataset(
        val_imgs,
        val_angles,
        val_ids_sub,
        val_labels,
        transform=get_transforms("val"),
        stats=stats,
    )

    test_dataset = IcebergDataset(
        test_imgs_sub,
        test_angles_sub,
        test_ids_sub,
        labels=None,
        transform=get_transforms("test"),
        stats=stats,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Returns DataLoaders for train, val, and test.
    """
    train_ds, val_ds, test_ds = get_datasets(load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
