import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from library.config import Config

# Global constants for Angle Normalization derived from dataset analysis
ANGLE_MEAN = 39.2829
ANGLE_STD = 3.8362


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg/Ship classification task.
    Handles on-the-fly augmentation, resizing, and angle normalization.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Pre-processed images (N, 75, 75, 3).
            angles (np.ndarray): Incidence angles (N,).
            labels (np.ndarray, optional): Target labels (N,).
            transform (albumentations.Compose): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve data
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply augmentations and resizing
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Normalize incidence angle (Standard Scaling)
        angle = (angle - ANGLE_MEAN) / ANGLE_STD
        angle = torch.tensor(angle, dtype=torch.float32)

        # Return tuple based on availability of labels
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def get_transforms(phase: str):
    """
    Constructs the Albumentations transformation pipeline.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The composed transform.
    """
    # Base transformation: Bicubic Upsampling to target size
    resize = A.Resize(
        height=Config.IMG_SIZE, width=Config.IMG_SIZE, interpolation=cv2.INTER_CUBIC
    )

    transforms = [resize]

    if phase == "train":
        # Geometric Augmentations for Training
        transforms.extend(
            [
                A.HorizontalFlip(p=Config.HORIZONTAL_FLIP_PROB),
                A.VerticalFlip(p=Config.VERTICAL_FLIP_PROB),
                # Discrete rotation for cardinal invariance
                A.RandomRotate90(p=0.5),
                # Continuous rotation for boundary smoothing
                A.Rotate(
                    limit=Config.ROTATION_LIMIT, interpolation=cv2.INTER_CUBIC, p=0.5
                ),
            ]
        )

    # Convert to Tensor (HWC -> CHW)
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def process_json(json_path):
    """
    Parses a raw JSON file and processes the image bands and metadata.

    Processing steps:
    1. Reshape flattened bands to 75x75.
    2. Apply Min-Max normalization to Band 1 and Band 2.
    3. Compute Band 3 as the average of normalized Band 1 and Band 2.
    4. Stack bands to create (N, 75, 75, 3) array.
    5. Impute missing incidence angles with the global mean.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        tuple: (images, angles, labels, ids)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ids = []
    band_1_list = []
    band_2_list = []
    angles = []
    labels = []

    # Check if labels exist (only in train.json)
    has_labels = "is_iceberg" in data[0] if len(data) > 0 else False

    for item in data:
        ids.append(item["id"])
        band_1_list.append(item["band_1"])
        band_2_list.append(item["band_2"])

        # Handle missing incidence angles
        ang = item["inc_angle"]
        if ang == "na":
            angles.append(ANGLE_MEAN)
        else:
            angles.append(float(ang))

        if has_labels:
            labels.append(item["is_iceberg"])

    # Convert lists to numpy arrays and reshape
    b1 = np.array(band_1_list, dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(band_2_list, dtype=np.float32).reshape(-1, 75, 75)

    # Independent Band Normalization
    b1 = (b1 - Config.BAND1_MIN) / (Config.BAND1_MAX - Config.BAND1_MIN)
    b2 = (b2 - Config.BAND2_MIN) / (Config.BAND2_MAX - Config.BAND2_MIN)

    # Composite Band (Average of normalized bands)
    b3 = (b1 + b2) / 2.0

    # Stack to form 3-channel image: (N, 75, 75, 3)
    images = np.stack([b1, b2, b3], axis=-1)

    angles = np.array(angles, dtype=np.float32)
    ids = np.array(ids)

    if has_labels:
        labels = np.array(labels, dtype=np.float32)
        return images, angles, labels, ids
    else:
        return images, angles, None, ids


def load_cached_data_or_process(load_cached_data=True):
    """
    Loads processed data from cache or processes raw JSONs if cache is missing.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (tr_imgs, tr_angs, tr_lbls, tr_ids, te_imgs, te_angs, te_ids)
    """
    cache_dir = Config.WORK_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.npz")
    test_cache = os.path.join(cache_dir, "test_processed.npz")

    # Attempt to load from cache
    if load_cached_data and os.path.exists(train_cache) and os.path.exists(test_cache):
        try:
            train_data = np.load(train_cache)
            test_data = np.load(test_cache)
            return (
                train_data["images"],
                train_data["angles"],
                train_data["labels"],
                train_data["ids"],
                test_data["images"],
                test_data["angles"],
                test_data["ids"],
            )
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # Process from source
    train_path = os.path.join(Config.INPUT_DIR, "train.json")
    test_path = os.path.join(Config.INPUT_DIR, "test.json")

    tr_imgs, tr_angs, tr_lbls, tr_ids = process_json(train_path)
    te_imgs, te_angs, _, te_ids = process_json(test_path)

    # Save to cache
    np.savez(train_cache, images=tr_imgs, angles=tr_angs, labels=tr_lbls, ids=tr_ids)
    np.savez(test_cache, images=te_imgs, angles=te_angs, ids=te_ids)

    return tr_imgs, tr_angs, tr_lbls, tr_ids, te_imgs, te_angs, te_ids


def get_dataloaders(
    train_meta_path=os.path.join(Config.METADATA_DIR, "train_metadata.csv"),
    val_meta_path=os.path.join(Config.METADATA_DIR, "val_metadata.csv"),
    test_meta_path=os.path.join(Config.METADATA_DIR, "test_metadata.csv"),
    load_cached_data=True,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        train_meta_path (str): Path to training metadata CSV.
        val_meta_path (str): Path to validation metadata CSV.
        test_meta_path (str): Path to test metadata CSV.
        load_cached_data (bool): Whether to use cached numpy arrays.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader, test_ids)
    """
    # Load all processed data
    tr_imgs, tr_angs, tr_lbls, tr_ids, te_imgs, te_angs, te_ids = (
        load_cached_data_or_process(load_cached_data)
    )

    # Load Metadata for splitting
    df_train = pd.read_csv(train_meta_path)
    df_val = pd.read_csv(val_meta_path)
    df_test = pd.read_csv(test_meta_path)

    # Create Training Dataset
    train_indices = df_train["sample_index"].values
    train_dataset = IcebergDataset(
        images=tr_imgs[train_indices],
        angles=tr_angs[train_indices],
        labels=tr_lbls[train_indices],
        transform=get_transforms("train"),
    )

    # Create Validation Dataset
    val_indices = df_val["sample_index"].values
    val_dataset = IcebergDataset(
        images=tr_imgs[val_indices],
        angles=tr_angs[val_indices],
        labels=tr_lbls[val_indices],
        transform=get_transforms("val"),
    )

    # Create Test Dataset
    test_indices = df_test["sample_index"].values
    test_dataset = IcebergDataset(
        images=te_imgs[test_indices],
        angles=te_angs[test_indices],
        labels=None,
        transform=get_transforms("test"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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

    # Return test_ids from metadata to ensure alignment with predictions
    return train_loader, val_loader, test_loader, df_test["id"].values
