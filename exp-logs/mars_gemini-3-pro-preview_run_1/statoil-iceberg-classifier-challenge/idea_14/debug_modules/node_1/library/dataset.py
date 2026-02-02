import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import log_message

# Constants for Angle Normalization (derived from Data Analysis)
ANGLE_MEAN = 39.2829
ANGLE_MIN = 30.0
ANGLE_MAX = 46.0


def process_json_to_numpy(json_path, mode="train"):
    """
    Parses the raw JSON file and converts it to numpy arrays.
    Handles 'na' in inc_angle by filling with global mean.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    ids = []
    bands_1 = []
    bands_2 = []
    angles = []
    labels = []

    for item in data:
        ids.append(item["id"])
        bands_1.append(item["band_1"])
        bands_2.append(item["band_2"])

        # Handle missing angles
        angle = item["inc_angle"]
        if angle == "na":
            angle = ANGLE_MEAN
        else:
            angle = float(angle)
        angles.append(angle)

        if mode == "train":
            labels.append(item["is_iceberg"])

    # Reshape bands to (N, 75, 75)
    # The raw data is a flattened list of 5625 floats
    b1 = np.array(bands_1, dtype=np.float32).reshape(-1, 75, 75)
    b2 = np.array(bands_2, dtype=np.float32).reshape(-1, 75, 75)

    # Stack to (N, 75, 75, 2)
    images = np.stack([b1, b2], axis=-1)

    ids = np.array(ids)
    angles = np.array(angles, dtype=np.float32)

    if mode == "train":
        labels = np.array(labels, dtype=np.float32)
        return images, angles, labels, ids
    else:
        return images, angles, ids


def load_and_process_data(mode="train", load_cached_data=True):
    """
    Loads data from cache if available, otherwise processes raw JSON and caches it.

    Args:
        mode (str): 'train' or 'test'.
        load_cached_data (bool): Whether to attempt loading from .npy files.

    Returns:
        Tuple of numpy arrays depending on mode.
    """
    # Define cache paths based on mode
    if mode == "train":
        cache_imgs = Config.CACHE_TRAIN_IMAGES
        cache_angles = Config.CACHE_TRAIN_ANGLES
        cache_labels = Config.CACHE_TRAIN_LABELS
        # We don't strictly need to cache IDs for train as we use metadata indices,
        # but good for consistency if needed.
        json_path = Config.TRAIN_JSON
    else:
        cache_imgs = Config.CACHE_TEST_IMAGES
        cache_angles = Config.CACHE_TEST_ANGLES
        cache_ids = Config.CACHE_TEST_IDS
        json_path = Config.TEST_JSON

    # Check cache existence
    cache_exists = False
    if mode == "train":
        if (
            os.path.exists(cache_imgs)
            and os.path.exists(cache_angles)
            and os.path.exists(cache_labels)
        ):
            cache_exists = True
    else:
        if (
            os.path.exists(cache_imgs)
            and os.path.exists(cache_angles)
            and os.path.exists(cache_ids)
        ):
            cache_exists = True

    # Load from cache
    if load_cached_data and cache_exists:
        log_message(f"Loading {mode} data from cache...")
        images = np.load(cache_imgs)
        angles = np.load(cache_angles)
        if mode == "train":
            labels = np.load(cache_labels)
            return images, angles, labels
        else:
            ids = np.load(cache_ids)
            return images, angles, ids

    # Process from scratch
    log_message(f"Processing {mode} data from raw JSON...")
    if mode == "train":
        images, angles, labels, ids = process_json_to_numpy(json_path, mode)
        # Save to cache
        np.save(cache_imgs, images)
        np.save(cache_angles, angles)
        np.save(cache_labels, labels)
        return images, angles, labels
    else:
        images, angles, ids = process_json_to_numpy(json_path, mode)
        # Save to cache
        np.save(cache_imgs, images)
        np.save(cache_angles, angles)
        np.save(cache_ids, ids)
        return images, angles, ids


class IcebergDataset(Dataset):
    def __init__(
        self, images, angles, labels=None, ids=None, transform=None, mode="train"
    ):
        """
        Args:
            images (np.array): Shape (N, 75, 75, 2)
            angles (np.array): Shape (N,)
            labels (np.array, optional): Shape (N,)
            ids (np.array, optional): Shape (N,)
            transform (A.Compose, optional): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve Data
        # Shape: (75, 75, 2)
        img = self.images[idx]
        angle = self.angles[idx]

        # 2. Corrected Composite Fusion & Normalization
        # Split bands
        b1 = img[:, :, 0]
        b2 = img[:, :, 1]

        # Global Min-Max Normalization
        b1_norm = (b1 - Config.BAND_1_MIN) / (Config.BAND_1_MAX - Config.BAND_1_MIN)
        b2_norm = (b2 - Config.BAND_2_MIN) / (Config.BAND_2_MAX - Config.BAND_2_MIN)

        # Composite Band (Average)
        b3_norm = (b1_norm + b2_norm) / 2.0

        # Stack to (75, 75, 3)
        # Values are roughly 0-1 now (though outliers could exceed slightly, which is fine for CNNs)
        img_composite = np.dstack((b1_norm, b2_norm, b3_norm))

        # 3. Upsampling (Bicubic)
        # Resize to (224, 224, 3)
        img_resized = cv2.resize(
            img_composite,
            (Config.RESIZE_WIDTH, Config.RESIZE_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

        # 4. Augmentation
        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            # Convert to tensor (C, H, W) manually if no transform provided
            # Albumentations ToTensorV2 does HWC->CHW and converts to float tensor
            img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()

        # 5. Angle Normalization
        angle_norm = (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)
        # Clip to 0-1 range just in case
        angle_norm = np.clip(angle_norm, 0.0, 1.0)
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # 6. Return
        if self.mode in ["train", "val"]:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label
        else:
            image_id = self.ids[idx]
            return img_tensor, angle_tensor, image_id


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def get_train_val_loaders(load_cached_data=True):
    """
    Creates DataLoaders for training and validation using pre-generated metadata splits.
    """
    # Load all training data
    images_all, angles_all, labels_all = load_and_process_data(
        mode="train", load_cached_data=load_cached_data
    )

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)

    # Extract indices
    train_indices = df_train["sample_index"].values
    val_indices = df_val["sample_index"].values

    # Subset data
    train_imgs = images_all[train_indices]
    train_angles = angles_all[train_indices]
    train_labels = labels_all[train_indices]

    val_imgs = images_all[val_indices]
    val_angles = angles_all[val_indices]
    val_labels = labels_all[val_indices]

    # Create Datasets
    train_dataset = IcebergDataset(
        train_imgs,
        train_angles,
        train_labels,
        transform=get_transforms("train"),
        mode="train",
    )
    val_dataset = IcebergDataset(
        val_imgs, val_angles, val_labels, transform=get_transforms("val"), mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    images, angles, ids = load_and_process_data(
        mode="test", load_cached_data=load_cached_data
    )

    # Use metadata to ensure order (though test.json order is usually preserved)
    # The metadata script preserves order of test.json, so direct loading is fine.
    # However, to be safe and consistent with metadata logic:
    df_test = pd.read_csv(Config.TEST_META)
    indices = df_test["sample_index"].values

    test_imgs = images[indices]
    test_angles = angles[indices]
    test_ids = ids[indices]

    dataset = IcebergDataset(
        test_imgs,
        test_angles,
        ids=test_ids,
        transform=get_transforms("test"),
        mode="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader


def get_pseudo_label_loader(pseudo_preds, load_cached_data=True):
    """
    Creates a combined DataLoader for Cycle 2 (Semi-Supervised).
    Merges original training data with high-confidence test samples.

    Args:
        pseudo_preds (dict): Dictionary mapping id -> probability (float).
        load_cached_data (bool): Whether to use cached data.
    """
    # 1. Load Original Training Data
    train_images, train_angles, train_labels = load_and_process_data(
        mode="train", load_cached_data=load_cached_data
    )

    # 2. Load Test Data
    test_images, test_angles, test_ids = load_and_process_data(
        mode="test", load_cached_data=load_cached_data
    )

    # 3. Filter Test Data based on Confidence
    selected_images = []
    selected_angles = []
    selected_labels = []

    # Map test_ids to indices for fast access
    # We iterate through test data and check predictions
    count_pseudo = 0

    for i, tid in enumerate(test_ids):
        if tid in pseudo_preds:
            prob = pseudo_preds[tid]

            # Check thresholds
            if prob >= Config.CONFIDENCE_THRESHOLD_HIGH:
                # High confidence Iceberg (1)
                selected_images.append(test_images[i])
                selected_angles.append(test_angles[i])
                selected_labels.append(1.0)
                count_pseudo += 1
            elif prob <= Config.CONFIDENCE_THRESHOLD_LOW:
                # High confidence Ship (0)
                selected_images.append(test_images[i])
                selected_angles.append(test_angles[i])
                selected_labels.append(0.0)
                count_pseudo += 1

    log_message(f"Pseudo-Labeling: Added {count_pseudo} samples from test set.")

    # 4. Concatenate
    if count_pseudo > 0:
        combined_images = np.concatenate(
            [train_images, np.array(selected_images)], axis=0
        )
        combined_angles = np.concatenate(
            [train_angles, np.array(selected_angles)], axis=0
        )
        combined_labels = np.concatenate(
            [train_labels, np.array(selected_labels)], axis=0
        )
    else:
        combined_images = train_images
        combined_angles = train_angles
        combined_labels = train_labels

    # 5. Create Dataset & Loader
    dataset = IcebergDataset(
        combined_images,
        combined_angles,
        combined_labels,
        transform=get_transforms("train"),
        mode="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
