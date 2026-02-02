import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_CSV,
    VAL_META_CSV,
    TEST_META_CSV,
    TRAIN_IMAGES_FILE,
    TRAIN_ANGLES_FILE,
    TRAIN_LABELS_FILE,
    TEST_IMAGES_FILE,
    TEST_ANGLES_FILE,
    TEST_IDS_FILE,
    WORKING_DIR,
    BAND_1_MEAN,
    BAND_1_STD,
    BAND_2_MEAN,
    BAND_2_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SIZE,
    SEED,
)
from library.utils import set_seed

# Calculated stats for Band 3 (Average of B1 and B2)
# Approximated as average of means and average of stds for normalization purposes
BAND_3_MEAN = (BAND_1_MEAN + BAND_2_MEAN) / 2.0
BAND_3_STD = (BAND_1_STD + BAND_2_STD) / 2.0


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx].copy()  # (3, 75, 75)
        angle = self.angles[idx]

        # Normalize
        # Channel 0: HH
        image[0] = (image[0] - BAND_1_MEAN) / BAND_1_STD
        # Channel 1: HV
        image[1] = (image[1] - BAND_2_MEAN) / BAND_2_STD
        # Channel 2: Avg
        image[2] = (image[2] - BAND_3_MEAN) / BAND_3_STD

        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms (Augmentations)
        # Transforms expect tensor (C, H, W)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            return image_tensor, angle_tensor


def get_transforms(phase):
    """
    Returns transformations for the given phase.
    """
    if phase == "train":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )
    else:
        return None


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it (images, angles, labels), and caches it as .npy files.
    Imputes missing angles in both train and test using training median.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define all required files
    required_files = [
        TRAIN_IMAGES_FILE,
        TRAIN_ANGLES_FILE,
        TRAIN_LABELS_FILE,
        TEST_IMAGES_FILE,
        TEST_ANGLES_FILE,
        TEST_IDS_FILE,
    ]

    # Check if we can load from cache
    if load_cached_data and all(os.path.exists(f) for f in required_files):
        print("Loading cached data...")
        X_train_full = np.load(TRAIN_IMAGES_FILE)
        angle_train_full = np.load(TRAIN_ANGLES_FILE)
        y_train_full = np.load(TRAIN_LABELS_FILE)

        X_test = np.load(TEST_IMAGES_FILE)
        angle_test = np.load(TEST_ANGLES_FILE)
        ids_test = np.load(TEST_IDS_FILE, allow_pickle=True)

        return (X_train_full, angle_train_full, y_train_full), (
            X_test,
            angle_test,
            ids_test,
        )

    print("Processing raw data from scratch...")

    # --- Process Training Data ---
    with open(TRAIN_JSON, "r") as f:
        train_data = json.load(f)

    df_train = pd.DataFrame(train_data)

    # Process Images
    # Band 1
    b1_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_1"]
        ]
    )
    # Band 2
    b2_train = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_train["band_2"]
        ]
    )
    # Band 3: Average
    b3_train = (b1_train + b2_train) / 2.0

    # Stack to (N, 3, 75, 75)
    X_train_full = np.stack([b1_train, b2_train, b3_train], axis=1)

    # Process Angles (Imputation)
    df_train["inc_angle"] = pd.to_numeric(df_train["inc_angle"], errors="coerce")
    angle_median = df_train["inc_angle"].median()
    df_train["inc_angle"] = df_train["inc_angle"].fillna(angle_median)
    angle_train_full = df_train["inc_angle"].values.astype(np.float32)

    # Process Labels
    y_train_full = df_train["is_iceberg"].values.astype(np.float32)

    # --- Process Test Data ---
    with open(TEST_JSON, "r") as f:
        test_data = json.load(f)

    df_test = pd.DataFrame(test_data)

    # Process Images
    b1_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_1"]
        ]
    )
    b2_test = np.array(
        [
            np.array(band).astype(np.float32).reshape(75, 75)
            for band in df_test["band_2"]
        ]
    )
    b3_test = (b1_test + b2_test) / 2.0

    X_test = np.stack([b1_test, b2_test, b3_test], axis=1)

    # Process Angles (Imputation using TRAIN median)
    df_test["inc_angle"] = pd.to_numeric(df_test["inc_angle"], errors="coerce")
    df_test["inc_angle"] = df_test["inc_angle"].fillna(angle_median)
    angle_test = df_test["inc_angle"].values.astype(np.float32)

    ids_test = df_test["id"].values

    # --- Save to Cache ---
    np.save(TRAIN_IMAGES_FILE, X_train_full)
    np.save(TRAIN_ANGLES_FILE, angle_train_full)
    np.save(TRAIN_LABELS_FILE, y_train_full)

    np.save(TEST_IMAGES_FILE, X_test)
    np.save(TEST_ANGLES_FILE, angle_test)
    np.save(TEST_IDS_FILE, ids_test)

    return (X_train_full, angle_train_full, y_train_full), (
        X_test,
        angle_test,
        ids_test,
    )


def get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    load_cached_data=True,
    debug=DEBUG,
    debug_size=DEBUG_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.
    Uses metadata files to split the processed training data.
    """
    set_seed(SEED)

    # 1. Get full processed data
    (X_full, angle_full, y_full), (X_test, angle_test, ids_test) = (
        process_and_cache_data(load_cached_data)
    )

    # 2. Load Metadata to determine splits
    train_meta = pd.read_csv(TRAIN_META_CSV)
    val_meta = pd.read_csv(VAL_META_CSV)
    test_meta = pd.read_csv(TEST_META_CSV)

    # 3. Subset Training Data
    # We use 'original_index' from metadata to pick correct rows from the full arrays
    train_indices = train_meta["original_index"].values
    val_indices = val_meta["original_index"].values

    X_train = X_full[train_indices]
    angle_train = angle_full[train_indices]
    y_train = y_full[train_indices]

    X_val = X_full[val_indices]
    angle_val = angle_full[val_indices]
    y_val = y_full[val_indices]

    # Test data usually corresponds 1:1 to test.json, but we can use metadata to be safe
    test_indices = test_meta["original_index"].values
    X_test_sorted = X_test[test_indices]
    angle_test_sorted = angle_test[test_indices]
    # ids_test is returned for submission generation, but dataset doesn't need it

    # 4. Debug Mode
    if debug:
        print(f"Debug mode enabled. Reducing dataset size to {debug_size}.")
        X_train = X_train[:debug_size]
        angle_train = angle_train[:debug_size]
        y_train = y_train[:debug_size]

        X_val = X_val[:debug_size]
        angle_val = angle_val[:debug_size]
        y_val = y_val[:debug_size]

        X_test_sorted = X_test_sorted[:debug_size]
        angle_test_sorted = angle_test_sorted[:debug_size]

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, angle_train, y_train, transform=get_transforms("train")
    )

    val_dataset = IcebergDataset(
        X_val, angle_val, y_val, transform=get_transforms("val")
    )

    test_dataset = IcebergDataset(
        X_test_sorted, angle_test_sorted, labels=None, transform=get_transforms("test")
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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

    return train_loader, val_loader, test_loader
