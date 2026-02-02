import os
import json
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.configuration import Config
from library.utilities import get_or_create_cached_array


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.

    Features:
    - Returns Dual-Views for training (Consistency Regularization).
    - Returns single view for validation/test.
    - Handles image and angle data.
    """

    def __init__(
        self, images, angles, labels=None, ids=None, transform=None, mode="train"
    ):
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (H, W, C)
        image = self.images[idx]
        angle = self.angles[idx]

        if self.mode == "train":
            # Dual-View for Consistency Regularization
            if self.transform:
                # Apply transform twice to get two different views
                aug1 = self.transform(image=image)["image"]
                aug2 = self.transform(image=image)["image"]
            else:
                # Fallback to tensor conversion if no transform provided
                converter = ToTensorV2()
                aug1 = converter(image=image)["image"]
                aug2 = converter(image=image)["image"]

            label = self.labels[idx]
            return (
                (aug1, aug2),
                torch.tensor(angle, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )

        elif self.mode == "val":
            # Single view for validation
            if self.transform:
                image = self.transform(image=image)["image"]
            else:
                image = ToTensorV2()(image=image)["image"]

            label = self.labels[idx]
            return (
                image,
                torch.tensor(angle, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )

        elif self.mode == "test":
            # Single view for testing, return ID instead of label
            if self.transform:
                image = self.transform(image=image)["image"]
            else:
                image = ToTensorV2()(image=image)["image"]

            img_id = self.ids[idx]
            return image, torch.tensor(angle, dtype=torch.float32), img_id


def _process_raw_data():
    """
    Internal function to load JSONs, process images/angles, and return numpy arrays.
    This is called by get_or_create_cached_array.
    """
    print("Processing raw data from JSONs...")

    # 1. Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data = json.load(f)

    # 2. Extract Bands and Angles
    # Helper to extract bands
    def extract_bands(data):
        b1 = np.array([item["band_1"] for item in data], dtype=np.float32).reshape(
            -1, 75, 75
        )
        b2 = np.array([item["band_2"] for item in data], dtype=np.float32).reshape(
            -1, 75, 75
        )
        return b1, b2

    train_b1, train_b2 = extract_bands(train_data)
    test_b1, test_b2 = extract_bands(test_data)

    # 3. Compute Normalization Stats (Train Only)
    # Independent Band Normalization
    min_b1 = train_b1.min()
    max_b1 = train_b1.max()
    min_b2 = train_b2.min()
    max_b2 = train_b2.max()

    print(f"Stats - B1: [{min_b1:.2f}, {max_b1:.2f}], B2: [{min_b2:.2f}, {max_b2:.2f}]")

    # 4. Normalize and Create Composite Band
    def process_images(b1, b2):
        # Min-Max Normalize
        b1_norm = (b1 - min_b1) / (max_b1 - min_b1)
        b2_norm = (b2 - min_b2) / (max_b2 - min_b2)

        # Composite Band (Average)
        b3_norm = (b1_norm + b2_norm) / 2.0

        # Stack: (N, 75, 75, 3)
        # Transpose to (N, H, W, C) for cv2 resize
        images = np.stack([b1_norm, b2_norm, b3_norm], axis=-1)
        return images

    train_imgs_raw = process_images(train_b1, train_b2)
    test_imgs_raw = process_images(test_b1, test_b2)

    # 5. Upsample to 224x224 (Bicubic)
    def upsample(images):
        upsampled = []
        for img in images:
            # cv2.resize expects (H, W, C)
            res = cv2.resize(
                img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_CUBIC
            )
            upsampled.append(res)
        return np.array(upsampled, dtype=np.float32)

    print(f"Upsampling training images to {Config.IMG_SIZE}x{Config.IMG_SIZE}...")
    train_images = upsample(train_imgs_raw)
    print(f"Upsampling test images to {Config.IMG_SIZE}x{Config.IMG_SIZE}...")
    test_images = upsample(test_imgs_raw)

    # 6. Process Incidence Angles
    def get_angles(data):
        angles = []
        for item in data:
            ang = item["inc_angle"]
            if ang == "na":
                angles.append(np.nan)
            else:
                angles.append(float(ang))
        return np.array(angles, dtype=np.float32)

    train_angles = get_angles(train_data)
    test_angles = get_angles(test_data)

    # Impute 'na' with training mean
    angle_mean = np.nanmean(train_angles)
    angle_std = np.nanstd(train_angles)

    # Fill NaNs
    train_angles[np.isnan(train_angles)] = angle_mean
    # Note: Test set might have NaNs too, though usually not in this dataset, good practice to handle
    test_angles[np.isnan(test_angles)] = angle_mean

    # Normalize (Standardize)
    train_angles = (train_angles - angle_mean) / angle_std
    test_angles = (test_angles - angle_mean) / angle_std

    # 7. Extract Labels and IDs
    train_labels = np.array(
        [item["is_iceberg"] for item in train_data], dtype=np.float32
    )
    test_ids = np.array([item["id"] for item in test_data])

    # Return dictionary to be saved
    return {
        "train_images": train_images,
        "train_angles": train_angles,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_angles": test_angles,
        "test_ids": test_ids,
    }


def get_data_arrays(load_cached_data=True):
    """
    Wrapper to get data arrays, handling caching via utilities.
    """
    # We define a wrapper for the compute function that returns the specific array needed
    # However, since we process everything together, we check if one file exists to decide.

    cache_files = [
        "train_images.npy",
        "train_angles.npy",
        "train_labels.npy",
        "test_images.npy",
        "test_angles.npy",
        "test_ids.npy",
    ]

    all_exist = all(
        os.path.exists(os.path.join(Config.CACHE_DIR, f)) for f in cache_files
    )

    if load_cached_data and all_exist:
        # Load individually
        train_images = np.load(os.path.join(Config.CACHE_DIR, "train_images.npy"))
        train_angles = np.load(os.path.join(Config.CACHE_DIR, "train_angles.npy"))
        train_labels = np.load(os.path.join(Config.CACHE_DIR, "train_labels.npy"))
        test_images = np.load(os.path.join(Config.CACHE_DIR, "test_images.npy"))
        test_angles = np.load(os.path.join(Config.CACHE_DIR, "test_angles.npy"))
        test_ids = np.load(os.path.join(Config.CACHE_DIR, "test_ids.npy"))
    else:
        # Compute all
        data = _process_raw_data()

        # Save all
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(
            os.path.join(Config.CACHE_DIR, "train_images.npy"), data["train_images"]
        )
        np.save(
            os.path.join(Config.CACHE_DIR, "train_angles.npy"), data["train_angles"]
        )
        np.save(
            os.path.join(Config.CACHE_DIR, "train_labels.npy"), data["train_labels"]
        )
        np.save(os.path.join(Config.CACHE_DIR, "test_images.npy"), data["test_images"])
        np.save(os.path.join(Config.CACHE_DIR, "test_angles.npy"), data["test_angles"])
        np.save(os.path.join(Config.CACHE_DIR, "test_ids.npy"), data["test_ids"])

        train_images = data["train_images"]
        train_angles = data["train_angles"]
        train_labels = data["train_labels"]
        test_images = data["test_images"]
        test_angles = data["test_angles"]
        test_ids = data["test_ids"]

    return train_images, train_angles, train_labels, test_images, test_angles, test_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    # 1. Load Processed Arrays
    (
        train_imgs_all,
        train_angles_all,
        train_labels_all,
        test_imgs,
        test_angles,
        test_ids,
    ) = get_data_arrays(load_cached_data)

    # 2. Load Metadata for Splits
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)
    df_test_meta = pd.read_csv(Config.TEST_META)

    # 3. Subset Training Data
    # The arrays are indexed by the order in train.json.
    # The metadata 'sample_index' column maps to this order.
    train_indices = df_train_meta["sample_index"].values
    val_indices = df_val_meta["sample_index"].values

    X_train = train_imgs_all[train_indices]
    a_train = train_angles_all[train_indices]
    y_train = train_labels_all[train_indices]

    X_val = train_imgs_all[val_indices]
    a_val = train_angles_all[val_indices]
    y_val = train_labels_all[val_indices]

    # Test data is already in order of test.json, but let's ensure alignment if needed.
    # The test metadata also has sample_index, which should be 0..N-1.
    test_indices = df_test_meta["sample_index"].values
    X_test = test_imgs[test_indices]
    a_test = test_angles[test_indices]
    ids_test = test_ids[test_indices]

    # 4. Define Transforms
    # Training: Geometric Augmentation
    train_transform = A.Compose(
        [
            A.Rotate(
                limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),  # Continuous Rotation
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    # Validation/Test: Just ToTensor (Normalization already done in preprocessing)
    val_transform = A.Compose([ToTensorV2()])

    # 5. Create Datasets
    train_dataset = IcebergDataset(
        X_train, a_train, y_train, transform=train_transform, mode="train"
    )
    val_dataset = IcebergDataset(
        X_val, a_val, y_val, transform=val_transform, mode="val"
    )
    test_dataset = IcebergDataset(
        X_test, a_test, ids=ids_test, transform=val_transform, mode="test"
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for stability
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
