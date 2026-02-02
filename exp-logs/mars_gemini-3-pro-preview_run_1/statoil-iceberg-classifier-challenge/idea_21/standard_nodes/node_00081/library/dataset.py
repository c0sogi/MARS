import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META,
    VAL_META,
    TEST_META,
    WORK_DIR,
    BAND1_MIN,
    BAND1_MAX,
    BAND2_MIN,
    BAND2_MAX,
    INC_ANGLE_MIN,
    INC_ANGLE_MAX,
    INC_ANGLE_MISSING_VAL,
    IMG_SIZE,
    SEED,
)
from library.utils import set_seed


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    # Cite solution_lesson_node_00042: Prefer Min-Max Scaling over Distribution Standardization for SAR.
    # We remove ImageNet normalization statistics and use mean=0, std=1 to preserve the [0, 1] intensity range.
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)

    if phase == "train":
        return A.Compose(
            [
                # Bicubic Upsampling to 224x224
                A.Resize(
                    height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                # Geometric Augmentations
                A.RandomRotate90(p=0.5),
                # ShiftScaleRotate with limit 20 degrees as per plan
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=20,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                # Normalization (Max pixel value 1.0 because we manually normalize bands to [0,1])
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Deterministic Resize
                A.Resize(
                    height=IMG_SIZE, width=IMG_SIZE, interpolation=cv2.INTER_CUBIC
                ),
                # Normalization
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 2) - Raw dB values.
            angles (np.ndarray): Shape (N,) - Incidence angles.
            labels (np.ndarray, optional): Shape (N,) - 0 or 1.
            ids (np.ndarray, optional): Shape (N,) - Image IDs.
            transform (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve raw image (75, 75, 2)
        image = self.images[idx]

        # 2. Independent Band Normalization
        # Band 1 (HH)
        b1 = (image[:, :, 0] - BAND1_MIN) / (BAND1_MAX - BAND1_MIN)
        # Band 2 (HV)
        b2 = (image[:, :, 1] - BAND2_MIN) / (BAND2_MAX - BAND2_MIN)

        # 3. Create Composite Band (Average)
        b3 = (b1 + b2) / 2.0

        # 4. Stack to create 3-channel image (75, 75, 3)
        # Result is float32 in range [0, 1] (mostly)
        img_3ch = np.stack([b1, b2, b3], axis=-1).astype(np.float32)

        # 5. Apply Transforms (Resize, Augment, ImageNet Norm, ToTensor)
        if self.transform:
            augmented = self.transform(image=img_3ch)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform provided (shouldn't happen in pipeline)
            img_tensor = torch.from_numpy(img_3ch.transpose(2, 0, 1))

        # 6. Process Incidence Angle
        angle = self.angles[idx]
        # Normalize angle to approx [0, 1]
        # Note: Missing values (0.0) will become negative, which is distinct.
        angle_norm = (angle - INC_ANGLE_MIN) / (INC_ANGLE_MAX - INC_ANGLE_MIN)
        angle_tensor = torch.tensor(angle_norm, dtype=torch.float32)

        # 7. Process Label and ID
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(-1.0, dtype=torch.float32)  # Dummy for test

        img_id = self.ids[idx] if self.ids is not None else ""

        return img_tensor, angle_tensor, label, img_id


def _process_json_subset(json_data, metadata_df):
    """
    Extracts images, angles, labels, and ids for a subset defined by metadata.
    """
    indices = metadata_df["sample_index"].values

    # Pre-allocate arrays
    n_samples = len(indices)
    images = np.zeros((n_samples, 75, 75, 2), dtype=np.float32)
    angles = np.zeros(n_samples, dtype=np.float32)
    ids = []
    labels = []
    has_labels = "is_iceberg" in metadata_df.columns

    for i, idx in enumerate(indices):
        item = json_data[idx]

        # Extract Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        images[i, :, :, 0] = b1
        images[i, :, :, 1] = b2

        # Extract Angle
        # Handle 'na' by checking type or value
        ang = item["inc_angle"]
        if ang == "na":
            angles[i] = INC_ANGLE_MISSING_VAL
        else:
            angles[i] = float(ang)

        # Extract ID
        ids.append(item["id"])

        # Extract Label
        if has_labels:
            labels.append(item["is_iceberg"])

    ids = np.array(ids)
    if has_labels:
        labels = np.array(labels, dtype=np.float32)
    else:
        labels = None

    return images, angles, labels, ids


def load_data_splits(load_cached_data=True):
    """
    Loads train, val, and test data. Uses caching to speed up subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy files.

    Returns:
        tuple: (train_data, val_data, test_data)
               Each is a dict with keys: 'images', 'angles', 'labels', 'ids'.
    """
    # Define cache filenames
    cache_files = {
        "train": [
            "train_images.npy",
            "train_angles.npy",
            "train_labels.npy",
            "train_ids.npy",
        ],
        "val": ["val_images.npy", "val_angles.npy", "val_labels.npy", "val_ids.npy"],
        "test": ["test_images.npy", "test_angles.npy", "test_ids.npy"],
    }

    # Check if cache exists
    cache_exists = True
    for split, files in cache_files.items():
        for f in files:
            if not os.path.exists(os.path.join(WORK_DIR, f)):
                cache_exists = False
                break

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        data_splits = {}

        # Load Train
        data_splits["train"] = {
            "images": np.load(os.path.join(WORK_DIR, "train_images.npy")),
            "angles": np.load(os.path.join(WORK_DIR, "train_angles.npy")),
            "labels": np.load(os.path.join(WORK_DIR, "train_labels.npy")),
            "ids": np.load(os.path.join(WORK_DIR, "train_ids.npy")),
        }

        # Load Val
        data_splits["val"] = {
            "images": np.load(os.path.join(WORK_DIR, "val_images.npy")),
            "angles": np.load(os.path.join(WORK_DIR, "val_angles.npy")),
            "labels": np.load(os.path.join(WORK_DIR, "val_labels.npy")),
            "ids": np.load(os.path.join(WORK_DIR, "val_ids.npy")),
        }

        # Load Test
        data_splits["test"] = {
            "images": np.load(os.path.join(WORK_DIR, "test_images.npy")),
            "angles": np.load(os.path.join(WORK_DIR, "test_angles.npy")),
            "labels": None,
            "ids": np.load(os.path.join(WORK_DIR, "test_ids.npy")),
        }

        return data_splits["train"], data_splits["val"], data_splits["test"]

    print("Cache not found or disabled. Processing raw data...")

    # Load Metadata
    df_train_meta = pd.read_csv(TRAIN_META)
    df_val_meta = pd.read_csv(VAL_META)
    df_test_meta = pd.read_csv(TEST_META)

    # --- Process Train/Val (from train.json) ---
    print(f"Loading {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r") as f:
        train_json_data = json.load(f)

    print("Processing Train split...")
    train_imgs, train_angs, train_lbls, train_ids = _process_json_subset(
        train_json_data, df_train_meta
    )

    print("Processing Val split...")
    val_imgs, val_angs, val_lbls, val_ids = _process_json_subset(
        train_json_data, df_val_meta
    )

    # Free memory
    del train_json_data

    # --- Process Test (from test.json) ---
    print(f"Loading {TEST_JSON}...")
    with open(TEST_JSON, "r") as f:
        test_json_data = json.load(f)

    print("Processing Test split...")
    test_imgs, test_angs, _, test_ids = _process_json_subset(
        test_json_data, df_test_meta
    )

    del test_json_data

    # --- Save to Cache ---
    print("Saving to cache...")
    # Train
    np.save(os.path.join(WORK_DIR, "train_images.npy"), train_imgs)
    np.save(os.path.join(WORK_DIR, "train_angles.npy"), train_angs)
    np.save(os.path.join(WORK_DIR, "train_labels.npy"), train_lbls)
    np.save(os.path.join(WORK_DIR, "train_ids.npy"), train_ids)

    # Val
    np.save(os.path.join(WORK_DIR, "val_images.npy"), val_imgs)
    np.save(os.path.join(WORK_DIR, "val_angles.npy"), val_angs)
    np.save(os.path.join(WORK_DIR, "val_labels.npy"), val_lbls)
    np.save(os.path.join(WORK_DIR, "val_ids.npy"), val_ids)

    # Test
    np.save(os.path.join(WORK_DIR, "test_images.npy"), test_imgs)
    np.save(os.path.join(WORK_DIR, "test_angles.npy"), test_angs)
    np.save(os.path.join(WORK_DIR, "test_ids.npy"), test_ids)

    train_data = {
        "images": train_imgs,
        "angles": train_angs,
        "labels": train_lbls,
        "ids": train_ids,
    }
    val_data = {
        "images": val_imgs,
        "angles": val_angs,
        "labels": val_lbls,
        "ids": val_ids,
    }
    test_data = {
        "images": test_imgs,
        "angles": test_angs,
        "labels": None,
        "ids": test_ids,
    }

    return train_data, val_data, test_data
