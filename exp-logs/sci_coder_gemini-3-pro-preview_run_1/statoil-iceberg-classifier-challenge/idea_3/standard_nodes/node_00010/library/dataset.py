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
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    IMG_SIZE,
)


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Ship vs Iceberg classification task.
    """

    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, H, W, C).
            angles (np.ndarray): Array of incidence angles with shape (N,).
            labels (np.ndarray, optional): Array of labels with shape (N,).
            transform (albumentations.Compose, optional): Augmentation pipeline.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image and angle
        image = self.images[idx]
        angle = self.angles[idx]

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default conversion to tensor (HWC -> CHW) if no transform is provided
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # Convert angle to tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, angle, label
        else:
            return image, angle


def get_transforms(mode="train"):
    """
    Returns the augmentation pipeline based on the mode.

    Args:
        mode (str): 'train' for aggressive geometric augmentations, 'valid'/'test' for standard formatting.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def _process_images(json_data, indices):
    """
    Helper function to process raw JSON band data into resized 3-channel images.
    """
    # Filter data based on indices
    selected_data = [json_data[i] for i in indices]

    processed_imgs = []
    for item in selected_data:
        # Reshape flattened bands to 75x75
        band_1 = np.array(item["band_1"]).reshape(75, 75)
        band_2 = np.array(item["band_2"]).reshape(75, 75)

        # Create 3-channel composite: Band 1, Band 2, Mean(B1, B2)
        img = np.zeros((75, 75, 3), dtype=np.float32)
        img[:, :, 0] = band_1
        img[:, :, 1] = band_2
        img[:, :, 2] = (band_1 + band_2) / 2.0

        # Upsample to 224x224 using Bicubic interpolation
        img_resized = cv2.resize(
            img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC
        )
        processed_imgs.append(img_resized)

    return np.array(processed_imgs, dtype=np.float32)


def load_data(mode="train", load_cached_data=True):
    """
    Loads, processes, and caches the dataset.

    Args:
        mode (str): 'train' (loads all labeled data) or 'test'.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.

    Returns:
        tuple: (images, angles, labels) for train mode, (images, angles, None) for test mode.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_prefix = "train_full" if mode == "train" else "test"
    img_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_images.npy")
    ang_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_angles.npy")
    lbl_cache_path = os.path.join(WORKING_DIR, f"{cache_prefix}_labels.npy")

    scaler_path = os.path.join(WORKING_DIR, "scaler_params.json")
    angle_mean_path = os.path.join(WORKING_DIR, "angle_mean.json")

    # 1. Attempt to load from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(ang_cache_path):
            # Check for labels if training
            if mode == "train" and not os.path.exists(lbl_cache_path):
                pass  # Cache incomplete, proceed to process
            else:
                print(f"Loading cached {mode} data from {WORKING_DIR}...")
                images = np.load(img_cache_path)
                angles = np.load(ang_cache_path)
                labels = np.load(lbl_cache_path) if mode == "train" else None
                return images, angles, labels

    print(f"Processing {mode} data from scratch...")

    # 2. Load Metadata and Raw JSON
    if mode == "train":
        # Combine train and val metadata to get full labeled dataset for Cross-Validation
        df_train = pd.read_csv(TRAIN_META_PATH)
        df_val = pd.read_csv(VAL_META_PATH)
        df_meta = pd.concat([df_train, df_val], ignore_index=True)

        with open(TRAIN_JSON, "r") as f:
            raw_json = json.load(f)
    else:
        df_meta = pd.read_csv(TEST_META_PATH)
        with open(TEST_JSON, "r") as f:
            raw_json = json.load(f)

    # 3. Process Images
    indices = df_meta["sample_index"].values
    images = _process_images(raw_json, indices)

    # 4. Scale Images (Min-Max to [0, 1])
    if mode == "train":
        # Compute global statistics from training data
        min_val = float(images.min())
        max_val = float(images.max())

        # Save scaler parameters for use during inference
        with open(scaler_path, "w") as f:
            json.dump({"min": min_val, "max": max_val}, f)
    else:
        # Load scaler parameters
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Scaler params not found. Please run training data loading first to generate statistics."
            )
        with open(scaler_path, "r") as f:
            params = json.load(f)
            min_val = params["min"]
            max_val = params["max"]

    # Apply scaling
    images = (images - min_val) / (max_val - min_val)
    # Clip to ensure valid range [0, 1]
    images = np.clip(images, 0, 1)

    # 5. Process Incidence Angles
    raw_angles = pd.to_numeric(df_meta["inc_angle"], errors="coerce").values

    if mode == "train":
        # Calculate mean from valid training angles
        valid_mask = ~np.isnan(raw_angles)
        angle_mean = float(np.mean(raw_angles[valid_mask]))

        # Save mean for inference
        with open(angle_mean_path, "w") as f:
            json.dump({"mean": angle_mean}, f)
    else:
        # Load mean
        if not os.path.exists(angle_mean_path):
            # Fallback if not found (e.g., debug run), use approximate value from analysis
            angle_mean = 39.28
        else:
            with open(angle_mean_path, "r") as f:
                angle_mean = json.load(f)["mean"]

    # Impute missing values
    raw_angles[np.isnan(raw_angles)] = angle_mean

    # Normalize angles (Divide by 45.0 to map approx [0, 45] -> [0, 1])
    angles = raw_angles / 45.0
    angles = angles.astype(np.float32)

    # 6. Process Labels
    labels = None
    if mode == "train":
        labels = df_meta["is_iceberg"].values.astype(np.float32)

    # 7. Save to Cache
    np.save(img_cache_path, images)
    np.save(ang_cache_path, angles)
    if labels is not None:
        np.save(lbl_cache_path, labels)

    return images, angles, labels
