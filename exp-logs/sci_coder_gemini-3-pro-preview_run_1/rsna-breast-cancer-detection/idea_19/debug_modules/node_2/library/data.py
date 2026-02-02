import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIR,
    CACHE_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import load_image_robust

# Ensure deterministic behavior for albumentations and numpy
import random

random.seed(SEED)
np.random.seed(SEED)


def get_age_stats(load_cached_data=True):
    """
    Computes or loads the mean and standard deviation of patient age from the training set.
    Used for standard scaling of the Age channel.
    """
    cache_path = os.path.join(CACHE_DIR, "age_stats.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            stats = np.load(cache_path, allow_pickle=True).item()
            return stats["mean"], stats["std"]
        except Exception:
            pass  # Fallback to re-computing if load fails

    # Compute from scratch
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {TRAIN_METADATA_PATH}")

    df = pd.read_csv(TRAIN_METADATA_PATH)
    age_series = df["age"]

    # Handle missing values for statistics calculation
    valid_ages = age_series.dropna()

    mean_age = valid_ages.mean()
    std_age = valid_ages.std()

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, {"mean": mean_age, "std": std_age})

    return mean_age, std_age


class SiameseBreastCancerDataset(Dataset):
    def __init__(self, df, age_mean, age_std, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            age_mean (float): Mean age for normalization.
            age_std (float): Std age for normalization.
            transform (albumentations.Compose): Transforms to apply.
            is_test (bool): Whether this is the test set (returns prediction_id).
        """
        self.df = df.reset_index(drop=True)
        self.age_mean = age_mean
        self.age_std = age_std
        self.transform = transform
        self.is_test = is_test

        # Build lookup dictionary for contralateral images
        # Key: (patient_id, view, laterality) -> List of file_paths
        self.image_lookup = {}
        for idx, row in self.df.iterrows():
            key = (row["patient_id"], row["view"], row["laterality"])
            if key not in self.image_lookup:
                self.image_lookup[key] = []
            self.image_lookup[key].append(row["file_path"])

    def __len__(self):
        return len(self.df)

    def _load_and_process_image(self, file_path):
        """
        Loads image, converts to float32, scales to [0, 1].
        """
        full_path = os.path.join(INPUT_DIR, file_path)
        # load_image_robust raises FileNotFoundError or IOError if things go wrong
        img = load_image_robust(full_path)

        # Handle bit-depth and normalization
        # If image is 16-bit (uint16), max is 65535. If 8-bit, 255.
        # We convert to float32 and normalize to [0, 1].
        img = img.astype(np.float32)
        if img.max() > 255.0:
            img /= 65535.0
        else:
            img /= 255.0

        # Ensure it's 2D (H, W)
        if len(img.shape) == 3:
            img = img[:, :, 0]

        return img

    def _create_3channel_tensor(self, img_array, age, implant, target_shape):
        """
        Creates the (H, W, 3) tensor: [Image, Age_Map, Implant_Map]
        """
        h, w = target_shape

        # Channel 0: Image
        # Resize image to target shape if not already (though transforms usually handle this,
        # we need a consistent shape for stacking channels before transforms)
        # However, albumentations handles resizing. We just need to stack.
        # But wait, if input image is different size than expected, we rely on Albumentations Resize.
        # So we just stack matching the image's current dimensions.
        img_h, img_w = img_array.shape

        # Normalize Age
        if pd.isna(age):
            norm_age = 0.0  # (Mean - Mean) / Std = 0
        else:
            norm_age = (age - self.age_mean) / self.age_std

        # Implant
        implant_val = 1.0 if implant == 1 else 0.0

        # Create channels
        age_channel = np.full((img_h, img_w), norm_age, dtype=np.float32)
        implant_channel = np.full((img_h, img_w), implant_val, dtype=np.float32)

        # Stack: (H, W, 3)
        tensor_3ch = np.dstack([img_array, age_channel, implant_channel])
        return tensor_3ch

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Target Image
        target_path = row["file_path"]
        target_img_raw = self._load_and_process_image(target_path)

        # 2. Construct Target 3-Channel Input
        target_3ch = self._create_3channel_tensor(
            target_img_raw,
            row.get("age", np.nan),
            row.get("implant", 0),
            target_img_raw.shape,
        )

        # 3. Find and Load Contralateral Image
        # Logic: Same Patient, Same View, Opposite Laterality
        contra_lat = "R" if row["laterality"] == "L" else "L"
        contra_key = (row["patient_id"], row["view"], contra_lat)

        contra_candidates = self.image_lookup.get(contra_key, [])

        if len(contra_candidates) > 0:
            # Pick the first available contralateral image
            contra_path = contra_candidates[0]
            contra_img_raw = self._load_and_process_image(contra_path)

            # Construct Contra 3-Channel Input
            # Note: Age and Implant are patient-level, so they are same as target.
            contra_3ch = self._create_3channel_tensor(
                contra_img_raw,
                row.get("age", np.nan),
                row.get("implant", 0),
                contra_img_raw.shape,
            )
        else:
            # Missing Contralateral: Substitute zero-tensor
            # Shape should match target to allow stacking/transforms
            contra_3ch = np.zeros_like(target_3ch, dtype=np.float32)

        # 4. Apply Synchronized Transforms
        if self.transform:
            # Albumentations requires 'image' and we use 'additional_targets' for the pair
            # Note: The transform pipeline must have additional_targets={'contra': 'image'}
            augmented = self.transform(image=target_3ch, contra=contra_3ch)
            target_tensor = augmented["image"]
            contra_tensor = augmented["contra"]
        else:
            # Fallback to simple tensor conversion
            converter = ToTensorV2()
            target_tensor = converter(image=target_3ch)["image"]
            contra_tensor = converter(image=contra_3ch)["image"]

        # 5. Prepare Output
        # Extract label
        if self.is_test:
            label = 0.0  # Dummy
            pred_id = row["prediction_id"]
            return target_tensor, contra_tensor, label, pred_id
        else:
            label = float(row["cancer"])
            # For train/val, prediction_id is not strictly needed, but we can return a dummy or the index
            pred_id = f"{row['patient_id']}_{row['image_id']}"
            return target_tensor, contra_tensor, label, pred_id


def get_dataloaders(load_cached_data=True, max_samples=None):
    """
    Creates DataLoaders for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached stats/metadata.
        max_samples (int, optional): If set, limits dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Metadata
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Debugging: Subsample
    if max_samples:
        df_train = df_train.iloc[:max_samples]
        df_val = df_val.iloc[:max_samples]
        df_test = df_test.iloc[:max_samples]

    # 2. Get Age Statistics (Cached or Computed)
    age_mean, age_std = get_age_stats(load_cached_data)

    # 3. Define Transforms
    # We use additional_targets to apply the EXACT same geometric transform to both images

    # Training: Geometric Augmentations (Flip, Rotate, Shift)
    # No photometric augs (Brightness/Contrast) as per instructions.
    train_transform = A.Compose(
        [
            A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Affine(translate_percent={"x": 0.1, "y": 0.1}, p=0.5),
            ToTensorV2(),
        ],
        additional_targets={"contra": "image"},
    )

    # Validation/Test: Resize only
    val_transform = A.Compose(
        [A.Resize(height=IMG_SIZE[0], width=IMG_SIZE[1]), ToTensorV2()],
        additional_targets={"contra": "image"},
    )

    # 4. Create Datasets
    train_dataset = SiameseBreastCancerDataset(
        df_train, age_mean, age_std, transform=train_transform, is_test=False
    )

    val_dataset = SiameseBreastCancerDataset(
        df_val, age_mean, age_std, transform=val_transform, is_test=False
    )

    test_dataset = SiameseBreastCancerDataset(
        df_test, age_mean, age_std, transform=val_transform, is_test=True
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches to stabilize BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
