import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Optional pydicom import for robustness, though not in the mandatory list
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# Import configuration and utilities from the provided library
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IMG_HEIGHT,
    IMG_WIDTH,
    AGE_MEAN,
    AGE_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DEBUG,
    DEBUG_SAMPLE_SIZE,
)


def load_image(path):
    """
    Robust image loading function.
    Attempts to load using cv2 first, then pydicom if available.
    Raises FileNotFoundError if file is missing or cannot be loaded.
    """
    full_path = os.path.join(INPUT_DIR, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image file not found: {full_path}")

    # Method 1: OpenCV
    # cv2.IMREAD_UNCHANGED is used to attempt loading raw data (e.g. 16-bit PNG/TIFF)
    # or standard formats disguised as DCM.
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None and HAS_PYDICOM:
        # Method 2: Pydicom fallback
        try:
            dcm = pydicom.dcmread(full_path)
            img = dcm.pixel_array
        except Exception:
            img = None

    if img is None:
        raise FileNotFoundError(f"Failed to load image data from: {full_path}")

    # Ensure image is 2D (Grayscale)
    if len(img.shape) == 3:
        # If loaded as BGR/RGB, convert to Grayscale
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # If loaded as RGBA, convert
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    return img


def process_metadata(df, split_name, load_cached_data=True):
    """
    Processes metadata to pair target images with contralateral images.
    Implements caching using Parquet.
    """
    cache_path = os.path.join(WORKING_DIR, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Create lookup dictionary for finding contralateral images
    # Key: (patient_id, view, laterality) -> file_path
    # We need to look up the OPPOSITE laterality
    lookup = {}
    for idx, row in df.iterrows():
        key = (row["patient_id"], row["view"], row["laterality"])
        lookup[key] = row["file_path"]

    def get_contralateral_path(row):
        target_lat = row["laterality"]
        contra_lat = "R" if target_lat == "L" else "L"
        key = (row["patient_id"], row["view"], contra_lat)
        return lookup.get(key, None)

    df["contralateral_file_path"] = df.apply(get_contralateral_path, axis=1)

    # Save to cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


class MammographyDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Pre-convert columns to lists for faster indexing
        self.file_paths = self.df["file_path"].values
        self.contra_paths = self.df["contralateral_file_path"].values
        self.ages = self.df["age"].values
        self.implants = self.df["implant"].values

        # Labels are only available for train/val
        if mode != "test":
            self.labels = self.df["cancer"].values.astype(np.float32)
        else:
            self.labels = None
            self.prediction_ids = self.df["prediction_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Target Image
        target_path = self.file_paths[idx]
        img_target = load_image(target_path)

        # 2. Load Contralateral Image
        contra_path = self.contra_paths[idx]
        if contra_path is not None:
            try:
                img_contra = load_image(contra_path)
            except FileNotFoundError:
                # Fallback if file listed in metadata is corrupt/missing
                img_contra = np.zeros_like(img_target)
        else:
            # No contralateral image exists for this patient/view
            img_contra = np.zeros_like(img_target)

        # Ensure dimensions match (resize happens in transform, but initial shape consistency helps)
        # We rely on Albumentations Resize to handle shape differences.

        # 3. Prepare Metadata Features
        # Normalize Age
        age = self.ages[idx]
        if pd.isna(age):
            age = AGE_MEAN
        norm_age = (age - AGE_MEAN) / AGE_STD

        # Implant
        implant = self.implants[idx]
        if pd.isna(implant):
            implant = 0
        norm_implant = float(implant)

        # 4. Construct 3-Channel Volumes (Spatial Expansion) BEFORE Augmentation
        # This ensures geometric transforms (rotation/crop) apply to the metadata maps too.
        # Note: We need to resize images to a consistent size first if they vary wildly,
        # but Albumentations handles that. However, to stack, we need matching H,W.
        # We will let Albumentations handle the resizing of the 1-channel image first?
        # No, we must construct the volume first.
        # To do this, we resize the raw images to IMG_SIZE immediately.

        img_target = cv2.resize(img_target, (IMG_WIDTH, IMG_HEIGHT))
        img_contra = cv2.resize(img_contra, (IMG_WIDTH, IMG_HEIGHT))

        # Normalize Image to [0, 1]
        img_target = img_target.astype(np.float32)
        if img_target.max() > 1.0:
            img_target /= 255.0

        img_contra = img_contra.astype(np.float32)
        if img_contra.max() > 1.0:
            img_contra /= 255.0

        # Create Metadata Maps
        age_map = np.full((IMG_HEIGHT, IMG_WIDTH), norm_age, dtype=np.float32)
        implant_map = np.full((IMG_HEIGHT, IMG_WIDTH), norm_implant, dtype=np.float32)

        # Stack -> (H, W, 3)
        vol_target = np.stack([img_target, age_map, implant_map], axis=-1)
        vol_contra = np.stack([img_contra, age_map, implant_map], axis=-1)

        # 5. Apply Augmentations
        if self.transforms:
            # We pass the contralateral volume as an additional target to ensure
            # identical geometric transformations (flip, rotate) are applied to both.
            augmented = self.transforms(image=vol_target, contralateral=vol_contra)
            vol_target = augmented["image"]
            vol_contra = augmented["contralateral"]

        # 6. Return
        # ToTensorV2 converts (H, W, C) -> (C, H, W)

        if self.mode == "test":
            return {
                "image": vol_target,
                "contralateral": vol_contra,
                "prediction_id": self.prediction_ids[idx],
            }
        else:
            return {
                "image": vol_target,
                "contralateral": vol_contra,
                "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Ensures synchronized augmentation for paired inputs.
    """
    if mode == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                # Ensure output is tensor
                ToTensorV2(),
            ],
            additional_targets={"contralateral": "image"},
        )
    else:
        return A.Compose(
            [
                # Validation/Test: Just convert to tensor (Resizing handled in __getitem__)
                ToTensorV2()
            ],
            additional_targets={"contralateral": "image"},
        )


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    """
    # Load Metadata
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    df_val = pd.read_csv(VAL_METADATA_PATH)
    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Debug Mode
    if DEBUG:
        df_train = df_train.head(DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(DEBUG_SAMPLE_SIZE)

    # Process Metadata (Pairing)
    df_train = process_metadata(df_train, "train", load_cached_data)
    df_val = process_metadata(df_val, "val", load_cached_data)
    df_test = process_metadata(df_test, "test", load_cached_data)

    # Create Datasets
    train_ds = MammographyDataset(
        df_train, transforms=get_transforms("train"), mode="train"
    )
    val_ds = MammographyDataset(df_val, transforms=get_transforms("val"), mode="val")
    test_ds = MammographyDataset(
        df_test, transforms=get_transforms("test"), mode="test"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
