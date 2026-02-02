import os
import cv2
import torch
import numpy as np
import pandas as pd
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import read_dicom, create_mask


def get_transforms(split):
    """
    Returns the Albumentations transformation pipeline for a given split.
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.RandomBrightnessContrast(p=0.2),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_SIZE * 0.1),
                    max_width=int(Config.IMG_SIZE * 0.1),
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=0,
                    mask_fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=Config.IMG_SIZE, width=Config.IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def load_data_arrays(df, split, load_cached_data=True):
    """
    Handles the deterministic data processing and caching mechanism.
    Loads data from .npy cache if available, otherwise processes DICOMs and saves to cache.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths based on split and dataset size to ensure consistency
    cache_prefix = f"{split}_{len(df)}"
    img_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_masks.npy")
    label_cache_path = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_labels.npy")

    # 1. Try to load from cache
    if load_cached_data:
        images_exist = os.path.exists(img_cache_path)
        masks_exist = os.path.exists(mask_cache_path)
        labels_exist = os.path.exists(label_cache_path)

        if split == "test" and images_exist:
            print(f"Loading cached {split} data from {img_cache_path}...")
            images = np.load(img_cache_path)
            return {"images": images, "masks": None, "labels": None}
        elif split != "test" and images_exist and masks_exist and labels_exist:
            print(f"Loading cached {split} data from {img_cache_path}...")
            images = np.load(img_cache_path)
            masks = np.load(mask_cache_path)
            labels = np.load(label_cache_path)
            return {"images": images, "masks": masks, "labels": labels}

    # 2. Compute from scratch
    print(f"Processing {split} data (Cache miss or force reload)...")

    img_list = []
    mask_list = []
    label_list = []

    # Study level columns for labels
    study_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read Image (Resized to Config.IMG_SIZE)
        img = read_dicom(full_path, img_size=Config.IMG_SIZE)
        img_list.append(img)

        if split != "test":
            # Read Original Dims for Mask generation
            # We use pydicom to read headers quickly without decoding pixels
            try:
                dcm_header = pydicom.dcmread(full_path, stop_before_pixels=True)
                orig_h = dcm_header.Rows
                orig_w = dcm_header.Columns
            except Exception:
                # Fallback to current size if header read fails
                orig_h, orig_w = Config.IMG_SIZE, Config.IMG_SIZE

            # Create Mask
            boxes_str = row.get("boxes", "")
            mask = create_mask(
                boxes_str, img_size=Config.IMG_SIZE, orig_w=orig_w, orig_h=orig_h
            )
            mask_list.append(mask)

            # Get Label
            lbl = row[study_cols].values.astype(np.float32)
            label_list.append(lbl)

    # Convert lists to NumPy arrays
    images = np.stack(img_list)  # Shape: (N, H, W, 3), dtype: uint8

    if split != "test":
        masks = np.stack(mask_list)  # Shape: (N, H, W), dtype: float32
        labels = np.stack(label_list)  # Shape: (N, 4), dtype: float32
    else:
        masks = None
        labels = None

    # 3. Save to cache
    print(f"Saving processed {split} data to cache at {Config.CACHE_DIR}...")
    np.save(img_cache_path, images)
    if split != "test":
        np.save(mask_cache_path, masks)
        np.save(label_cache_path, labels)

    return {"images": images, "masks": masks, "labels": labels}


class SIIMDataset(Dataset):
    def __init__(self, df, split, transform=None, load_cached_data=True):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            load_cached_data (bool): Whether to use cached .npy files.
        """
        self.df = df
        self.split = split
        self.transform = transform

        # Load data into memory (utilizing the large RAM available)
        data = load_data_arrays(df, split, load_cached_data)
        self.images = data["images"]
        self.masks = data["masks"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, 3) uint8
        image = self.images[idx]

        # Retrieve IDs
        study_id = self.df.iloc[idx]["study_id"]
        image_id = self.df.iloc[idx]["image_id"]

        if self.split != "test":
            # Retrieve mask (H, W) float32 and label
            mask = self.masks[idx]
            label = self.labels[idx]

            if self.transform:
                # Apply augmentations
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                # Manual conversion if no transform provided
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
                mask = torch.tensor(mask)

            # Ensure mask has channel dimension: (1, H, W)
            # Albumentations ToTensorV2 returns (H, W) for 2D inputs
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            return {
                "image": image,
                "label": torch.tensor(label, dtype=torch.float32),
                "mask": mask.float(),
                "study_id": study_id,
                "image_id": image_id,
            }

        else:
            # Test set processing
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            else:
                image = torch.tensor(image).permute(2, 0, 1).float() / 255.0

            return {"image": image, "study_id": study_id, "image_id": image_id}
