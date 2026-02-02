import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
)
from library.utils import get_age_scaler


class SiameseBreastDataset(Dataset):
    def __init__(
        self, metadata_path, transform=None, load_cached_data=True, is_train=False
    ):
        """
        Dataset for Siamese Network.
        Yields pairs of (Target, Contralateral) images with Age/Implant channels.
        """
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.is_train = is_train

        # Load Age Scaler (Cached)
        self.age_scaler = get_age_scaler(load_cached_data=load_cached_data)
        # Extract scalar stats for broadcasting
        self.age_mean = float(self.age_scaler.mean_[0])
        self.age_std = float(self.age_scaler.scale_[0])

        # Build Contralateral Lookup Index
        # Key: (patient_id, view, laterality) -> Value: file_path
        self.file_index = {}
        for _, row in self.df.iterrows():
            # Store path for every image to allow lookup of the opposite side
            key = (row["patient_id"], row["view"], row["laterality"])
            self.file_index[key] = row["file_path"]

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        full_path = os.path.join(INPUT_DIR, rel_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Attempt to load with OpenCV
        img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"cv2 failed to load image (returned None): {full_path}")

        # Handle high bit-depth (e.g., 16-bit) -> Normalize to 8-bit [0, 255]
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Ensure Grayscale (H, W)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def _build_3channel_tensor(self, img_arr, age_norm, implant_val):
        """
        Constructs (3, H, W) tensor: [Image, Age, Implant]
        """
        # 1. Image Channel: Normalize [0, 255] -> [0, 1]
        img_tensor = torch.from_numpy(img_arr).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # (1, H, W)

        _, h, w = img_tensor.shape

        # 2. Age Channel: Spatially broadcasted standardized age
        age_channel = torch.full((1, h, w), age_norm, dtype=torch.float32)

        # 3. Implant Channel: Spatially broadcasted binary implant status
        implant_channel = torch.full((1, h, w), implant_val, dtype=torch.float32)

        # Concatenate
        return torch.cat([img_tensor, age_channel, implant_channel], dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Identify Target & Contralateral ---
        patient_id = row["patient_id"]
        view = row["view"]
        laterality = row["laterality"]
        target_path = row["file_path"]

        # Determine opposite laterality
        contra_laterality = "R" if laterality == "L" else "L"
        contra_key = (patient_id, view, contra_laterality)
        contra_path = self.file_index.get(contra_key)

        # --- 2. Load Images ---
        # Target
        try:
            img_target = self._load_image(target_path)
        except Exception as e:
            raise RuntimeError(f"Failed loading target {target_path}: {e}")

        # Contralateral (with placeholder fallback)
        img_contra = None
        if contra_path:
            try:
                img_contra = self._load_image(contra_path)
            except Exception:
                # If lookup exists but file fails, strictly we could fail,
                # but let's fallback to placeholder to be robust against single file corruption
                img_contra = None

        # Create Black Placeholder if missing
        if img_contra is None:
            img_contra = np.zeros_like(img_target)

        # --- 3. Resize ---
        # Resize to fixed size before augmentation/stacking
        img_target = cv2.resize(
            img_target, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )
        img_contra = cv2.resize(
            img_contra, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )

        # --- 4. Synchronized Augmentation ---
        if self.transform:
            # Apply same transform to both
            transformed = self.transform(image=img_target, image_contra=img_contra)
            img_target = transformed["image"]
            img_contra = transformed["image_contra"]

        # --- 5. Feature Engineering (Scalars) ---
        # Age
        age = row["age"]
        if pd.isna(age):
            age_norm = 0.0  # Mean centered (0)
        else:
            age_norm = (age - self.age_mean) / (self.age_std + 1e-7)

        # Implant
        implant = row["implant"]
        if pd.isna(implant):
            implant_val = 0.0
        else:
            implant_val = float(implant)

        # --- 6. Tensor Construction ---
        target_tensor = self._build_3channel_tensor(img_target, age_norm, implant_val)
        contra_tensor = self._build_3channel_tensor(img_contra, age_norm, implant_val)

        # --- 7. Label ---
        if self.is_train:
            label = torch.tensor(row["cancer"], dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)

        return target_tensor, contra_tensor, label


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Configures 'image_contra' as an additional target for synchronized augmentation.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.Affine(translate_percent={"x": 0.1, "y": 0.1}, p=0.5),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return None


def get_dataloaders(
    train_path=TRAIN_METADATA_PATH,
    val_path=VAL_METADATA_PATH,
    test_path=TEST_METADATA_PATH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Creates and returns DataLoaders for Train, Val, and Test sets.
    """
    # Transforms
    train_transform = get_transforms(mode="train")

    # Datasets
    # Note: load_cached_data=True allows using the cached Age Scaler stats
    train_dataset = SiameseBreastDataset(
        metadata_path=train_path,
        transform=train_transform,
        is_train=True,
        load_cached_data=True,
    )

    val_dataset = SiameseBreastDataset(
        metadata_path=val_path, transform=None, is_train=True, load_cached_data=True
    )

    test_dataset = SiameseBreastDataset(
        metadata_path=test_path, transform=None, is_train=False, load_cached_data=True
    )

    # Loaders
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

    return train_loader, val_loader, test_loader
