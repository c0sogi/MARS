import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import (
    INPUT_ROOT,
    WORKING_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    IMG_SIZE,
    ORIG_SIZE,
    DEPTH_MEAN,
    DEPTH_STD,
    DEPTH_FILL_VALUE,
    ELASTIC_ALPHA,
    ELASTIC_SIGMA,
    AUG_PROB,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Prevent OpenCV multithreading from conflicting with PyTorch DataLoader
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_transforms(phase):
    """
    Returns the albumentations transform pipeline for the given phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
    """
    # Calculate mean/std for 1-channel grayscale based on ImageNet stats
    # We average the RGB values to adapt to the summed-weight model architecture
    mean_val = float(np.mean(IMAGENET_MEAN))
    std_val = float(np.mean(IMAGENET_STD))

    transforms_list = []

    # 1. Padding (Common to all)
    # Pad from 101x101 to 128x128 using reflection padding
    transforms_list.append(
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0,
        )
    )

    if phase == "train":
        # 2. Augmentations (Train only)
        transforms_list.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        # Elastic transform simulates non-rigid salt deformation
                        A.ElasticTransform(
                            alpha=ELASTIC_ALPHA,
                            sigma=ELASTIC_SIGMA,
                            alpha_affine=None,
                            p=1.0,
                        ),
                        A.GridDistortion(p=1.0),
                        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                    ],
                    p=AUG_PROB,
                ),
                # Rigid geometric transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=AUG_PROB,
                ),
            ]
        )

    # 3. Normalization and Tensor Conversion (Common)
    # Normalize pixel values to the pretrained distribution
    transforms_list.extend(
        [
            A.Normalize(mean=[mean_val], std=[std_val], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


class SaltDataset(Dataset):
    """
    Dataset class for Salt Segmentation.
    Handles loading, caching, and preprocessing of seismic images and masks.
    """

    def __init__(self, mode="train", load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): If True, attempts to load pre-processed arrays from disk.
        """
        self.mode = mode
        self.transform = get_transforms(mode)

        # Define cache paths in the working directory
        self.cache_dir = WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.images_path = os.path.join(self.cache_dir, f"{mode}_images.npy")
        self.masks_path = os.path.join(self.cache_dir, f"{mode}_masks.npy")
        self.depths_path = os.path.join(self.cache_dir, f"{mode}_depths.npy")
        self.ids_path = os.path.join(self.cache_dir, f"{mode}_ids.npy")

        # Data Loading Logic
        if load_cached_data and self._check_cache_exists():
            # Load from cache
            self.images = np.load(self.images_path)
            self.depths = np.load(self.depths_path)
            self.ids = np.load(self.ids_path)
            if mode != "test":
                self.masks = np.load(self.masks_path)
            else:
                self.masks = None
        else:
            # Process from scratch
            self._process_and_cache()

    def _check_cache_exists(self):
        """Checks if all required cache files exist for the current mode."""
        files = [self.images_path, self.depths_path, self.ids_path]
        if self.mode != "test":
            files.append(self.masks_path)
        return all(os.path.exists(f) for f in files)

    def _process_and_cache(self):
        """Reads metadata, loads images/masks, and saves them to .npy files."""
        # Select appropriate metadata file
        if self.mode == "train":
            csv_path = TRAIN_CSV
        elif self.mode == "val":
            csv_path = VAL_CSV
        else:
            csv_path = TEST_CSV

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        images = []
        masks = []
        depths = []
        ids = []

        for _, row in df.iterrows():
            # 1. Load Image
            img_path = os.path.join(INPUT_ROOT, row["image_path"])
            # Load as grayscale (1 channel)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Safety fallback
                img = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)

            images.append(img)
            ids.append(row["id"])

            # 2. Load Depth
            z = row["z"]
            if pd.isna(z):
                # If depth is missing, fill with mean (which will be 0 after standardization)
                depths.append(DEPTH_MEAN)
            else:
                depths.append(z)

            # 3. Load Mask (if available)
            if self.mode != "test":
                mask_path = os.path.join(INPUT_ROOT, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.uint8)
                masks.append(mask)

        # Convert to numpy arrays
        self.images = np.array(images, dtype=np.uint8)
        self.depths = np.array(depths, dtype=np.float32)
        self.ids = np.array(ids)

        # Save to cache
        np.save(self.images_path, self.images)
        np.save(self.depths_path, self.depths)
        np.save(self.ids_path, self.ids)

        if self.mode != "test":
            self.masks = np.array(masks, dtype=np.uint8)
            np.save(self.masks_path, self.masks)
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Retrieve raw data
        image = self.images[idx]  # Shape: (101, 101)
        depth_val = self.depths[idx]
        image_id = self.ids[idx]

        if self.masks is not None:
            mask = self.masks[idx]  # Shape: (101, 101)
        else:
            # Create dummy mask for test set
            mask = np.zeros_like(image)

        # 2. Apply Augmentations and Preprocessing
        # Albumentations handles padding and normalization
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]  # Shape: (1, 128, 128) due to ToTensorV2
        mask_tensor = augmented[
            "mask"
        ]  # Shape: (128, 128) - ToTensorV2 preserves shape for mask

        # 3. Process Depth
        # Standardize: (z - mean) / std
        z_norm = (depth_val - DEPTH_MEAN) / DEPTH_STD

        # Handle any residual NaNs (e.g., if STD was 0 or data issue)
        if np.isnan(z_norm):
            z_norm = DEPTH_FILL_VALUE

        # Convert depth to tensor: Shape (1,)
        depth_tensor = torch.tensor([z_norm], dtype=torch.float32)

        # 4. Process Mask Tensor
        # Convert mask from uint8/long (0-255) to float (0.0-1.0) for BCE Loss
        mask_tensor = mask_tensor.float() / 255.0
        # Add channel dimension: (128, 128) -> (1, 128, 128)
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor, depth_tensor, image_id
