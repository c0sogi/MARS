import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        albumentations.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def _load_data_internal(metadata_csv, input_dir, mode):
    """
    Internal function to load data from disk (CSVs and Images).
    """
    df = pd.read_csv(metadata_csv)

    # Cast IDs to fixed-width unicode to avoid pickle requirement in npy
    ids = df["id"].values.astype("U10")
    depths = df["z"].values.astype(np.float32)

    n_samples = len(df)
    # Images are 101x101, grayscale (uint8)
    images = np.zeros((n_samples, 101, 101), dtype=np.uint8)
    masks = None

    # Check if masks should be loaded
    has_masks = "mask_path" in df.columns and mode in ["train", "val"]

    # For test set, mask_path might exist but be NaN/None, check first row or column
    if has_masks:
        # Verify if the column actually contains valid paths for at least one entry
        if df["mask_path"].isnull().all():
            has_masks = False
        else:
            masks = np.zeros((n_samples, 101, 101), dtype=np.uint8)

    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(input_dir, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        images[idx] = img

        # Load Mask
        if has_masks and pd.notna(row["mask_path"]):
            mask_path = os.path.join(input_dir, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Binarize to 0/1
                masks[idx] = (mask > 127).astype(np.uint8)
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    return ids, images, masks, depths


def load_cached_data(metadata_csv, mode, input_dir, cache_dir, load_cached=True):
    """
    Handles caching logic for the dataset.
    Loads from .npy files if available and load_cached is True.
    Otherwise, loads from raw images and saves to cache.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Derive cache keys from metadata filename (e.g., 'train', 'val', 'test')
    meta_name = os.path.splitext(os.path.basename(metadata_csv))[0]

    p_ids = os.path.join(cache_dir, f"{meta_name}_ids.npy")
    p_images = os.path.join(cache_dir, f"{meta_name}_images.npy")
    p_masks = os.path.join(cache_dir, f"{meta_name}_masks.npy")
    p_depths = os.path.join(cache_dir, f"{meta_name}_depths.npy")

    # Check if essential files exist
    files_exist = (
        os.path.exists(p_ids) and os.path.exists(p_images) and os.path.exists(p_depths)
    )

    if load_cached and files_exist:
        try:
            # allow_pickle=False ensures we are using standard types (U10 for ids)
            ids = np.load(p_ids, allow_pickle=False)
            images = np.load(p_images, allow_pickle=False)
            depths = np.load(p_depths, allow_pickle=False)

            masks = None
            if os.path.exists(p_masks):
                masks = np.load(p_masks, allow_pickle=False)

            # Simple consistency check
            if len(ids) == len(images):
                return ids, images, masks, depths
        except Exception as e:
            print(f"Failed to load cache for {meta_name}: {e}. Recomputing...")

    # Compute from scratch
    ids, images, masks, depths = _load_data_internal(metadata_csv, input_dir, mode)

    # Save to cache
    np.save(p_ids, ids)
    np.save(p_images, images)
    np.save(p_depths, depths)
    if masks is not None:
        np.save(p_masks, masks)

    return ids, images, masks, depths


class SaltDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        transform=None,
        mode="train",
        input_dir="./input",
        cache_dir="./working/idea_3/",
        load_cached=True,
    ):
        """
        Dataset for Salt Segmentation.

        Args:
            metadata_csv (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Augmentations.
            mode (str): 'train', 'val', or 'test'.
            input_dir (str): Root directory for input data.
            cache_dir (str): Directory to store/load cached .npy files.
            load_cached (bool): Whether to attempt loading from cache.
        """
        self.transform = transform
        self.mode = mode

        # Load data (Cached or Fresh)
        self.ids, self.images, self.masks, self.depths = load_cached_data(
            metadata_csv, mode, input_dir, cache_dir, load_cached
        )

        # Constants for Depth Normalization
        self.z_min = 50.0
        self.z_max = 960.0

        # Padding constants (101x101 -> 128x128)
        self.orig_h, self.orig_w = 101, 101
        self.target_h, self.target_w = 128, 128
        self.pad_h = self.target_h - self.orig_h
        self.pad_w = self.target_w - self.orig_w
        self.pad_top = self.pad_h // 2
        self.pad_bottom = self.pad_h - self.pad_top
        self.pad_left = self.pad_w // 2
        self.pad_right = self.pad_w - self.pad_left

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Fetch Raw Data
        image = self.images[idx]  # (101, 101) uint8
        z = self.depths[idx]
        img_id = self.ids[idx]

        mask = None
        if self.masks is not None:
            mask = self.masks[idx]  # (101, 101) uint8 (0 or 1)

        # 2. Pad to 128x128 using Reflection Padding
        image = cv2.copyMakeBorder(
            image,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_REFLECT,
        )

        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask,
                self.pad_top,
                self.pad_bottom,
                self.pad_left,
                self.pad_right,
                cv2.BORDER_REFLECT,
            )

        # 3. Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Ensure tensors and correct types (Float 0-1)
        # Albumentations ToTensorV2 usually handles conversion to Tensor, but preserves dtype
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        else:
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float().unsqueeze(0)
            elif mask.dtype == torch.uint8:
                mask = mask.float()

            # Ensure mask has channel dim (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        # 4. Depth Fusion
        # Normalize depth
        z_norm = (z - self.z_min) / (self.z_max - self.z_min)

        # Create depth channel (1, H, W) matching image spatial dims
        _, h, w = image.shape
        depth_channel = torch.full((1, h, w), z_norm, dtype=torch.float32)

        # Concatenate: Result is (2, 128, 128)
        image = torch.cat([image, depth_channel], dim=0)

        result = {"image": image, "id": img_id}
        if mask is not None:
            result["mask"] = mask

        return result
