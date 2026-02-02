import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import rle_decode

# -------------------------------------------------------------------------
# Caching & Data Loading Logic
# -------------------------------------------------------------------------


def load_and_cache_data(metadata_df, cache_name, load_cached_data=True):
    """
    Loads images, masks, and depths from disk or cache.

    Args:
        metadata_df (pd.DataFrame): Dataframe containing file paths and metadata.
        cache_name (str): Unique identifier for the cache files (e.g., 'train_val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
            images: np.ndarray (N, 101, 101) uint8
            masks: np.ndarray (N, 101, 101) uint8 (or float32 for soft targets if implemented)
            depths: np.ndarray (N,) float32
            ids: np.ndarray (N,) string
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_img_path = os.path.join(Config.WORKING_DIR, f"cached_{cache_name}_images.npy")
    cache_msk_path = os.path.join(Config.WORKING_DIR, f"cached_{cache_name}_masks.npy")
    cache_dep_path = os.path.join(Config.WORKING_DIR, f"cached_{cache_name}_depths.npy")
    cache_ids_path = os.path.join(Config.WORKING_DIR, f"cached_{cache_name}_ids.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_img_path)
            and os.path.exists(cache_dep_path)
            and os.path.exists(cache_ids_path)
        ):

            # Check mask cache existence only if we expect masks (train/val)
            # For test set, we might not have masks, but the cache logic handles it via checking df columns
            has_masks = "rle_mask" in metadata_df.columns
            if not has_masks or os.path.exists(cache_msk_path):
                print(f"Loading {cache_name} data from cache...")
                images = np.load(cache_img_path)
                depths = np.load(cache_dep_path)
                ids = np.load(cache_ids_path, allow_pickle=True)

                masks = None
                if has_masks:
                    masks = np.load(cache_msk_path)

                return images, masks, depths, ids

    # 2. Process from scratch
    print(f"Processing {cache_name} data from scratch...")

    ids = metadata_df["id"].values
    depths = metadata_df["z"].values.astype(np.float32)

    # Pre-allocate arrays
    n_samples = len(metadata_df)
    images = np.zeros((n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

    has_masks = "rle_mask" in metadata_df.columns
    masks = (
        np.zeros((n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
        if has_masks
        else None
    )

    for idx, row in metadata_df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images[idx] = img

        # Load Mask (if available)
        if has_masks:
            # If we have a direct mask path, we could load it, but RLE is more robust here
            # since metadata guarantees RLE presence.
            if pd.notna(row["rle_mask"]):
                masks[idx] = rle_decode(row["rle_mask"])
            else:
                masks[idx] = np.zeros(
                    (Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8
                )

    # 3. Save to cache
    np.save(cache_img_path, images)
    np.save(cache_dep_path, depths)
    np.save(cache_ids_path, ids)
    if has_masks:
        np.save(cache_msk_path, masks)

    return images, masks, depths, ids


# -------------------------------------------------------------------------
# Augmentations
# -------------------------------------------------------------------------


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.
    """
    # Reflection padding to resize 101x101 -> 128x128
    # This avoids interpolation artifacts on the edges.
    base_transforms = [
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT,
            always_apply=True,
        )
    ]

    if phase == "train":
        # Conservative augmentations for training
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
        ]
    else:
        aug_transforms = []

    # Normalization and Tensor conversion
    # Note: We do manual normalization in the Dataset class to handle the depth channel consistently.
    # So we only use ToTensorV2 here to convert the result to PyTorch tensors.
    # However, Albumentations works on numpy arrays. We will handle the float conversion in the dataset.

    return A.Compose(base_transforms + aug_transforms)


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        ids,
        phase="train",
        transform=None,
        pseudo_labels=None,
    ):
        """
        Args:
            images (np.ndarray): (N, 101, 101) uint8
            masks (np.ndarray): (N, 101, 101) uint8 or None
            depths (np.ndarray): (N,) float32
            ids (np.ndarray): (N,) string
            phase (str): 'train', 'valid', or 'test'
            transform (A.Compose): Albumentations transforms
            pseudo_labels (np.ndarray): Optional soft masks for semi-supervised learning.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.phase = phase
        self.transform = transform
        self.pseudo_labels = pseudo_labels

        # Pre-calculate depth normalization parameters (Global Min/Max)
        # We assume depths roughly range from 0 to 1000 based on dataset analysis.
        # Hardcoding or dynamic calculation? Dynamic is safer.
        self.depth_min = 0.0
        self.depth_max = 1000.0  # Approximate max from analysis (959)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Prepare Image
        # Scale to [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0

        # 2. Prepare Depth Channel
        # Normalize depth to [0, 1]
        d = self.depths[idx]
        d_norm = (d - self.depth_min) / (self.depth_max - self.depth_min)
        d_norm = np.clip(d_norm, 0, 1)

        # Create depth channel (constant value map)
        depth_channel = np.full_like(img, d_norm)

        # 3. Multiplexing: [Seismic, Seismic, Depth]
        # Stack along the last axis for Albumentations: (H, W, C)
        # img is (101, 101), depth_channel is (101, 101)
        # Result: (101, 101, 3)
        image_stacked = np.dstack([img, img, depth_channel])

        # 4. Prepare Mask
        mask = None
        if self.phase == "train" or self.phase == "valid":
            if self.pseudo_labels is not None:
                # Use soft pseudo-label if provided
                mask = self.pseudo_labels[idx]
            elif self.masks is not None:
                mask = self.masks[idx].astype(np.float32)

            # Ensure mask is (H, W, 1) for Albumentations if it exists
            if mask is not None and mask.ndim == 2:
                mask = np.expand_dims(mask, axis=-1)

        # 5. Apply Transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image_stacked, mask=mask)
                image_stacked = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image_stacked)
                image_stacked = augmented["image"]

        # 6. Convert to Tensor (C, H, W)
        # Albumentations returns numpy arrays if ToTensorV2 is not the last step.
        # We manually convert to ensure float32 and correct channel order.

        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image_stacked).permute(2, 0, 1).float()

        if mask is not None:
            # (H, W, 1) -> (1, H, W)
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()
            return image_tensor, mask_tensor

        return image_tensor, self.ids[idx]


# -------------------------------------------------------------------------
# Loader Factory
# -------------------------------------------------------------------------


def get_loaders(fold, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for the specified fold.

    Args:
        fold (int): The validation fold index (0-4).
        load_cached_data (bool): Whether to use cached numpy arrays.
        debug (bool): If True, subsamples data for quick debugging.

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)

    # Combine for Stratified Split
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Load Data Arrays (Cached or Scratch)
    # We load the entire dataset into memory. It's small enough (~3000 images * 10kb ~ 30MB).
    images, masks, depths, ids = load_and_cache_data(
        full_df, cache_name="train_val", load_cached_data=load_cached_data
    )

    # 3. Stratified Split
    # We use the 'coverage_class' column created in metadata generation
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # Get indices for the requested fold
    # We iterate to find the correct fold indices
    fold_indices = list(skf.split(full_df, full_df["coverage_class"]))
    train_idx, val_idx = fold_indices[fold]

    # 4. Debug Subsampling
    if debug:
        train_idx = train_idx[:100]
        val_idx = val_idx[:50]

    # 5. Create Datasets
    train_dataset = SaltDataset(
        images=images[train_idx],
        masks=masks[train_idx],
        depths=depths[train_idx],
        ids=ids[train_idx],
        phase="train",
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        images=images[val_idx],
        masks=masks[val_idx],
        depths=depths[val_idx],
        ids=ids[val_idx],
        phase="valid",
        transform=get_transforms("valid"),
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Creates DataLoader for the test set.
    """
    test_meta = pd.read_csv(Config.TEST_METADATA)

    images, _, depths, ids = load_and_cache_data(
        test_meta, cache_name="test", load_cached_data=load_cached_data
    )

    test_dataset = SaltDataset(
        images=images,
        masks=None,
        depths=depths,
        ids=ids,
        phase="test",
        transform=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
