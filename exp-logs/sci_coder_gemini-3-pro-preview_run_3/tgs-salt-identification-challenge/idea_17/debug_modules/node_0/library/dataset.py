import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import Config


def get_transforms(phase="train"):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms = []

    # 1. Padding to 128x128 using Reflection Padding
    # This is applied to both train and validation/test to ensure consistent input size.
    transforms.append(
        A.PadIfNeeded(
            min_height=Config.MODEL_HEIGHT,
            min_width=Config.MODEL_WIDTH,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0,
        )
    )

    if phase == "train":
        # 2. Conservative Augmentations for Training
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.5
                ),
                # Conservative ShiftScaleRotate (Rotation < 5 deg, Scale < 5%)
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
            ]
        )

    # 3. Conversion to Tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms, additional_targets={"depth": "image"})


def load_and_cache_data(df, cache_prefix, load_cached_data=True):
    """
    Loads images, masks, and depths from disk or cache.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_prefix (str): Prefix for cache filenames (e.g., 'train_val', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, images, masks, depths) as numpy arrays.
               masks will be None if not present in df.
    """
    # Define cache paths
    path_ids = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_ids.npy")
    path_imgs = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_images.npy")
    path_masks = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_masks.npy")
    path_depths = os.path.join(Config.CACHE_DIR, f"cached_{cache_prefix}_depths.npy")

    has_masks = "mask_path" in df.columns

    # Attempt to load from cache
    if load_cached_data:
        if (
            os.path.exists(path_ids)
            and os.path.exists(path_imgs)
            and os.path.exists(path_depths)
        ):
            if not has_masks or os.path.exists(path_masks):
                try:
                    ids = np.load(path_ids, allow_pickle=True)
                    imgs = np.load(path_imgs)
                    depths = np.load(path_depths)
                    masks = np.load(path_masks) if has_masks else None
                    return ids, imgs, masks, depths
                except Exception:
                    pass  # Fallback to processing if load fails

    # Process from scratch
    ids = df["id"].values
    depths = df["z"].values.astype(np.float32)

    imgs_list = []
    masks_list = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for robustness, though metadata check passed
            img = np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)
        imgs_list.append(img)

        # Load Mask if available
        if has_masks:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)
            # Ensure binary (0, 1)
            mask = (mask > 127).astype(np.uint8)
            masks_list.append(mask)

    imgs = np.array(imgs_list, dtype=np.uint8)

    # Save to cache
    np.save(path_ids, ids)
    np.save(path_imgs, imgs)
    np.save(path_depths, depths)

    masks = None
    if has_masks:
        masks = np.array(masks_list, dtype=np.uint8)
        np.save(path_masks, masks)

    return ids, imgs, masks, depths


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles Input Channel Multiplexing [Seismic, Seismic, Depth].
    """

    def __init__(
        self, ids, images, masks, depths, transform=None, depth_min=0.0, depth_max=1.0
    ):
        self.ids = ids
        self.images = images
        self.masks = masks
        self.depths = depths
        self.transform = transform

        # Depth normalization parameters
        self.depth_min = depth_min
        self.depth_range = depth_max - depth_min if (depth_max - depth_min) > 0 else 1.0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1. Prepare Image (Seismic)
        # Scale to [0, 1]
        img_seismic = self.images[idx].astype(np.float32) / 255.0

        # 2. Prepare Depth
        # Normalize depth to [0, 1]
        z = self.depths[idx]
        z_norm = (z - self.depth_min) / self.depth_range
        # Create depth channel matching image spatial dimensions
        img_depth = np.full_like(img_seismic, z_norm, dtype=np.float32)

        # 3. Multiplex Channels: [Seismic, Seismic, Depth]
        # Shape: (H, W, 3)
        image_combined = np.stack([img_seismic, img_seismic, img_depth], axis=-1)

        # 4. Prepare Mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx].astype(np.float32)

        # 5. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image_combined, mask=mask)
                image_combined = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image_combined)
                image_combined = augmented["image"]

        # 6. Return
        # Image is already converted to Tensor (C, H, W) by ToTensorV2 in transform
        # Mask needs to be (1, H, W) or (H, W) depending on loss.
        # Albumentations ToTensorV2 doesn't transpose mask if it's 2D.
        # We return mask as (1, H, W) for consistency with PyTorch conventions.

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            return image_combined, mask, self.ids[idx]
        else:
            return image_combined, self.ids[idx]


def get_loaders(fold, debug=False, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    Merges train and val metadata, then performs Stratified K-Fold.

    Args:
        fold (int): The fold index to use for validation (0 to NUM_FOLDS-1).
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached numpy arrays.

    Returns:
        train_loader, val_loader
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge for consistent CV
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Load/Cache Data Arrays
    # We cache the full dataset first
    ids, images, masks, depths = load_and_cache_data(
        full_df, cache_prefix="train_val", load_cached_data=load_cached_data
    )

    # Calculate global depth stats for normalization
    depth_min = depths.min()
    depth_max = depths.max()

    # 3. Stratified Split
    # We use the 'coverage_class' column which exists in the metadata
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # The split indices
    fold_splits = list(skf.split(full_df, full_df["coverage_class"]))
    train_idx, val_idx = fold_splits[fold]

    # 4. Debug Subsampling
    if debug:
        train_idx = train_idx[: Config.DEBUG_SIZE]
        val_idx = val_idx[: Config.DEBUG_SIZE]

    # 5. Create Datasets
    train_dataset = SaltDataset(
        ids=ids[train_idx],
        images=images[train_idx],
        masks=masks[train_idx],
        depths=depths[train_idx],
        transform=get_transforms("train"),
        depth_min=depth_min,
        depth_max=depth_max,
    )

    val_dataset = SaltDataset(
        ids=ids[val_idx],
        images=images[val_idx],
        masks=masks[val_idx],
        depths=depths[val_idx],
        transform=get_transforms("valid"),
        depth_min=depth_min,
        depth_max=depth_max,
    )

    # 6. Create Loaders
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
    # 1. Load Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Load/Cache Data
    ids, images, masks, depths = load_and_cache_data(
        test_df, cache_prefix="test", load_cached_data=load_cached_data
    )

    # Need depth stats. Ideally should use stats from training set for consistency.
    # We will load train metadata just to get these stats.
    # Note: In a production pipeline, these stats should be saved in a config or artifact.
    # Here we quickly re-calculate or load cached train depths.
    try:
        train_depths_path = os.path.join(
            Config.CACHE_DIR, "cached_train_val_depths.npy"
        )
        if os.path.exists(train_depths_path):
            train_depths = np.load(train_depths_path)
            depth_min = train_depths.min()
            depth_max = train_depths.max()
        else:
            # Fallback to test set stats (suboptimal but functional)
            depth_min = depths.min()
            depth_max = depths.max()
    except:
        depth_min = depths.min()
        depth_max = depths.max()

    # 3. Create Dataset
    test_dataset = SaltDataset(
        ids=ids,
        images=images,
        masks=None,  # No masks for test
        depths=depths,
        transform=get_transforms("test"),
        depth_min=depth_min,
        depth_max=depth_max,
    )

    # 4. Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
