import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import rle_decode


def load_and_cache_data(df, prefix, load_cached_data=True):
    """
    Loads images and masks from disk or cache.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'id', 'image_path', etc.
        prefix (str): Prefix for cache filenames (e.g., 'train_combined', 'test').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths) numpy arrays.
               masks will be None if not present in df.
    """
    cache_dir = Config.WORK_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"cached_{prefix}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"cached_{prefix}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"cached_{prefix}_depths.npy")

    has_masks = "rle_mask" in df.columns

    # Try loading from cache
    if load_cached_data:
        if os.path.exists(img_cache_path) and os.path.exists(depth_cache_path):
            if has_masks and not os.path.exists(mask_cache_path):
                pass  # Cache incomplete, reload
            else:
                # print(f"Loading {prefix} data from cache...")
                images = np.load(img_cache_path)
                depths = np.load(depth_cache_path)
                masks = np.load(mask_cache_path) if has_masks else None
                return images, masks, depths

    # Process from scratch
    # print(f"Processing {prefix} data from scratch...")
    images = []
    masks = []
    depths = df["z"].values.astype(np.float32)

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for safety, though metadata check should prevent this
            img = np.zeros(
                (Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG), dtype=np.uint8
            )
        images.append(img)

        # Load Mask if available
        if has_masks:
            if pd.isna(row["rle_mask"]):
                mask = np.zeros(
                    (Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG), dtype=np.uint8
                )
            else:
                mask = rle_decode(
                    row["rle_mask"], (Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG)
                )
            masks.append(mask)

    images = np.array(images, dtype=np.uint8)

    if has_masks:
        masks = np.array(masks, dtype=np.uint8)
    else:
        masks = None

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    if has_masks:
        np.save(mask_cache_path, masks)

    return images, masks, depths


def get_transforms(phase):
    """
    Returns Albumentations transforms for the specified phase.
    """
    # Common: Pad to 128x128 using reflection
    # We do padding at the end of the pipeline to ensure output size

    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.2
                ),
                A.PadIfNeeded(
                    min_height=Config.IMG_HEIGHT_TRAIN,
                    min_width=Config.IMG_WIDTH_TRAIN,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
            ]
        )
    else:
        # Validation / Test / TTA
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.IMG_HEIGHT_TRAIN,
                    min_width=Config.IMG_WIDTH_TRAIN,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                )
            ]
        )


class SaltDataset(Dataset):
    def __init__(
        self, images, masks, depths, transforms=None, depth_min=0, depth_max=1000
    ):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.transforms = transforms
        self.depth_min = depth_min
        self.depth_max = depth_max

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (101, 101) uint8
        depth_val = self.depths[idx]

        if self.masks is not None:
            mask = self.masks[idx]  # (101, 101) uint8
        else:
            # Dummy mask for test set
            mask = np.zeros_like(img)

        # Apply Augmentations
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Normalization
        # Image: [0, 255] -> [0, 1]
        img = img.astype(np.float32) / 255.0

        # Depth: Min-Max Scaling to [0, 1]
        d_norm = (depth_val - self.depth_min) / (self.depth_max - self.depth_min + 1e-6)

        # Input Channel Multiplexing: [Seismic, Seismic, Depth]
        # Create constant depth channel matching image spatial dimensions
        d_channel = np.full_like(img, d_norm, dtype=np.float32)

        # Stack channels
        input_tensor = np.stack([img, img, d_channel], axis=-1)  # (H, W, 3)

        # To Tensor (C, H, W)
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()

        # Mask to Tensor (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return input_tensor, mask_tensor


def get_loaders(fold_idx, load_cached_data=True, debug=False):
    """
    Creates DataLoaders using the provided train/val split.
    Ignores fold_idx and StratifiedKFold logic to maximize training data usage (Cite 00025).
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_meta = train_meta.iloc[:100]
        val_meta = val_meta.iloc[:20]

    # 2. Load Data Arrays (Cached separately)
    # We use the provided splits directly.
    train_images, train_masks, train_depths = load_and_cache_data(
        train_meta,
        prefix="train_split" if not debug else "debug_train_split",
        load_cached_data=load_cached_data,
    )

    val_images, val_masks, val_depths = load_and_cache_data(
        val_meta,
        prefix="val_split" if not debug else "debug_val_split",
        load_cached_data=load_cached_data,
    )

    # Calculate global depth stats for normalization from training data
    d_min = train_depths.min()
    d_max = train_depths.max()

    # 4. Create Datasets
    train_ds = SaltDataset(
        train_images,
        train_masks,
        train_depths,
        transforms=get_transforms("train"),
        depth_min=d_min,
        depth_max=d_max,
    )

    val_ds = SaltDataset(
        val_images,
        val_masks,
        val_depths,
        transforms=get_transforms("val"),
        depth_min=d_min,
        depth_max=d_max,
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
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
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load Data Arrays
    images, _, depths = load_and_cache_data(
        test_df, prefix="test", load_cached_data=load_cached_data
    )

    # We need the same depth normalization as training.
    # We load training depths just to get min/max.
    # In a production pipeline, these stats should be saved in a config or file.
    # Here we quickly re-calculate from metadata.
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    all_depths = pd.concat([train_meta["z"], val_meta["z"]])
    d_min = all_depths.min()
    d_max = all_depths.max()

    test_ds = SaltDataset(
        images,
        None,  # No masks for test
        depths,
        transforms=get_transforms("test"),
        depth_min=d_min,
        depth_max=d_max,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader, test_df["id"].values
