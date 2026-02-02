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
from library.utils import rle_decode

# =========================================================================
# Constants
# =========================================================================
DEPTH_MIN = 0.0
DEPTH_MAX = 1000.0  # Max depth in dataset is ~959


# =========================================================================
# Augmentation & Transforms
# =========================================================================
def get_transforms(phase):
    """
    Returns the augmentation pipelines for the specified phase.
    Separates intensity and geometric transforms to handle depth channel correctly.
    """
    if phase == "train":
        # Intensity transforms apply only to the seismic image content
        intensity_transforms = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
            ]
        )

        # Geometric transforms apply to the stacked [Seismic, Seismic, Depth] image and mask
        geometric_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=Config.AUG_SHIFT_LIMIT,
                    scale_limit=Config.AUG_SCALE_LIMIT,
                    rotate_limit=Config.AUG_ROTATE_LIMIT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5,
                ),
                A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No intensity augs, just resize
        intensity_transforms = None
        geometric_transforms = A.Compose(
            [A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH), ToTensorV2()]
        )

    return intensity_transforms, geometric_transforms


# =========================================================================
# Data Caching Logic
# =========================================================================
def load_and_cache_data(df, name, load_cached=True):
    """
    Loads images and masks from disk or cache.

    Args:
        df (pd.DataFrame): Metadata dataframe containing 'id', 'image_path', 'z', and optionally 'rle_mask'.
        name (str): Unique identifier for this dataset split (e.g., 'train_fold0', 'val', 'test').
        load_cached (bool): Whether to attempt loading from .npy cache.

    Returns:
        tuple: (images, masks, depths, ids)
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # File paths for cache
    p_images = os.path.join(cache_dir, f"cached_{name}_images.npy")
    p_masks = os.path.join(cache_dir, f"cached_{name}_masks.npy")
    p_depths = os.path.join(cache_dir, f"cached_{name}_depths.npy")
    p_ids = os.path.join(cache_dir, f"cached_{name}_ids.npy")

    # 1. Try to load from cache
    if load_cached:
        if (
            os.path.exists(p_images)
            and os.path.exists(p_masks)
            and os.path.exists(p_depths)
            and os.path.exists(p_ids)
        ):
            try:
                images = np.load(p_images)
                masks = np.load(p_masks)
                depths = np.load(p_depths)
                ids = np.load(p_ids, allow_pickle=True)
                return images, masks, depths, ids
            except Exception as e:
                print(f"Failed to load cache for {name}: {e}. Recomputing...")

    # 2. Compute from scratch
    ids = df["id"].values
    depths = df["z"].values.astype(np.float32)

    image_list = []
    mask_list = []

    # Pre-check columns
    has_mask = "rle_mask" in df.columns

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for robustness, though metadata check should prevent this
            img = np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)

        image_list.append(img)

        # Load Mask
        if has_mask and pd.notna(row["rle_mask"]):
            mask = rle_decode(
                row["rle_mask"], shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH)
            )
            mask_list.append(mask)
        else:
            # Empty mask for test or missing labels
            mask_list.append(
                np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)
            )

    images = np.array(image_list, dtype=np.uint8)
    masks = np.array(mask_list, dtype=np.uint8)

    # 3. Save to cache
    np.save(p_images, images)
    np.save(p_masks, masks)
    np.save(p_depths, depths)
    np.save(p_ids, ids)

    return images, masks, depths, ids


# =========================================================================
# Dataset Class
# =========================================================================
class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, phase="train"):
        """
        Args:
            images (np.ndarray): (N, H, W) uint8
            masks (np.ndarray): (N, H, W) uint8
            depths (np.ndarray): (N,) float32
            ids (np.ndarray): (N,) string
            phase (str): 'train', 'val', or 'test'
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.phase = phase

        self.intensity_trans, self.geometric_trans = get_transforms(phase)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load raw data
        img = self.images[idx]  # (101, 101)
        mask = self.masks[idx]  # (101, 101)
        depth_val = self.depths[idx]

        # 2. Apply Intensity Augmentations (Seismic only)
        # Albumentations expects HWC, so we expand dims temporarily
        if self.intensity_trans:
            # img is HW, make it HWC for albumentations
            res = self.intensity_trans(image=img)
            img = res["image"]

        # 3. Normalization & Channel Construction
        # Normalize Seismic [0, 1]
        img = img.astype(np.float32) / 255.0

        # Create Depth Channel [0, 1]
        # Normalize depth using global min/max
        d_norm = (depth_val - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        # Create constant plane matching image spatial dims
        depth_plane = np.full_like(img, d_norm, dtype=np.float32)

        # Stack Channels: [Seismic, Seismic, Depth]
        # Result shape: (H, W, 3)
        img_3c = np.stack([img, img, depth_plane], axis=-1)

        # 4. Apply Geometric Augmentations (Flip, Rotate, Resize)
        # These must apply to both the 3D image stack and the mask
        if self.geometric_trans:
            res = self.geometric_trans(image=img_3c, mask=mask)
            img_tensor = res["image"]  # (3, 128, 128)
            mask_tensor = res["mask"]  # (128, 128)

            # Add channel dim to mask: (1, 128, 128)
            mask_tensor = mask_tensor.unsqueeze(0).float()

            return img_tensor, mask_tensor, self.ids[idx]

        # Fallback (should not be reached given get_transforms logic)
        return (
            torch.from_numpy(img_3c).permute(2, 0, 1),
            torch.from_numpy(mask).unsqueeze(0),
            self.ids[idx],
        )


# =========================================================================
# Stratification Helper
# =========================================================================
def get_stratified_folds(n_splits=5):
    """
    Loads metadata, merges train/val, and returns stratified folds.

    Returns:
        list of (train_df, val_df) tuples.
    """
    # Load metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge to get full dataset
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # Stratified Split
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)

    folds = []
    # Stratify by 'coverage_class'
    for train_idx, val_idx in skf.split(full_df, full_df["coverage_class"]):
        train_fold = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold = full_df.iloc[val_idx].reset_index(drop=True)
        folds.append((train_fold, val_fold))

    return folds


# =========================================================================
# Loader Factory
# =========================================================================
def make_loader(
    df,
    phase="train",
    batch_size=Config.BATCH_SIZE,
    load_cached=True,
    cache_name=None,
    shuffle=None,
):
    """
    Creates a DataLoader for the given dataframe.

    Args:
        df (pd.DataFrame): Data to load.
        phase (str): 'train', 'val', or 'test'.
        batch_size (int): Batch size.
        load_cached (bool): Whether to use disk caching for arrays.
        cache_name (str): Unique suffix for cache files. If None, uses 'phase'.
        shuffle (bool): Whether to shuffle. Defaults to True for train, False otherwise.

    Returns:
        DataLoader
    """
    if cache_name is None:
        cache_name = phase

    # 1. Load/Cache Data Arrays
    images, masks, depths, ids = load_and_cache_data(
        df, cache_name, load_cached=load_cached
    )

    # 2. Create Dataset
    dataset = SaltDataset(images, masks, depths, ids, phase=phase)

    # 3. Create DataLoader
    if shuffle is None:
        shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=(phase == "train"),
    )

    return loader
