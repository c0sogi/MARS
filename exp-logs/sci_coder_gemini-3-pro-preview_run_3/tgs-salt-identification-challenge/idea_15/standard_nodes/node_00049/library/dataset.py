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


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles 3-channel input construction [Seismic, Seismic, Depth] and on-the-fly augmentation.
    """

    def __init__(self, images, masks, depths, ids, transforms=None, mode="train"):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        image = self.images[idx]  # (101, 101) uint8
        depth = self.depths[idx]  # scalar float

        # Normalize Image [0, 1]
        image = image.astype(np.float32) / 255.0

        # Create Depth Channel
        # Depth is expected to be normalized to [0, 1] before passing to this class
        depth_channel = np.full_like(image, depth, dtype=np.float32)

        # Construct 3-channel input: [Seismic, Seismic, Depth]
        # Stack along last dimension for Albumentations: (H, W, C)
        input_image = np.dstack([image, image, depth_channel])

        mask = None
        if self.mode != "test" and self.masks is not None:
            mask = self.masks[idx]  # (101, 101) uint8
            # Ensure mask is binary 0/1 float
            mask = (mask > 0).astype(np.float32)

        # Apply Augmentations
        if self.transforms:
            if self.mode != "test" and mask is not None:
                augmented = self.transforms(image=input_image, mask=mask)
                input_image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=input_image)
                input_image = augmented["image"]

        # Return tuple based on mode
        if self.mode != "test":
            return input_image, mask, self.ids[idx]
        else:
            return input_image, self.ids[idx]


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for train/val/test.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                # Conservative geometric augmentations
                A.ShiftScaleRotate(
                    scale_limit=0.05,
                    rotate_limit=5,
                    shift_limit=0.05,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                # Pad to 128x128 using reflection
                A.PadIfNeeded(
                    min_height=Config.IMG_HEIGHT,
                    min_width=Config.IMG_WIDTH,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test: Only Padding
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.IMG_HEIGHT,
                    min_width=Config.IMG_WIDTH,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )


def load_and_cache_data(
    metadata_df, cache_prefix, load_cached_data=True, is_test=False
):
    """
    Loads images, masks (if not test), depths, and ids.
    Uses caching mechanisms to store processed numpy arrays.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    p_imgs = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_images.npy")
    p_masks = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_masks.npy")
    p_depths = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_depths.npy")
    p_ids = os.path.join(Config.CACHE_DIR, f"{cache_prefix}_ids.npy")

    # Check if cache exists
    cache_exists = (
        os.path.exists(p_imgs) and os.path.exists(p_depths) and os.path.exists(p_ids)
    )
    if not is_test:
        cache_exists = cache_exists and os.path.exists(p_masks)

    if load_cached_data and cache_exists:
        print(f"Loading cached data for {cache_prefix}...")
        images = np.load(p_imgs)
        depths = np.load(p_depths)
        ids = np.load(p_ids)
        masks = np.load(p_masks) if not is_test else None
        return images, masks, depths, ids

    print(f"Processing data for {cache_prefix}...")

    # Lists to store data
    img_list = []
    mask_list = []
    depth_list = []
    id_list = []

    for idx, row in metadata_df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_list.append(img)

        # Load Mask (if not test)
        if not is_test:
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            mask_list.append(mask)

        # Depth
        depth_list.append(row["z"])

        # ID
        id_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    if not is_test:
        masks = np.array(mask_list, dtype=np.uint8)
    else:
        masks = None

    # Save to cache
    np.save(p_imgs, images)
    np.save(p_depths, depths)
    np.save(p_ids, ids)
    if not is_test:
        np.save(p_masks, masks)

    return images, masks, depths, ids


def get_dataloaders(fold=0, load_cached_data=True, debug=False):
    """
    Creates train and validation DataLoaders for a specific fold.
    Merges train and val metadata, then performs StratifiedKFold.
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge into a single dataframe for cross-validation
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Debug subset
    if debug or Config.DEBUG_SAMPLES:
        n_debug = Config.DEBUG_SAMPLES if Config.DEBUG_SAMPLES else 100
        full_meta = full_meta.iloc[:n_debug]
        print(f"Debug mode: Using {len(full_meta)} samples.")

    # 2. Load Data (Images, Masks, Depths)
    # Use a unique cache prefix based on debug status to avoid collisions
    cache_prefix = "train_val_debug" if (debug or Config.DEBUG_SAMPLES) else "train_val"
    images, masks, depths, ids = load_and_cache_data(
        full_meta, cache_prefix, load_cached_data, is_test=False
    )

    # 3. Normalize Depths
    # Using fixed scaling [0, 1000] to maintain consistency with test set and avoid data leakage/shift issues.
    # Depth range is approx 50-960.
    depths = depths / 1000.0

    # 4. Stratified Split
    # We use the 'coverage_class' from metadata for stratification
    coverage_classes = full_meta["coverage_class"].values

    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # Get indices for the requested fold
    splits = list(skf.split(images, coverage_classes))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} out of range (0-{len(splits)-1})")

    train_idx, val_idx = splits[fold]

    # 5. Create Datasets
    train_dataset = SaltDataset(
        images=images[train_idx],
        masks=masks[train_idx],
        depths=depths[train_idx],
        ids=ids[train_idx],
        transforms=get_transforms("train"),
        mode="train",
    )

    val_dataset = SaltDataset(
        images=images[val_idx],
        masks=masks[val_idx],
        depths=depths[val_idx],
        ids=ids[val_idx],
        transforms=get_transforms("val"),
        mode="val",
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
    Creates test DataLoader.
    """
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    cache_prefix = "test"
    images, _, depths, ids = load_and_cache_data(
        test_meta, cache_prefix, load_cached_data, is_test=True
    )

    # Normalize depths using same fixed constant as training
    depths = depths / 1000.0

    dataset = SaltDataset(
        images=images,
        masks=None,
        depths=depths,
        ids=ids,
        transforms=get_transforms("test"),
        mode="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
