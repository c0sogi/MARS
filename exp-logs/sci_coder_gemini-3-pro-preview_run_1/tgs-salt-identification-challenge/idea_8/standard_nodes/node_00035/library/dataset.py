import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from library.config import Config


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles image loading, depth fusion, reflection padding, and augmentation.
    """

    def __init__(
        self,
        images,
        masks=None,
        depths=None,
        ids=None,
        transforms=None,
        mode="train",
    ):
        """
        Args:
            images (np.ndarray): Array of images (N, 101, 101).
            masks (np.ndarray, optional): Array of masks (N, 101, 101).
            depths (np.ndarray): Array of normalized depths (N,).
            ids (np.ndarray): Array of image IDs.
            transforms (albumentations.Compose): Augmentation pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.transforms = transforms
        self.mode = mode

        # Padding parameters to go from 101x101 to 128x128
        # Target: 128. Diff: 27.
        # Top/Left: 13, Bottom/Right: 14
        self.pad_h = (13, 14)
        self.pad_w = (13, 14)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load and Normalize Image
        # Input is uint8 (0-255), convert to float32 (0-1)
        img = self.images[idx].astype(np.float32) / 255.0

        # 2. Apply Reflection Padding to Image
        # Pad from 101x101 to 128x128
        img = np.pad(img, (self.pad_h, self.pad_w), mode="reflect")

        # 3. Fuse Depth Information
        # Create a dense depth channel matching the padded image dimensions
        z = self.depths[idx]
        depth_channel = np.full_like(img, z)

        # Stack to create (H, W, 2) input
        img_combined = np.dstack([img, depth_channel])

        # 4. Handle Mask (if available)
        mask = None
        if self.masks is not None:
            mask = self.masks[idx]
            # Normalize to 0-1 float
            mask = (mask > 127).astype(np.float32)
            # Apply same reflection padding to mask
            mask = np.pad(mask, (self.pad_h, self.pad_w), mode="reflect")

        # 5. Apply Augmentations
        # Albumentations expects (H, W, C) for image
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=img_combined, mask=mask)
                img_combined = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=img_combined)
                img_combined = augmented["image"]

        # 6. Final Tensor Conversion Check
        # ToTensorV2 usually handles this, but ensuring manual fallback for safety
        if not isinstance(img_combined, torch.Tensor):
            # Transpose (H, W, C) -> (C, H, W)
            img_combined = img_combined.transpose(2, 0, 1)
            img_combined = torch.from_numpy(img_combined).float()

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()

            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)

        # Return format depends on mode
        if self.mode == "test":
            return img_combined, self.ids[idx]
        else:
            return img_combined, mask, self.ids[idx]


def cache_and_load_data(df, prefix, load_cached=True):
    """
    Loads images and masks from disk, caches them as .npy files,
    and returns the arrays. Implements deterministic caching logic.

    Args:
        df (pd.DataFrame): Metadata dataframe containing paths.
        prefix (str): Prefix for cache files (e.g., 'train', 'val').
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (images, masks, depths, ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    img_cache_path = os.path.join(cache_dir, f"{prefix}_images.npy")
    mask_cache_path = os.path.join(cache_dir, f"{prefix}_masks.npy")
    depth_cache_path = os.path.join(cache_dir, f"{prefix}_depths.npy")
    id_cache_path = os.path.join(cache_dir, f"{prefix}_ids.npy")

    # 1. Try to load from cache
    if load_cached:
        files_exist = (
            os.path.exists(img_cache_path)
            and os.path.exists(depth_cache_path)
            and os.path.exists(id_cache_path)
        )
        # For train/val, mask cache must also exist
        if "test" not in prefix:
            files_exist = files_exist and os.path.exists(mask_cache_path)

        if files_exist:
            print(f"Loading cached {prefix} data from {cache_dir}...")
            images = np.load(img_cache_path)
            depths = np.load(depth_cache_path)
            ids = np.load(id_cache_path, allow_pickle=True)
            masks = None
            if "test" not in prefix:
                masks = np.load(mask_cache_path)
            return images, masks, depths, ids

    # 2. Process from scratch if cache missing or load_cached=False
    print(f"Processing {prefix} data from scratch...")

    images = []
    masks = []
    depths = []
    ids = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        # Force grayscale load
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Fallback if image is saved as RGBA but content is grayscale
        if img is None:
            img_temp = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_temp is not None:
                if len(img_temp.shape) == 3:
                    img = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
                elif len(img_temp.shape) == 2:
                    img = img_temp

        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        images.append(img)
        depths.append(row["z"])
        ids.append(row["id"])

        # Load Mask if path exists
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids)

    # Save to cache
    np.save(img_cache_path, images)
    np.save(depth_cache_path, depths)
    np.save(id_cache_path, ids)

    if len(masks) > 0:
        masks = np.array(masks, dtype=np.uint8)
        np.save(mask_cache_path, masks)
    else:
        masks = None

    return images, masks, depths, ids


def get_dataloaders(debug=Config.DEBUG, load_cached=True):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Handles data loading, global depth normalization, and transform definition.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached (bool): Whether to use cached .npy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if debug:
        print(f"Debug mode: Reducing dataset size to {Config.DEBUG_SIZE}")
        df_train = df_train.iloc[: Config.DEBUG_SIZE]
        df_val = df_val.iloc[: Config.DEBUG_SIZE]
        df_test = df_test.iloc[: Config.DEBUG_SIZE]

    # 2. Load Data Arrays (with Caching)
    train_imgs, train_masks, train_depths, train_ids = cache_and_load_data(
        df_train, "train", load_cached
    )
    val_imgs, val_masks, val_depths, val_ids = cache_and_load_data(
        df_val, "val", load_cached
    )
    test_imgs, _, test_depths, test_ids = cache_and_load_data(
        df_test, "test", load_cached
    )

    # Merge Train and Val for 90/10 Split (Maximize Data Utilization)
    if not debug:
        print("Merging Train and Val sets for 90/10 split...")
        all_imgs = np.concatenate([train_imgs, val_imgs])
        all_masks = np.concatenate([train_masks, val_masks])
        all_depths = np.concatenate([train_depths, val_depths])
        all_ids = np.concatenate([train_ids, val_ids])

        # Merge DataFrames to get stratification labels
        df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

        # Stratified Split
        train_idx, val_idx = train_test_split(
            np.arange(len(all_imgs)),
            test_size=0.1,
            random_state=Config.SEED,
            stratify=df_all["coverage_class"],
        )

        train_imgs, val_imgs = all_imgs[train_idx], all_imgs[val_idx]
        train_masks, val_masks = all_masks[train_idx], all_masks[val_idx]
        train_depths, val_depths = all_depths[train_idx], all_depths[val_idx]
        train_ids, val_ids = all_ids[train_idx], all_ids[val_idx]

    # 3. Global Depth Normalization
    # Compute min/max across entire dataset to ensure consistency
    all_depths = np.concatenate([train_depths, val_depths, test_depths])

    # Handle NaNs (e.g. missing test depths) to prevent propagation to train/val
    if np.isnan(all_depths).any():
        print("Warning: NaNs detected in depth data. Filling with mean value.")
        mean_depth = np.nanmean(all_depths)
        train_depths = np.nan_to_num(train_depths, nan=mean_depth)
        val_depths = np.nan_to_num(val_depths, nan=mean_depth)
        test_depths = np.nan_to_num(test_depths, nan=mean_depth)
        # Update all_depths for min/max calculation
        all_depths = np.concatenate([train_depths, val_depths, test_depths])

    min_depth = all_depths.min()
    max_depth = all_depths.max()

    # Avoid division by zero
    denom = (max_depth - min_depth) if (max_depth - min_depth) > 0 else 1.0

    train_depths = (train_depths - min_depth) / denom
    val_depths = (val_depths - min_depth) / denom
    test_depths = (test_depths - min_depth) / denom

    # 4. Define Transforms
    # Train: Horizontal Flip + ToTensor
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    # Val/Test: ToTensor only
    val_transform = A.Compose(
        [
            ToTensorV2(),
        ]
    )

    # 5. Create Dataset Instances
    train_dataset = SaltDataset(
        train_imgs,
        train_masks,
        train_depths,
        train_ids,
        transforms=train_transform,
        mode="train",
    )

    val_dataset = SaltDataset(
        val_imgs,
        val_masks,
        val_depths,
        val_ids,
        transforms=val_transform,
        mode="val",
    )

    test_dataset = SaltDataset(
        test_imgs,
        None,
        test_depths,
        test_ids,
        transforms=val_transform,
        mode="test",
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
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
