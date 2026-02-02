import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from library.config import Config
from library.utils import rle_decode


def pad_image(img, target_height=Config.IMG_HEIGHT, target_width=Config.IMG_WIDTH):
    """
    Applies reflection padding to resize image from (101, 101) to (128, 128).
    """
    h, w = img.shape[:2]
    pad_h = target_height - h
    pad_w = target_width - w

    if pad_h < 0 or pad_w < 0:
        return cv2.resize(img, (target_width, target_height))

    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Use Reflection Padding (101 variant avoids repeating the edge pixel twice)
    return cv2.copyMakeBorder(
        img, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_REFLECT_101
    )


def load_and_cache_data(csv_path, cache_prefix, load_cached_data=True):
    """
    Loads data from CSV, applies padding, and caches to numpy files.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache paths
    paths = {
        "images": os.path.join(cache_dir, f"{cache_prefix}_images.npy"),
        "masks": os.path.join(cache_dir, f"{cache_prefix}_masks.npy"),
        "depths": os.path.join(cache_dir, f"{cache_prefix}_depths.npy"),
        "ids": os.path.join(cache_dir, f"{cache_prefix}_ids.npy"),
        "covs": os.path.join(cache_dir, f"{cache_prefix}_covs.npy"),
    }

    # Try loading from cache
    if load_cached_data and os.path.exists(paths["images"]):
        images = np.load(paths["images"])
        depths = np.load(paths["depths"])
        ids = np.load(paths["ids"], allow_pickle=True)
        masks = np.load(paths["masks"]) if os.path.exists(paths["masks"]) else None
        covs = np.load(paths["covs"]) if os.path.exists(paths["covs"]) else None
        return images, masks, depths, ids, covs

    # Load from source
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    img_list = []
    mask_list = []
    depth_list = []
    id_list = []
    cov_list = []

    has_masks = "rle_mask" in df.columns

    for _, row in df.iterrows():
        # Load Image
        full_img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Create empty placeholder if file missing (should not happen based on validation)
            img = np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)

        # Pad Image
        img_padded = pad_image(img)
        img_list.append(img_padded)

        # Load Mask if exists
        if has_masks:
            if pd.isna(row["rle_mask"]):
                mask = np.zeros((Config.ORIG_HEIGHT, Config.ORIG_WIDTH), dtype=np.uint8)
            else:
                mask = rle_decode(row["rle_mask"])

            mask_padded = pad_image(mask)
            mask_list.append(mask_padded)

            # Coverage class for stratification
            cov_list.append(row.get("coverage_class", 0))

        depth_list.append(row["z"])
        id_list.append(row["id"])

    # Convert to numpy arrays
    images = np.array(img_list, dtype=np.uint8)
    depths = np.array(depth_list, dtype=np.float32)
    ids = np.array(id_list)

    # Save cache
    np.save(paths["images"], images)
    np.save(paths["depths"], depths)
    np.save(paths["ids"], ids)

    if has_masks:
        masks = np.array(mask_list, dtype=np.uint8)
        covs = np.array(cov_list, dtype=np.int32)
        np.save(paths["masks"], masks)
        np.save(paths["covs"], covs)
    else:
        masks = None
        covs = None

    return images, masks, depths, ids, covs


class SaltDataset(Dataset):
    def __init__(self, images, masks, depths, ids, mode="train", transform=False):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.ids = ids
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]
        depth = self.depths[idx]
        id_ = self.ids[idx]

        # Normalize Image (0-255 -> 0.0-1.0)
        img = img.astype(np.float32) / 255.0

        # Handle Mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx].astype(np.float32)  # Binary 0.0 or 1.0

        # Augmentation (Horizontal Flip)
        if self.mode == "train" and self.transform and Config.AUGMENTATION_HFLIP:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                if mask is not None:
                    mask = np.flip(mask, axis=1).copy()

        # Convert to Tensor (Add channel dimension)
        # Image: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)

        # Depth: (1,)
        depth_tensor = torch.tensor([depth], dtype=torch.float32)

        if mask is not None:
            # Mask: (H, W) -> (1, H, W)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            return img_tensor, mask_tensor, depth_tensor, id_

        return img_tensor, depth_tensor, id_


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Val, and Test.
    Merges original train/val splits and re-splits 90/10.
    """
    # 1. Load and Cache Data
    # Load original train split (80%)
    train_imgs, train_masks, train_depths, train_ids, train_covs = load_and_cache_data(
        Config.TRAIN_CSV, "train_split", load_cached_data
    )

    # Load original val split (20%)
    val_imgs, val_masks, val_depths, val_ids, val_covs = load_and_cache_data(
        Config.VAL_CSV, "val_split", load_cached_data
    )

    # Load test set
    test_imgs, _, test_depths, test_ids, _ = load_and_cache_data(
        Config.TEST_CSV, "test", load_cached_data
    )

    # 2. Merge Train and Val for Stratified Re-split (90/10)
    all_imgs = np.concatenate([train_imgs, val_imgs], axis=0)
    all_masks = np.concatenate([train_masks, val_masks], axis=0)
    all_depths = np.concatenate([train_depths, val_depths], axis=0)
    all_ids = np.concatenate([train_ids, val_ids], axis=0)
    all_covs = np.concatenate([train_covs, val_covs], axis=0)

    # 3. Stratified Split (90/10)
    # Using random_state from Config for reproducibility
    indices = np.arange(len(all_imgs))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=Config.SEED, stratify=all_covs
    )

    # 4. Debugging Subsample
    if Config.DEBUG:
        train_idx = train_idx[: Config.DEBUG_SAMPLE_SIZE]
        val_idx = val_idx[: Config.DEBUG_SAMPLE_SIZE]
        test_imgs = test_imgs[: Config.DEBUG_SAMPLE_SIZE]
        test_depths = test_depths[: Config.DEBUG_SAMPLE_SIZE]
        test_ids = test_ids[: Config.DEBUG_SAMPLE_SIZE]

    # 5. Create Datasets
    train_dataset = SaltDataset(
        all_imgs[train_idx],
        all_masks[train_idx],
        all_depths[train_idx],
        all_ids[train_idx],
        mode="train",
        transform=True,
    )

    val_dataset = SaltDataset(
        all_imgs[val_idx],
        all_masks[val_idx],
        all_depths[val_idx],
        all_ids[val_idx],
        mode="val",
        transform=False,
    )

    test_dataset = SaltDataset(
        test_imgs, None, test_depths, test_ids, mode="test", transform=False
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Print Statistics
    print(f"Data Loaded Successfully:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader
