import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

from library.config import PathConfig, DataConfig, GeneralConfig, TrainConfig


class SaltDataset(Dataset):
    def __init__(self, images, depths, ids, masks=None, transform=None, mode="train"):
        """
        Args:
            images (np.array): Array of images (N, H, W).
            depths (np.array): Array of depths (N,).
            ids (np.array): Array of IDs (N,).
            masks (np.array, optional): Array of masks (N, H, W).
            transform (A.Compose, optional): Albumentations pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.images = images
        self.depths = depths
        self.ids = ids
        self.masks = masks
        self.transform = transform
        self.mode = mode

        # Pre-calculate depth min/max for normalization (using global stats from dataset)
        # We use fixed reasonable bounds to avoid data leakage or batch-dependency
        # Depth range in CSV is approx 50 to 960.
        self.depth_min = 0.0
        self.depth_max = 1000.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (H, W) - uint8
        image = self.images[idx]
        depth_val = self.depths[idx]
        image_id = self.ids[idx]

        if self.mode != "test" and self.masks is not None:
            mask = self.masks[idx]
        else:
            # Placeholder mask for test set
            mask = np.zeros_like(image)

        # Apply spatial augmentations (Padding, Flip, Rotate, etc.)
        # We pass both image and mask to ensure geometric consistency
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Construct 3-Channel Input: [Seismic, Seismic, Depth]
        # 1. Normalize Image to [0, 1]
        image = image.astype(np.float32) / 255.0

        # 2. Normalize Depth to [0, 1]
        depth_norm = (depth_val - self.depth_min) / (self.depth_max - self.depth_min)
        depth_channel = np.full_like(image, depth_norm, dtype=np.float32)

        # 3. Stack channels
        # Shape becomes (H, W, 3)
        input_tensor = np.stack([image, image, depth_channel], axis=-1)

        # 4. Apply ImageNet Normalization manually
        # Albumentations Normalize is usually applied before ToTensor, but we constructed channels manually.
        # We apply the standard mean/std normalization here.
        mean = np.array(DataConfig.MEAN, dtype=np.float32)
        std = np.array(DataConfig.STD, dtype=np.float32)
        input_tensor = (input_tensor - mean) / std

        # 5. Convert to Tensor (C, H, W)
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = torch.from_numpy(input_tensor).float()

        # Process Mask
        # Expand dims to (1, H, W) and float
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask).float()

        return input_tensor, mask, image_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    Note: Normalization and ToTensor are handled manually in Dataset to support custom channel construction.
    """
    transforms_list = []

    # 1. Reflection Padding: 101x101 -> 128x128
    # We use PadIfNeeded. border_mode=cv2.BORDER_REFLECT_101 ensures texture continuity.
    transforms_list.append(
        A.PadIfNeeded(
            min_height=DataConfig.IMG_H,
            min_width=DataConfig.IMG_W,
            border_mode=cv2.BORDER_REFLECT_101,
            value=0,
            mask_value=0,
            always_apply=True,
        )
    )

    if mode == "train":
        # 2. Horizontal Flip
        transforms_list.append(A.HorizontalFlip(p=0.5))

        # 3. ShiftScaleRotate (Conservative)
        transforms_list.append(
            A.ShiftScaleRotate(
                shift_limit=DataConfig.SHIFT_LIMIT,
                scale_limit=DataConfig.SCALE_LIMIT,
                rotate_limit=DataConfig.ROTATION_LIMIT,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            )
        )

        # 4. Random Brightness/Contrast
        transforms_list.append(
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5)
        )

    return A.Compose(transforms_list)


def load_data(mode="train", load_cached_data=True, debug=False):
    """
    Loads data from disk or cache.
    mode: 'train' (loads merged train+val) or 'test'.
    """
    cache_dir = PathConfig.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    prefix = "train" if mode == "train" else "test"
    if debug:
        prefix += "_debug"

    cache_files = {
        "images": os.path.join(cache_dir, f"cached_{prefix}_images.npy"),
        "masks": os.path.join(cache_dir, f"cached_{prefix}_masks.npy"),
        "depths": os.path.join(cache_dir, f"cached_{prefix}_depths.npy"),
        "ids": os.path.join(cache_dir, f"cached_{prefix}_ids.npy"),
        "coverage": os.path.join(
            cache_dir, f"cached_{prefix}_coverage.npy"
        ),  # Only for train
    }

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(
            os.path.exists(f)
            for k, f in cache_files.items()
            if k != "coverage" or mode == "train"
        )
        if mode == "test":
            # Test doesn't have masks usually, but we cache placeholder or skip
            # We don't need coverage for test
            pass

        if all_exist:
            print(f"Loading {mode} data from cache...")
            images = np.load(cache_files["images"])
            depths = np.load(cache_files["depths"])
            ids = np.load(cache_files["ids"])

            masks = None
            coverage_classes = None

            if mode == "train":
                masks = np.load(cache_files["masks"])
                coverage_classes = np.load(cache_files["coverage"])

            return images, masks, depths, ids, coverage_classes

    print(f"Processing {mode} data from scratch...")

    # Load Metadata
    if mode == "train":
        # Consolidate Train and Val metadata
        df_train = pd.read_csv(PathConfig.TRAIN_METADATA)
        df_val = pd.read_csv(PathConfig.VAL_METADATA)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = pd.read_csv(PathConfig.TEST_METADATA)

    if debug:
        df = df.head(GeneralConfig.DEBUG_DATA_LIMIT)

    # Containers
    images_list = []
    masks_list = []
    depths_list = []
    ids_list = []
    coverage_list = []

    for _, row in df.iterrows():
        img_id = row["id"]

        # Load Image
        img_path = os.path.join(PathConfig.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        images_list.append(img)
        depths_list.append(row["z"])
        ids_list.append(img_id)

        if mode == "train":
            # Load Mask
            mask_path = os.path.join(PathConfig.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks_list.append(mask)
            coverage_list.append(row["coverage_class"])

    # Convert to Numpy Arrays
    images_np = np.array(images_list, dtype=np.uint8)
    depths_np = np.array(depths_list, dtype=np.float32)
    ids_np = np.array(ids_list)

    masks_np = None
    coverage_np = None

    if mode == "train":
        masks_np = np.array(masks_list, dtype=np.uint8)
        coverage_np = np.array(coverage_list, dtype=np.int32)

    # Save to Cache
    np.save(cache_files["images"], images_np)
    np.save(cache_files["depths"], depths_np)
    np.save(cache_files["ids"], ids_np)

    if mode == "train":
        np.save(cache_files["masks"], masks_np)
        np.save(cache_files["coverage"], coverage_np)

    return images_np, masks_np, depths_np, ids_np, coverage_np


def get_loaders(fold_idx, debug=False, load_cached_data=True):
    """
    Creates train and validation DataLoaders for a specific fold.
    Uses StratifiedKFold on the consolidated dataset.
    """
    # Load all training data (Train + Val merged)
    images, masks, depths, ids, coverage_classes = load_data(
        mode="train", load_cached_data=load_cached_data, debug=debug
    )

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=DataConfig.NUM_FOLDS, shuffle=True, random_state=GeneralConfig.SEED
    )

    # We split based on coverage_classes to ensure balanced salt amounts
    splits = list(skf.split(images, coverage_classes))
    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_dataset = SaltDataset(
        images=images[train_idx],
        depths=depths[train_idx],
        ids=ids[train_idx],
        masks=masks[train_idx],
        transform=get_transforms(mode="train"),
        mode="train",
    )

    val_dataset = SaltDataset(
        images=images[val_idx],
        depths=depths[val_idx],
        ids=ids[val_idx],
        masks=masks[val_idx],
        transform=get_transforms(mode="val"),
        mode="val",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=GeneralConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=GeneralConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(debug=False, load_cached_data=True):
    """
    Creates a DataLoader for the test set.
    """
    images, _, depths, ids, _ = load_data(
        mode="test", load_cached_data=load_cached_data, debug=debug
    )

    test_dataset = SaltDataset(
        images=images,
        depths=depths,
        ids=ids,
        masks=None,
        transform=get_transforms(mode="test"),
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=DataConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=GeneralConfig.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
