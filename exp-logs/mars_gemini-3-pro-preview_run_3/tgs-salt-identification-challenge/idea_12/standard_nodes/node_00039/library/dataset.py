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
from library.utils import seed_everything

# Set seeds for reproducibility
seed_everything(Config.SEED)


class SaltDataset(Dataset):
    """
    PyTorch Dataset for Salt Segmentation.
    Handles 3-channel input construction [Seismic, Seismic, Depth] and augmentation.
    """

    def __init__(
        self, images, depths, masks=None, ids=None, phase="train", transforms=None
    ):
        """
        Args:
            images (np.ndarray): Array of grayscale images (N, H, W).
            depths (np.ndarray): Array of depth values (N,).
            masks (np.ndarray, optional): Array of binary masks (N, H, W).
            ids (list, optional): List of image IDs.
            phase (str): 'train', 'val', or 'test'.
            transforms (A.Compose, optional): Albumentations transforms.
        """
        self.images = images
        self.depths = depths
        self.masks = masks
        self.ids = ids
        self.phase = phase
        self.transforms = transforms

        # Pixel-level transforms for the seismic channel only (Train only)
        self.pixel_transforms = None
        if self.phase == "train":
            self.pixel_transforms = A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load base data
        img = self.images[idx]  # (101, 101), uint8
        z = self.depths[idx]  # scalar, float (normalized)

        # 2. Apply pixel-level augmentations to seismic image (before stacking)
        if self.pixel_transforms:
            augmented = self.pixel_transforms(image=img)
            img = augmented["image"]

        # 3. Normalize Image to [0, 1]
        img = img.astype(np.float32) / 255.0

        # 4. Construct Depth Plane (already normalized in loading step)
        # Shape (101, 101)
        depth_plane = np.full_like(img, z, dtype=np.float32)

        # 5. Stack to create 3-channel input: [Seismic, Seismic, Depth]
        # Shape (101, 101, 3)
        input_vol = np.dstack([img, img, depth_plane])

        # 6. Prepare Mask
        mask = None
        if self.masks is not None:
            mask = self.masks[idx].astype(np.float32)
            # Ensure mask is (H, W)
            if mask.ndim == 3:
                mask = mask.squeeze()

        # 7. Apply Spatial Transforms (Padding, Flips, etc.)
        # Albumentations expects image as (H, W, C)
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=input_vol, mask=mask)
                input_vol = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=input_vol)
                input_vol = augmented["image"]

        # 8. Post-processing
        # ToTensorV2 converts to (C, H, W)
        # input_vol is already a tensor from ToTensorV2

        # If mask exists, add channel dim: (H, W) -> (1, H, W)
        if mask is not None:
            mask = mask.unsqueeze(0)
            return input_vol, mask

        # For test set, return image and ID
        image_id = self.ids[idx] if self.ids is not None else str(idx)
        return input_vol, image_id


def get_transforms(phase):
    """
    Factory for Albumentations transforms.
    """
    # Common: Pad to 128x128 using Reflection
    # Note: ToTensorV2 converts HWC to CHW

    common_transforms = [
        A.PadIfNeeded(
            min_height=Config.IMG_SIZE,
            min_width=Config.IMG_SIZE,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        )
    ]

    if phase == "train":
        # Conservative Augmentations for Training
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # Conservative ShiftScaleRotate
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
                *common_transforms,
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose([*common_transforms, ToTensorV2()])


def _load_and_cache_data(mode, load_cached_data):
    """
    Internal function to load data from CSV/Images or Cache.
    mode: 'train_val' or 'test'
    """
    cache_files = {
        "images": os.path.join(Config.CACHE_DIR, f"cached_{mode}_images.npy"),
        "masks": os.path.join(Config.CACHE_DIR, f"cached_{mode}_masks.npy"),
        "depths": os.path.join(Config.CACHE_DIR, f"cached_{mode}_depths.npy"),
        "ids": os.path.join(Config.CACHE_DIR, f"cached_{mode}_ids.npy"),
        "classes": os.path.join(
            Config.CACHE_DIR, f"cached_{mode}_coverage_classes.npy"
        ),
    }

    # 1. Try Loading Cache
    if load_cached_data:
        all_exist = True
        # For test, we don't need masks/classes
        required_keys = ["images", "depths", "ids"]
        if mode == "train_val":
            required_keys.extend(["masks", "classes"])

        for k in required_keys:
            if not os.path.exists(cache_files[k]):
                all_exist = False
                break

        if all_exist:
            print(f"Loading {mode} data from cache...")
            data = {}
            for k in required_keys:
                data[k] = np.load(cache_files[k], allow_pickle=True)
            return data

    # 2. Process from Scratch
    print(f"Processing {mode} data from source...")

    if mode == "train_val":
        # Merge Train and Val metadata
        df_train = pd.read_csv(Config.TRAIN_METADATA)
        df_val = pd.read_csv(Config.VAL_METADATA)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = pd.read_csv(Config.TEST_METADATA)

    # Pre-allocate arrays
    n_samples = len(df)
    images = np.zeros((n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)
    depths = np.zeros((n_samples,), dtype=np.float32)
    ids = df["id"].values

    masks = None
    classes = None
    if mode == "train_val":
        masks = np.zeros(
            (n_samples, Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8
        )
        classes = df["coverage_class"].values

    # Iterate and Load
    for idx, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        images[idx] = img

        # Load Depth and Normalize (Scale 0-1000 to 0-1)
        # Max depth is ~960, so dividing by 1000 is safe and keeps scale reasonable
        depths[idx] = row["z"] / 1000.0

        # Load Mask (Train/Val only)
        if mode == "train_val":
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            # Binarize just in case
            masks[idx] = (mask > 127).astype(np.uint8)

    # 3. Save to Cache
    print(f"Saving {mode} data to cache...")
    np.save(cache_files["images"], images)
    np.save(cache_files["depths"], depths)
    np.save(cache_files["ids"], ids)

    data = {"images": images, "depths": depths, "ids": ids}

    if mode == "train_val":
        np.save(cache_files["masks"], masks)
        np.save(cache_files["classes"], classes)
        data["masks"] = masks
        data["classes"] = classes

    return data


def get_fold_loaders(fold_idx, load_cached_data=True):
    """
    Returns train and validation DataLoaders for a specific fold.
    Uses StratifiedKFold on the combined train+val dataset.
    """
    # Load merged data
    data = _load_and_cache_data("train_val", load_cached_data)

    images = data["images"]
    masks = data["masks"]
    depths = data["depths"]
    ids = data["ids"]
    classes = data["classes"]

    # Stratified Split
    skf = StratifiedKFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)

    # We need a dummy x for splitting
    dummy_x = np.zeros(len(classes))
    splits = list(skf.split(dummy_x, classes))

    train_idx, val_idx = splits[fold_idx]

    # Create Datasets
    train_dataset = SaltDataset(
        images=images[train_idx],
        depths=depths[train_idx],
        masks=masks[train_idx],
        ids=ids[train_idx],
        phase="train",
        transforms=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        images=images[val_idx],
        depths=depths[val_idx],
        masks=masks[val_idx],
        ids=ids[val_idx],
        phase="val",
        transforms=get_transforms("val"),
    )

    # Create Loaders
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

    print(
        f"Fold {fold_idx}: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
    )

    return train_loader, val_loader


def get_test_loader(load_cached_data=True):
    """
    Returns DataLoader for the test set.
    """
    data = _load_and_cache_data("test", load_cached_data)

    test_dataset = SaltDataset(
        images=data["images"],
        depths=data["depths"],
        masks=None,
        ids=data["ids"],
        phase="test",
        transforms=get_transforms("test"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Test samples: {len(test_dataset)}")

    return test_loader
