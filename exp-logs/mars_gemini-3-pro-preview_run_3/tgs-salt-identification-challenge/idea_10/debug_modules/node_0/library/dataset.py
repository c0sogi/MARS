import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from library.config import Config
from library.utils import pad_image, rle_decode


def load_dataset_arrays(mode, load_cached_data=True):
    """
    Loads dataset arrays from cache or processes them from scratch.

    Args:
        mode (str): 'train' (includes val) or 'test'.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for images, masks, depths, ids, etc.
    """
    cache_prefix = "train_val" if mode in ["train", "val"] else "test"

    # Define cache paths
    cache_paths = {
        "images": os.path.join(Config.WORK_DIR, f"cached_{cache_prefix}_images.npy"),
        "masks": os.path.join(Config.WORK_DIR, f"cached_{cache_prefix}_masks.npy"),
        "depths": os.path.join(Config.WORK_DIR, f"cached_{cache_prefix}_depths.npy"),
        "ids": os.path.join(Config.WORK_DIR, f"cached_{cache_prefix}_ids.npy"),
        "coverage_classes": os.path.join(
            Config.WORK_DIR, f"cached_{cache_prefix}_coverage_classes.npy"
        ),
    }

    # Try loading from cache
    if load_cached_data:
        all_exist = all(
            os.path.exists(p)
            for p in cache_paths.values()
            if "masks" not in p or mode != "test"
        )
        # For test, we don't need masks and coverage_classes
        if mode == "test":
            all_exist = (
                os.path.exists(cache_paths["images"])
                and os.path.exists(cache_paths["depths"])
                and os.path.exists(cache_paths["ids"])
            )

        if all_exist:
            data = {}
            data["images"] = np.load(cache_paths["images"])
            data["depths"] = np.load(cache_paths["depths"])
            data["ids"] = np.load(cache_paths["ids"])

            if mode != "test":
                data["masks"] = np.load(cache_paths["masks"])
                data["coverage_classes"] = np.load(cache_paths["coverage_classes"])

            return data

    # Process from scratch
    if mode in ["train", "val"]:
        # Merge train and val metadata for consolidated CV
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df = pd.concat([df_train, df_val], ignore_index=True)
    else:
        df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle DEBUG mode
    if Config.DEBUG:
        df = df.iloc[:50]  # Small subset for debugging

    images = []
    masks = []
    depths = []
    ids = []
    coverage_classes = []

    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing images (should not happen based on metadata check)
            img = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

        # Pad Image to 128x128
        img_padded = pad_image(img, target_size=Config.IMG_SIZE)
        images.append(img_padded)

        # Store Depth and ID
        depths.append(row["z"])
        ids.append(row["id"])

        if mode != "test":
            # Load and Decode Mask
            rle = row["rle_mask"]
            mask = rle_decode(rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE))

            # Pad Mask
            mask_padded = pad_image(mask, target_size=Config.IMG_SIZE)
            masks.append(mask_padded)

            # Store Coverage Class
            coverage_classes.append(row["coverage_class"])

    # Convert to numpy arrays
    images_arr = np.array(images, dtype=np.uint8)
    depths_arr = np.array(depths, dtype=np.int32)
    ids_arr = np.array(ids)

    data = {"images": images_arr, "depths": depths_arr, "ids": ids_arr}

    # Save to cache
    np.save(cache_paths["images"], images_arr)
    np.save(cache_paths["depths"], depths_arr)
    np.save(cache_paths["ids"], ids_arr)

    if mode != "test":
        masks_arr = np.array(masks, dtype=np.uint8)
        coverage_classes_arr = np.array(coverage_classes, dtype=np.int32)

        data["masks"] = masks_arr
        data["coverage_classes"] = coverage_classes_arr

        np.save(cache_paths["masks"], masks_arr)
        np.save(cache_paths["coverage_classes"], coverage_classes_arr)

    return data


class SaltDataset(Dataset):
    def __init__(
        self, mode="train", fold_index=0, n_folds=Config.FOLDS, load_cached_data=True
    ):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            fold_index (int): Index of the fold (0 to n_folds-1).
            n_folds (int): Total number of folds.
            load_cached_data (bool): Whether to use cached data.
        """
        self.mode = mode
        self.fold_index = fold_index
        self.n_folds = n_folds

        # Load data
        data = load_dataset_arrays(mode, load_cached_data)

        self.images = data["images"]
        self.depths = data["depths"]
        self.ids = data["ids"]

        if mode != "test":
            self.masks = data["masks"]
            self.coverage_classes = data["coverage_classes"]

            # Perform Stratified Split
            skf = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=Config.SEED
            )

            # We need a dummy X for split, y is coverage_classes
            # The split returns indices
            folds = list(skf.split(self.images, self.coverage_classes))
            train_idx, val_idx = folds[fold_index]

            # Select indices based on mode
            if mode == "train":
                self.indices = train_idx
            else:
                self.indices = val_idx
        else:
            # Test mode uses all data
            self.indices = np.arange(len(self.images))

        # Define Augmentations
        if mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Map logical index to physical index
        real_idx = self.indices[idx]

        # 1. Get Image (128, 128)
        img = self.images[real_idx]  # uint8

        # 2. Get Depth
        depth_val = self.depths[real_idx]

        # 3. Construct Input Tensor [Seismic, Seismic, Depth]
        # Create depth channel
        depth_channel = np.full_like(img, depth_val, dtype=np.float32)

        # Stack channels: (H, W, 3)
        # Note: We convert img to float32 here to allow stacking with depth
        img_float = img.astype(np.float32)
        input_image = np.dstack([img_float, img_float, depth_channel])

        # 4. Uniform Scaling (Normalize to 0-1)
        # Both seismic (0-255) and depth (50-959) are divided by 255.0
        # This is consistent with the "Idea" description.
        input_image = input_image / 255.0

        # 5. Handle Mask
        mask = None
        if self.mode != "test":
            mask = self.masks[real_idx]  # uint8 (0 or 1)
            mask = mask.astype(np.float32)
            # Add channel dimension to mask for Albumentations: (H, W, 1)
            # But Albumentations expects (H, W) for mask usually, or we pass it as 'mask'
            # ToTensorV2 will convert it.

        # 6. Augmentation
        if self.mode == "train":
            augmented = self.transform(image=input_image.astype(np.float32), mask=mask)
            input_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0)  # Add channel dim: (1, H, W)
        elif self.mode == "val":
            augmented = self.transform(image=input_image.astype(np.float32), mask=mask)
            input_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0)
        else:
            augmented = self.transform(image=input_image.astype(np.float32))
            input_tensor = augmented["image"]

        # Return
        if self.mode != "test":
            return input_tensor, mask_tensor, self.ids[real_idx]
        else:
            return input_tensor, self.ids[real_idx]


def get_stratified_folds(n_folds=Config.FOLDS):
    """
    Helper to verify fold distribution or debug.
    Not strictly needed for the Dataset class but useful for external analysis.
    """
    data = load_dataset_arrays("train")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
    return list(skf.split(data["images"], data["coverage_classes"]))
