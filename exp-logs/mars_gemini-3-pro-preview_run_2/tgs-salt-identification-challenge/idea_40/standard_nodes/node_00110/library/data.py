import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import pad_image

# Dataset-specific statistics (calculated from EDA)
# Mean: ~148/255 = 0.58, Std: ~65/255 = 0.25
DATASET_MEAN = [0.5806]
DATASET_STD = [0.2557]


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline.
    Args:
        mode (str): 'train', 'valid', or 'test'.
    """
    transforms = []

    if mode == "train":
        # Non-Rigid Augmentation (Crucial for Seismic Data)
        transforms.append(
            A.ElasticTransform(
                alpha=Config.AUG_ELASTIC_ALPHA,
                sigma=Config.AUG_ELASTIC_SIGMA,
                p=Config.AUG_ELASTIC_PROB,
                border_mode=cv2.BORDER_REFLECT_101,
            )
        )
        # Rigid Augmentation
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=Config.AUG_RIGID_PROB,
                border_mode=cv2.BORDER_REFLECT_101,
            )
        )
        transforms.append(A.HorizontalFlip(p=0.5))

    # Normalization and Tensor Conversion
    transforms.append(
        A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=255.0)
    )
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def load_cache_or_process(df, cache_prefix, load_cached_data=True):
    """
    Loads data from cache or processes it from scratch.
    Pads images/masks to Config.IMG_SIZE (128x128) before caching.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_prefix (str): Prefix for cache files (e.g., 'train_fold0').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'images', 'masks', 'depths', 'ids'.
    """
    cache_dir = Config.WORKING_DIR
    cache_files = {
        "images": os.path.join(cache_dir, f"{cache_prefix}_images.npy"),
        "masks": os.path.join(cache_dir, f"{cache_prefix}_masks.npy"),
        "depths": os.path.join(cache_dir, f"{cache_prefix}_depths.npy"),
        "ids": os.path.join(cache_dir, f"{cache_prefix}_ids.npy"),
    }

    # 1. Try to load from cache
    if load_cached_data:
        if all(os.path.exists(p) for p in cache_files.values()):
            # print(f"Loading cached data for {cache_prefix}...")
            data = {}
            for k, v in cache_files.items():
                data[k] = np.load(v, allow_pickle=True)
            return data

    # 2. Process from scratch
    # print(f"Processing data for {cache_prefix}...")
    images = []
    masks = []
    depths = []
    ids = []

    # Pre-allocate if possible, but list append is fast enough for 3000 items
    for _, row in df.iterrows():
        # Load Image
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Pad Image (101 -> 128)
        img_padded = pad_image(img)
        images.append(img_padded)

        # Load Mask (if available)
        if "mask_path" in row and pd.notna(row["mask_path"]):
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Threshold to binary 0/1
            mask = (mask > 127).astype(np.uint8)
            mask_padded = pad_image(mask)
            masks.append(mask_padded)
        else:
            # Placeholder for test set
            masks.append(np.zeros_like(img_padded))

        # Depth
        depths.append(row["z"])
        ids.append(row["id"])

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)
    depths = np.array(depths, dtype=np.float32)
    ids = np.array(ids, dtype=object)

    # Save to cache
    np.save(cache_files["images"], images)
    np.save(cache_files["masks"], masks)
    np.save(cache_files["depths"], depths)
    np.save(cache_files["ids"], ids)

    return {"images": images, "masks": masks, "depths": depths, "ids": ids}


class SaltDataset(Dataset):
    """
    Standard Dataset for Labeled Data.
    Used for Teacher Training (Stage 1) and Student Training (Stage 3 - Labeled part).
    """

    def __init__(self, data_dict, transform=None, depth_scaler=None):
        self.images = data_dict["images"]
        self.masks = data_dict["masks"]
        self.depths = data_dict["depths"]
        self.ids = data_dict["ids"]
        self.transform = transform
        self.depth_scaler = depth_scaler

        # Pre-scale depths if scaler provided
        if self.depth_scaler:
            self.depths = self.depth_scaler.transform(
                self.depths.reshape(-1, 1)
            ).flatten()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        depth = self.depths[idx]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Minimal transform if none provided (ToTensor)
            t = ToTensorV2()
            augmented = t(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure mask has channel dim (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Depth to tensor
        depth = torch.tensor([depth], dtype=torch.float32)

        return image, mask, depth, self.ids[idx]


class PseudoDataset(Dataset):
    """
    Dataset for Unlabeled Data with Soft Targets.
    Used for Student Training (Stage 3 - Unlabeled part).
    """

    def __init__(self, data_dict, soft_masks_dict, transform=None, depth_scaler=None):
        self.images = data_dict["images"]
        self.ids = data_dict["ids"]
        self.depths = data_dict[
            "depths"
        ]  # Available but not used for training target in student
        self.soft_masks_dict = soft_masks_dict
        self.transform = transform
        self.depth_scaler = depth_scaler

        if self.depth_scaler:
            self.depths = self.depth_scaler.transform(
                self.depths.reshape(-1, 1)
            ).flatten()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = self.images[idx]

        # Retrieve soft mask (Probability map 0.0-1.0)
        # Soft mask should already be padded to 128x128 by the generation process
        if img_id in self.soft_masks_dict:
            soft_mask = self.soft_masks_dict[img_id]
        else:
            # Fallback (should not happen if logic is correct)
            soft_mask = np.zeros_like(image, dtype=np.float32)

        # Ensure soft_mask is (H, W) for Albumentations
        if soft_mask.ndim == 3 and soft_mask.shape[0] == 1:
            soft_mask = soft_mask[0]

        # Apply transforms
        # Note: Albumentations works with uint8 images mostly, but can handle float masks
        if self.transform:
            # We treat soft_mask as mask.
            # Ensure soft_mask is float32
            augmented = self.transform(image=image, mask=soft_mask)
            image = augmented["image"]
            soft_mask = augmented["mask"]
        else:
            # Fallback to ToTensorV2 if no transform is provided
            t = ToTensorV2()
            augmented = t(image=image, mask=soft_mask)
            image = augmented["image"]
            soft_mask = augmented["mask"]

        # Ensure mask has channel dim
        if soft_mask.ndim == 2:
            soft_mask = soft_mask.unsqueeze(0)

        # Dummy depth (Student doesn't use injected depth, but we return it for consistency if needed)
        depth = torch.tensor([self.depths[idx]], dtype=torch.float32)

        return image, soft_mask, depth, img_id


def get_fold_datasets(fold_idx, load_cached_data=True):
    """
    Prepares Train and Validation datasets for a specific fold.
    Merges train.csv and val.csv metadata, then performs StratifiedKFold.

    Args:
        fold_idx (int): Fold index (0 to N_FOLDS-1).
        load_cached_data (bool): Use cached numpy arrays.

    Returns:
        tuple: (train_dataset, val_dataset, depth_scaler)
    """
    # 1. Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Merge to perform global CV split
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    # 2. Stratified Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We split based on coverage_class
    fold_generator = skf.split(full_df, full_df["coverage_class"])

    train_indices, val_indices = list(fold_generator)[fold_idx]

    train_df = full_df.iloc[train_indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices].reset_index(drop=True)

    # 3. Process/Load Data
    train_data = load_cache_or_process(
        train_df, f"fold_{fold_idx}_train", load_cached_data
    )
    val_data = load_cache_or_process(val_df, f"fold_{fold_idx}_val", load_cached_data)

    # 4. Fit Depth Scaler on Training Data
    scaler = StandardScaler()
    scaler.fit(train_data["depths"].reshape(-1, 1))

    # 5. Create Datasets
    train_ds = SaltDataset(
        train_data, transform=get_transforms("train"), depth_scaler=scaler
    )

    val_ds = SaltDataset(
        val_data, transform=get_transforms("valid"), depth_scaler=scaler
    )

    return train_ds, val_ds, scaler


def get_test_dataset(depth_scaler, load_cached_data=True):
    """
    Prepares the Test dataset.

    Args:
        depth_scaler (StandardScaler): Scaler fitted on training data.
        load_cached_data (bool): Use cached numpy arrays.

    Returns:
        SaltDataset: Test dataset.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    test_data = load_cache_or_process(test_df, "test_data", load_cached_data)

    test_ds = SaltDataset(
        test_data, transform=get_transforms("test"), depth_scaler=depth_scaler
    )

    return test_ds
