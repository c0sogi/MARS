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

# -----------------------------------------------------------------------------
# Augmentation Pipeline
# -----------------------------------------------------------------------------


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.
    Adapts ImageNet stats for 1-channel input.
    """
    # Calculate 1-channel mean/std from ImageNet 3-channel stats
    # We average the RGB values to get a reasonable grayscale prior
    mean = [np.mean(Config.IMAGENET_MEAN)]
    std = [np.mean(Config.IMAGENET_STD)]

    if mode == "train" or mode == "pseudo_train":
        return A.Compose(
            [
                # Spatial Alignment: Pad 101 -> 128 with Reflection
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                # Mandatory Non-Rigid Transform (Key to Strategy)
                A.ElasticTransform(
                    alpha=Config.ELASTIC_ALPHA,
                    sigma=Config.ELASTIC_SIGMA,
                    alpha_affine=Config.ELASTIC_ALPHA_AFFINE,
                    p=1.0,
                ),
                # Rigid Transform
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=Config.SHIFT_SCALE_ROTATE_P,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                # Normalization & Tensor Conversion
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=Config.IMG_SIZE,
                    min_width=Config.IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                    always_apply=True,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


# -----------------------------------------------------------------------------
# Data Preparation & Caching
# -----------------------------------------------------------------------------


def prepare_train_data(load_cached_data=True):
    """
    Loads train/val metadata, merges them, handles depth normalization,
    creates 5-fold splits, and caches the result.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "train_data_processed.parquet")
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")

    if load_cached_data and os.path.exists(cache_path) and os.path.exists(stats_path):
        return pd.read_parquet(cache_path)

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Combine for K-Fold splitting and global depth stats
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Depth Normalization
    # "Apply Standard Scaling (zero mean, unit variance) to the training depth values"
    depth_mean = df["z"].mean()
    depth_std = df["z"].std()

    df["z_norm"] = (df["z"] - depth_mean) / depth_std

    # Create Folds
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )
    df["fold"] = -1
    # Stratify by coverage_class to maintain salt distribution balance
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["coverage_class"])):
        df.loc[val_idx, "fold"] = fold_idx

    # Cache Data and Stats
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    stats = pd.DataFrame({"mean": [depth_mean], "std": [depth_std]})
    stats.to_csv(stats_path, index=False)

    return df


def prepare_test_data(load_cached_data=True):
    """
    Loads test metadata and applies depth normalization using training stats.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "test_data_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    df = pd.read_csv(Config.TEST_CSV)

    # Load depth stats from training set
    stats_path = os.path.join(Config.CACHE_DIR, "depth_stats.csv")
    if not os.path.exists(stats_path):
        # Ensure training data is prepared to get stats
        _ = prepare_train_data(load_cached_data=load_cached_data)

    stats = pd.read_csv(stats_path)
    depth_mean = stats["mean"].iloc[0]
    depth_std = stats["std"].iloc[0]

    df["z_norm"] = (df["z"] - depth_mean) / depth_std

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class SaltDataset(Dataset):
    def __init__(self, df, mode="train", pseudo_labels=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', 'test', 'pseudo_train'.
            pseudo_labels (dict, optional): Dict {id: mask_array} for soft labels.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.pseudo_labels = pseudo_labels
        self.transform = get_transforms(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["id"]

        # 1. Load Image
        # Load as grayscale (H, W)
        image_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for robustness, though metadata check should prevent this
            image = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.uint8)

        # Expand to (H, W, 1) for Albumentations consistency
        image = np.expand_dims(image, axis=-1)

        # 2. Load Mask (if applicable)
        mask = None
        if self.mode in ["train", "val"]:
            # Load Ground Truth
            if "mask_path" in row and pd.notna(row["mask_path"]):
                mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    # Binarize (0 or 255 -> 0 or 1)
                    mask = (mask_img > 0).astype(np.float32)

            if mask is None and "rle_mask" in row and pd.notna(row["rle_mask"]):
                # Decode RLE
                mask_img = rle_decode(row["rle_mask"])
                mask = (mask_img > 0).astype(np.float32)

            if mask is None:
                # Empty mask
                mask = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.float32)

        elif self.mode == "pseudo_train" and self.pseudo_labels is not None:
            # Load Soft Label (Probabilities)
            if image_id in self.pseudo_labels:
                mask = self.pseudo_labels[image_id]  # Expected to be (H, W) float
            else:
                mask = np.zeros((Config.ORIG_SIZE, Config.ORIG_SIZE), dtype=np.float32)

        # 3. Apply Augmentations
        if mask is not None:
            # Albumentations expects mask to be (H, W) or (H, W, C)
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

            # Ensure mask is (1, H, W) for PyTorch
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)
        else:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 4. Load Depth
        # Use normalized depth
        depth = row["z_norm"]
        depth = torch.tensor([depth], dtype=torch.float32)

        # Return
        if mask is not None:
            return image, mask, depth, image_id
        else:
            return image, depth, image_id


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def get_dataloaders(fold=0, load_cached_data=True, pseudo_labels=None):
    """
    Creates train and val dataloaders for a specific fold.
    If pseudo_labels is provided, the training set uses 'pseudo_train' mode.
    """
    # Load Data
    df = prepare_train_data(load_cached_data=load_cached_data)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SIZE)

    # Split
    train_df = df[df["fold"] != fold]
    val_df = df[df["fold"] == fold]

    # Determine mode for training set
    train_mode = "pseudo_train" if pseudo_labels is not None else "train"

    # Datasets
    train_ds = SaltDataset(train_df, mode=train_mode, pseudo_labels=pseudo_labels)
    val_ds = SaltDataset(val_df, mode="val")

    # Loaders
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


def get_test_dataloader(load_cached_data=True):
    """
    Creates dataloader for the test set.
    """
    df = prepare_test_data(load_cached_data=load_cached_data)

    if Config.DEBUG:
        df = df.head(Config.DEBUG_SIZE)

    ds = SaltDataset(df, mode="test")

    loader = DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader
