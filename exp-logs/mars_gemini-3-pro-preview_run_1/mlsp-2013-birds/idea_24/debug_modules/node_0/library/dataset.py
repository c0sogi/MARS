import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    PyTorch Dataset for Bird Species Classification.
    Handles dynamic loading of spectrograms, resizing, channel replication,
    and applying augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, species columns).
            transforms (A.Compose): Albumentations transform pipeline.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in self.df.columns if c.startswith("species_")]

        # Pre-calculate full paths to avoid overhead in __getitem__
        self.image_paths = []
        for rel_path in self.df["file_path"]:
            # Map .wav path from metadata to .bmp path in spectrograms dir
            # Example: essential_data/src_wavs/PC10_...wav -> PC10_...bmp
            basename = os.path.basename(rel_path)
            bmp_name = os.path.splitext(basename)[0] + ".bmp"
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)
            self.image_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.image_paths[idx]

        # Load as grayscale (1 channel)
        # Dynamic loading: read from disk every time
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should not happen based on EDA)
            # Create a blank black image of correct size
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # 2. Resize to High-Fidelity Resolution (256x640)
        image = cv2.resize(
            image, (Config.IMG_WIDTH, Config.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )

        # 3. Channel Replication (1 -> 3 channels)
        # Converts Grayscale to RGB by replicating the single channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 4. Apply Augmentations / Normalization
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # 5. Get Labels
        if self.mode in ["train", "val"]:
            labels = self.df.iloc[idx][self.label_cols].values.astype(np.float32)
            return image, torch.tensor(labels)
        else:
            # Test mode: return image and rec_id for submission tracking
            rec_id = self.df.iloc[idx]["rec_id"]
            return image, torch.tensor(rec_id)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.
    """
    if mode == "train":
        return A.Compose(
            [
                # Augmentation: Horizontal Flip (Time Inversion)
                A.HorizontalFlip(p=0.5) if Config.TTA_FLIP else A.NoOp(),
                # Augmentation: Unstructured Cutout (CoarseDropout)
                # Masking out rectangular regions to encourage robustness
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization (ImageNet stats)
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Inference
        return A.Compose(
            [
                # Normalization (ImageNet stats)
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                # Convert to Tensor
                ToTensorV2(),
            ]
        )


class Mixup:
    """
    Implements Mixup regularization.
    Mixes inputs and targets with a coefficient lambda sampled from Beta(alpha, alpha).
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.rng = np.random.default_rng(Config.SEED)

    def __call__(self, batch_x, batch_y):
        """
        Args:
            batch_x (torch.Tensor): Input batch images.
            batch_y (torch.Tensor): Input batch labels.

        Returns:
            mixed_x (torch.Tensor): Mixed images.
            y_a (torch.Tensor): Original labels.
            y_b (torch.Tensor): Permuted labels.
            lam (float): Mixing coefficient.
        """
        if self.alpha > 0:
            lam = self.rng.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)

        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]

        return mixed_x, y_a, y_b, lam


def load_data(load_cached_data=True):
    """
    Loads train, validation, and test metadata DataFrames.
    Implements caching using Parquet files in the working directory.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cached.parquet")
    val_cache = os.path.join(cache_dir, "val_cached.parquet")
    test_cache = os.path.join(cache_dir, "test_cached.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df

    # 2. Load from source metadata
    print("Loading data from source metadata...")
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # 3. Handle Debug Mode (Subset data)
    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Subsetting data to {Config.DEBUG_SUBSET_SIZE} samples."
        )
        train_df = train_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SUBSET_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 4. Cache the processed dataframes
    print("Caching dataframes...")
    train_df.to_parquet(train_cache)
    val_df.to_parquet(val_cache)
    test_df.to_parquet(test_cache)

    return train_df, val_df, test_df
