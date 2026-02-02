import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import IterativeStratification
from library.config import Config
from library.utils import seed_everything


class BirdDataset(Dataset):
    """
    Custom Dataset for Bird Species Classification.
    Handles loading filtered spectrograms, resizing, and applying augmentations
    specifically designed for spectrograms (Time-Rolling, SpecAugment).
    """

    def __init__(
        self, df, phase="train", img_size=Config.IMG_SIZE_DEFAULT, transform=None
    ):
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.img_size = img_size
        self.transform = transform

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = self.df[self.label_cols].values.astype(np.float32)

        # Pre-calculate paths to avoid doing it in __getitem__
        self.file_paths = []
        for rel_path in self.df["file_path_spec"]:
            # Replace default 'spectrograms' with configured directory (filtered_spectrograms)
            # The metadata path format is "supplemental_data/spectrograms/filename.bmp"
            path_parts = rel_path.split(os.sep)
            if "spectrograms" in path_parts:
                idx = path_parts.index("spectrograms")
                path_parts[idx] = Config.SPECTROGRAM_DIR_NAME

            full_path = os.path.join(Config.INPUT_DIR, *path_parts)
            self.file_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        rec_id = self.df.iloc[idx]["rec_id"]
        labels = self.labels[idx]

        # Load Image
        # cv2.imread returns None if file missing, but we assume metadata is validated
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing files (should not happen based on EDA)
            # Create a blank image
            img = np.zeros(self.img_size, dtype=np.uint8)

        # Resize to target resolution (Freq x Time) -> (Height x Width)
        # cv2.resize expects (Width, Height)
        target_h, target_w = self.img_size
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Augmentations
        if self.phase == "train":
            # 1. Time-Rolling (Circular Shift)
            # Randomly roll along the time axis (width)
            shift = np.random.randint(0, target_w)
            img = np.roll(img, shift, axis=1)

            # 2. SpecAugment (Simple Numpy Implementation)
            # Frequency Masking
            num_freq_masks = 1
            freq_mask_param = target_h // 8
            for _ in range(num_freq_masks):
                f = np.random.randint(0, freq_mask_param)
                f0 = np.random.randint(0, target_h - f)
                img[f0 : f0 + f, :] = 0

            # Time Masking
            num_time_masks = 1
            time_mask_param = target_w // 8
            for _ in range(num_time_masks):
                t = np.random.randint(0, time_mask_param)
                t0 = np.random.randint(0, target_w - t)
                img[:, t0 : t0 + t] = 0

        # Normalization (0-1)
        img = img.astype(np.float32) / 255.0

        # Pseudo-RGB
        # Stack 1-channel to 3-channel
        img = np.stack([img, img, img], axis=0)  # (3, H, W)

        # Convert to Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(labels)

        return {
            "image": img_tensor,
            "labels": label_tensor,
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }


def make_folds(load_cached_data=True):
    """
    Creates 5-fold CV splits from the development set using Iterative Stratification.
    Caches the resulting dataframe to disk.
    """
    cache_path = os.path.join(Config.WORKING_DIR, "folds_data.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Creating new folds...")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine to form full development set
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare for Iterative Stratification
    X = df["rec_id"].values.reshape(-1, 1)
    label_cols = [c for c in df.columns if c.startswith("species_")]
    y = df[label_cols].values

    # Initialize Stratifier
    stratifier = IterativeStratification(
        n_splits=Config.NUM_FOLDS,
        order=1,
        sample_distribution_per_fold=[1.0 / Config.NUM_FOLDS] * Config.NUM_FOLDS,
    )

    # Assign folds
    df["fold"] = -1
    for fold_idx, (train_indices, val_indices) in enumerate(stratifier.split(X, y)):
        df.loc[val_indices, "fold"] = fold_idx

    # Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(
    fold_idx,
    img_size=Config.IMG_SIZE_DEFAULT,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
):
    """
    Returns train and validation dataloaders for a specific fold.
    """
    # Load folds
    df = make_folds(load_cached_data=True)

    # Split
    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    # Debug mode: subset data
    if debug:
        print(f"DEBUG: Using {Config.DEBUG_DATA_SUBSET:.0%} of data")
        train_df = train_df.sample(
            frac=Config.DEBUG_DATA_SUBSET, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            frac=Config.DEBUG_DATA_SUBSET, random_state=Config.SEED
        ).reset_index(drop=True)

    # Create Datasets
    train_dataset = BirdDataset(train_df, phase="train", img_size=img_size)
    val_dataset = BirdDataset(val_df, phase="val", img_size=img_size)

    # Create Loaders
    # Note: We use drop_last=True for training to ensure consistent batch sizes for statistics
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, val_loader


def get_test_loader(img_size=Config.IMG_SIZE_DEFAULT, batch_size=Config.BATCH_SIZE):
    """
    Returns the test dataloader.
    """
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    test_dataset = BirdDataset(test_df, phase="test", img_size=img_size)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )

    return test_loader


def seed_worker(worker_id):
    """
    Helper to seed dataloader workers for reproducibility.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # random.seed(worker_seed) # If using python random module
