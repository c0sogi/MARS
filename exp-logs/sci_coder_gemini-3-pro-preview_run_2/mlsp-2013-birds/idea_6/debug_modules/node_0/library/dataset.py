import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from skmultilearn.model_selection import IterativeStratifiedKFold
from library import config, utils


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Loads spectrograms, applies augmentations, and computes deltas for 3-channel input.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Identify label columns (species_0 to species_18)
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full path to the spectrogram
        img_path = os.path.join(config.INPUT_DIR, row["file_path_spec"])

        # Load image in grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files (robustness)
        if image is None:
            image = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH), dtype=np.uint8)

        # Resize to target dimensions (Height, Width)
        # Note: cv2.resize expects (Width, Height)
        image = cv2.resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))

        # Apply Albumentations (works on numpy arrays)
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to Tensor (1, H, W)
        image_tensor = torch.tensor(image).unsqueeze(0)

        # Compute Deltas to get 3 channels (Spec, Delta, Delta-Delta)
        # Result shape: (3, H, W)
        final_tensor = utils.compute_deltas(image_tensor)

        # Prepare Labels
        if self.mode == "test":
            # Dummy labels for test set
            labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        else:
            labels = torch.tensor(row[self.label_cols].values.astype(np.float32))

        return final_tensor, labels


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.
    """
    if mode == "train":
        return A.Compose(
            [
                # Photometric Augmentation: Handle recording gain variations
                A.RandomBrightnessContrast(p=0.5),
                # Geometric/SpecAugment Simulation: CoarseDropout
                # Creates rectangular holes to simulate Time/Frequency masking
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
            ]
        )
    else:
        # No test-time augmentation (TTA) in this basic pipeline,
        # resizing is handled in Dataset.__getitem__
        return None


def get_iterative_folds(load_cached_data=True):
    """
    Loads training and validation metadata, merges them, and applies
    Iterative Stratified K-Fold splitting. Caches the result to disk.
    """
    cache_path = os.path.join(config.WORKING_DIR, "folds_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating new folds...")

    # 2. Load and Merge Data
    if not os.path.exists(config.TRAIN_CSV) or not os.path.exists(config.VAL_CSV):
        raise FileNotFoundError("Metadata CSV files not found.")

    df_train = pd.read_csv(config.TRAIN_CSV)
    df_val = pd.read_csv(config.VAL_CSV)

    # Combine to create full development set
    df_full = pd.concat([df_train, df_val], axis=0, ignore_index=True)

    # 3. Prepare for Iterative Stratification
    # X is just indices, y is the multi-label matrix
    label_cols = [c for c in df_full.columns if c.startswith("species_")]
    X = df_full["rec_id"].values.reshape(-1, 1)
    y = df_full[label_cols].values

    # 4. Perform Split
    k_fold = IterativeStratifiedKFold(
        n_splits=config.NUM_FOLDS, order=1, random_state=config.SEED
    )

    # Initialize fold column
    df_full["fold"] = -1

    for fold_id, (train_idx, val_idx) in enumerate(k_fold.split(X, y)):
        df_full.loc[val_idx, "fold"] = fold_id

    # 5. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_full.to_parquet(cache_path, index=False)
    print(f"Saved folds to {cache_path}")

    return df_full


def get_dataloaders(fold_id, df_folds, batch_size=config.BATCH_SIZE, debug=False):
    """
    Creates Train and Validation DataLoaders for a specific fold.
    """
    # Filter for Train/Val
    df_train = df_folds[df_folds["fold"] != fold_id].copy()
    df_val = df_folds[df_folds["fold"] == fold_id].copy()

    # Debug Mode: Subset data
    if debug:
        df_train = df_train.iloc[: config.DEBUG_SUBSET_SIZE]
        df_val = df_val.iloc[: config.DEBUG_SUBSET_SIZE]
        print(
            f"DEBUG: Reduced train size to {len(df_train)}, val size to {len(df_val)}"
        )

    # Create Datasets
    train_dataset = BirdDataset(
        df_train, transforms=get_transforms(mode="train"), mode="train"
    )
    val_dataset = BirdDataset(df_val, transforms=get_transforms(mode="val"), mode="val")

    # Worker Init Function for Reproducibility
    def worker_init_fn(worker_id):
        utils.set_seed(config.SEED + worker_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=config.BATCH_SIZE):
    """
    Creates a DataLoader for the Test set.
    """
    if not os.path.exists(config.TEST_CSV):
        raise FileNotFoundError("Test metadata CSV not found.")

    df_test = pd.read_csv(config.TEST_CSV)

    test_dataset = BirdDataset(
        df_test, transforms=get_transforms(mode="test"), mode="test"
    )

    def worker_init_fn(worker_id):
        utils.set_seed(config.SEED + worker_id)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return test_loader
