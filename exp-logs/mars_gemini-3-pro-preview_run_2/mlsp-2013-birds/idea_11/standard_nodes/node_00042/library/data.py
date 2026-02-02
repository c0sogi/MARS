import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything


class BirdDataset(Dataset):
    """
    Dataset class for loading bird spectrograms.
    Handles path correction to use filtered spectrograms, resizing, and pseudo-RGB conversion.
    """

    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]
        self.labels = None
        if self.mode != "test":
            self.labels = self.df[self.label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct file path
        # Metadata points to 'spectrograms', but we strictly use 'filtered_spectrograms'
        original_rel_path = row["file_path_spec"]
        filename = os.path.basename(original_rel_path)
        img_path = os.path.join(Config.FILTERED_SPECTROGRAM_DIR, filename)

        # Load Image
        # Load as-is (likely grayscale BMP)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Fallback for missing files (safety net)
        if image is None:
            # Create black image with correct aspect ratio
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8
            )

        # Ensure image is 2D (H, W) before stacking
        if len(image.shape) == 3:
            image = image[:, :, 0]

        # Resize
        # Config.IMAGE_SIZE is (Height, Width) = (224, 448)
        # cv2.resize expects (Width, Height)
        target_h, target_w = Config.IMAGE_SIZE
        image = cv2.resize(image, (target_w, target_h))

        # Convert to Pseudo-RGB (3 channels)
        image = np.stack([image, image, image], axis=-1)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Default transform: Normalize and ToTensor
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

        # Return Logic
        if self.mode == "test":
            # Return image and rec_id for submission mapping
            rec_id = row["rec_id"]
            return image, torch.tensor(rec_id, dtype=torch.long)
        else:
            # Return image and multi-hot labels
            label = torch.tensor(self.labels[idx])
            return image, label


def get_transforms(mode="train"):
    """
    Returns albumentations transforms.
    Simulates SpecAugment using CoarseDropout in training mode.
    """
    target_h, target_w = Config.IMAGE_SIZE

    if mode == "train":
        return A.Compose(
            [
                # Geometric: SpecAugment simulation
                # Mask out random rectangles (approximating time/freq masking)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(target_h * 0.15),
                    max_width=int(target_w * 0.15),
                    min_holes=2,
                    fill_value=0,
                    p=0.5,
                ),
                # Photometric: Robustness to recording quality
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                # Normalization (ImageNet)
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


def get_data_with_folds(load_cached_data=True):
    """
    Loads training data and assigns folds using Iterative Stratification.
    Caches the result to parquet to ensure deterministic splits across runs.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds_data.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)

            # Cite debug_lesson_5: Validate Persistent Caches Against Runtime Configuration
            if "fold" in df.columns:
                present_folds = set(df["fold"].unique())
                expected_folds = set(range(Config.N_FOLDS))

                # We expect all folds 0..N-1 to be present.
                # Note: -1 might exist if stratification dropped samples, but 0..N-1 must exist.
                if expected_folds.issubset(present_folds):
                    print(f"Loaded valid folds from cache: {cache_path}")
                    return df
                else:
                    print(
                        f"Cache invalid: Found folds {present_folds}, expected {expected_folds}. Regenerating."
                    )
            else:
                print("Cache invalid: 'fold' column missing. Regenerating.")

        except Exception as e:
            print(f"Cache load failed: {e}. Regenerating.")

    # 2. Compute folds from scratch
    if not os.path.exists(Config.TRAIN_CSV) or not os.path.exists(Config.VAL_CSV):
        raise FileNotFoundError("Metadata CSVs not found in ./metadata")

    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Combine train and val to perform global K-Fold
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # Prepare data for stratification
    label_cols = [c for c in df_full.columns if c.startswith("species_")]
    X = df_full["rec_id"].values.reshape(-1, 1)  # Dummy X
    y = df_full[label_cols].values

    # Initialize Iterative Stratification
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    df_full["fold"] = -1

    # Assign folds
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df_full.loc[val_indices, "fold"] = fold_idx

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    df_full.to_parquet(cache_path, index=False)

    return df_full


def get_loaders(
    fold_idx, df_folds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Creates train and validation DataLoaders for a specific fold.
    """
    # Split Data
    df_train = df_folds[df_folds["fold"] != fold_idx].reset_index(drop=True)
    df_val = df_folds[df_folds["fold"] == fold_idx].reset_index(drop=True)

    # Apply Debug Subset if enabled
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)
        df_val = df_val.head(Config.DEBUG_SUBSET_SIZE)

    # Create Datasets
    train_dataset = BirdDataset(
        df_train, transform=get_transforms(mode="train"), mode="train"
    )
    val_dataset = BirdDataset(df_val, transform=get_transforms(mode="val"), mode="val")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates the test DataLoader.
    """
    if not os.path.exists(Config.TEST_CSV):
        raise FileNotFoundError("Test metadata CSV not found.")

    df_test = pd.read_csv(Config.TEST_CSV)

    test_dataset = BirdDataset(
        df_test, transform=get_transforms(mode="test"), mode="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
