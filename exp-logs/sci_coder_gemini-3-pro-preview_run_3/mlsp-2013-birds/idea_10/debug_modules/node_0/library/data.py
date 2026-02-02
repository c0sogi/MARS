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
    Dataset class for Bird Species Classification.
    Loads BMP spectrograms, converts to 3-channel RGB, and applies transforms.
    """

    def __init__(self, df, data_dir, transform=None, mode="train"):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.num_classes = 19

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Map wav filename to bmp filename
        # Metadata contains relative path to wav, e.g., essential_data/src_wavs/file.wav
        wav_rel_path = row["file_path"]
        wav_filename = os.path.basename(wav_rel_path)
        bmp_filename = wav_filename.replace(".wav", ".bmp")

        img_path = os.path.join(self.data_dir, bmp_filename)

        # Load Image (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Handle missing files (robustness)
        if image is None:
            # Create a blank black image of expected size
            image = np.zeros((224, 224), dtype=np.uint8)

        # Input Adaptation: Replicate single channel to 3 channels for ImageNet models
        image = np.stack([image, image, image], axis=-1)

        # Apply Albumentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Return based on mode
        if self.mode == "test":
            # Return image and rec_id for submission
            return image, torch.tensor(row["rec_id"], dtype=torch.long)
        else:
            # Parse labels
            label_str = str(row["labels"])
            labels = np.zeros(self.num_classes, dtype=np.float32)

            # Check for valid label string
            if label_str != "?" and label_str.lower() != "nan" and label_str.strip():
                try:
                    indices = [int(x) for x in label_str.split()]
                    # Ensure indices are within bounds
                    valid_indices = [i for i in indices if 0 <= i < self.num_classes]
                    labels[valid_indices] = 1.0
                except ValueError:
                    pass  # Keep zero vector if parsing fails

            return image, torch.tensor(labels, dtype=torch.float32)


def get_transforms(mode, cfg):
    """
    Returns the Albumentations transformation pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(cfg.image_size[0], cfg.image_size[1]),
                # Horizontal Translation (Shift) with Zero-Padding
                A.ShiftScaleRotate(
                    shift_limit_x=cfg.time_shift_limit,
                    shift_limit_y=0,
                    scale_limit=0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentation
                A.RandomBrightnessContrast(
                    brightness_limit=cfg.brightness_limit,
                    contrast_limit=cfg.contrast_limit,
                    p=0.5,
                ),
                # Normalize with ImageNet stats
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: Resize and Normalize only
        return A.Compose(
            [
                A.Resize(cfg.image_size[0], cfg.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def prepare_folds(cfg, load_cached_data=True):
    """
    Combines training and validation metadata, applies Iterative Stratification to create 5 folds,
    and caches the result to disk.
    """
    cache_path = os.path.join(cfg.working_dir, "folds.parquet")

    # Attempt to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Generating folds using Iterative Stratification...")

    # Load metadata
    train_df = pd.read_csv(cfg.train_metadata_path)
    val_df = pd.read_csv(cfg.val_metadata_path)

    # Combine datasets to maximize training data for CV
    df = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare data for stratification
    X = df["rec_id"].values.reshape(-1, 1)

    # Parse labels into binary matrix
    y = np.zeros((len(df), cfg.num_classes), dtype=int)
    for idx, row in df.iterrows():
        label_str = str(row["labels"])
        if label_str != "?" and label_str.lower() != "nan" and label_str.strip():
            try:
                indices = [int(x) for x in label_str.split()]
                valid_indices = [i for i in indices if 0 <= i < cfg.num_classes]
                y[idx, valid_indices] = 1
            except ValueError:
                pass

    # Perform Iterative Stratification
    stratifier = IterativeStratification(n_splits=cfg.n_folds, order=1)

    df["fold"] = -1

    # split(X, y) returns train_indices, test_indices
    # We assign the 'test_indices' of the split to be the validation set for that fold
    for fold_idx, (train_idx, val_idx) in enumerate(stratifier.split(X, y)):
        df.loc[val_idx, "fold"] = fold_idx

    # Save to cache
    os.makedirs(cfg.working_dir, exist_ok=True)
    df.to_parquet(cache_path)

    return df


def get_loaders(fold, df, cfg):
    """
    Creates DataLoaders for a specific fold.
    """
    # Split dataframe
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    # Debug mode subsetting
    if cfg.debug:
        train_df = train_df.head(cfg.debug_sample_size)
        val_df = val_df.head(cfg.debug_sample_size)

    # Create Datasets
    train_dataset = BirdDataset(
        train_df,
        cfg.spectrogram_dir,
        transform=get_transforms("train", cfg),
        mode="train",
    )

    val_dataset = BirdDataset(
        val_df, cfg.spectrogram_dir, transform=get_transforms("valid", cfg), mode="val"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(cfg):
    """
    Creates DataLoader for the test set.
    """
    test_df = pd.read_csv(cfg.test_metadata_path)

    if cfg.debug:
        test_df = test_df.head(cfg.debug_sample_size)

    dataset = BirdDataset(
        test_df, cfg.spectrogram_dir, transform=get_transforms("test", cfg), mode="test"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
