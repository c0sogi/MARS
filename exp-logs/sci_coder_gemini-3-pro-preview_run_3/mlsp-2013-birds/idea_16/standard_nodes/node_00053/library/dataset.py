import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class BirdDataset(Dataset):
    """
    Dataset class for Bird Species Classification.
    Loads spectrograms, applies transforms, and returns pseudo-RGB images and multi-hot labels.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.file_paths = df["file_path"].values

        # Pre-process labels for train/val modes
        if self.mode != "test":
            self.labels = df["labels"].values
            self.num_classes = Config.NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Map wav path to bmp spectrogram path
        wav_path = self.file_paths[idx]
        filename = os.path.basename(wav_path).replace(".wav", ".bmp")
        image_path = os.path.join(Config.SPECTROGRAM_DIR, filename)

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing images (should not happen based on analysis)
            image = np.zeros(
                (Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]), dtype=np.uint8
            )

        # Convert to RGB (Replicate channels)
        # We do this before transforms if transforms expect 3 channels,
        # or after if we want to treat it as grayscale.
        # Standard Albumentations works fine with grayscale, but backbones need 3 channels.
        # Strategy: Keep grayscale for geometric transforms, convert to 3-channel at the end.

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # If transforms didn't convert to tensor (e.g. custom pipeline), do it here
        # But our get_transforms will end with ToTensorV2

        # The image is now a Tensor (C, H, W).
        # If it's 1 channel, we need to repeat it to 3.
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Normalize to [0, 1] if not already done by ToTensorV2 (which divides by 255 for uint8)
        # Albumentations ToTensorV2 converts to float and divides by 255 if input is uint8.

        if self.mode == "test":
            return image, torch.zeros(Config.NUM_CLASSES)  # Dummy label

        # Process Labels
        label_str = self.labels[idx]
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)

        if (
            isinstance(label_str, str)
            and label_str != "?"
            and len(label_str.strip()) > 0
        ):
            indices = [int(x) for x in label_str.split()]
            label_vec[indices] = 1.0

        return image, label_vec


def get_transforms(data="train"):
    """
    Returns the Albumentations transformation pipeline.
    """
    if data == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                # Horizontal Translation (Time shifting) using zero-padding
                A.ShiftScaleRotate(
                    shift_limit_x=0.1,  # Shift time
                    shift_limit_y=0.0,  # Do not shift freq
                    scale_limit=0.0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
                # Photometric Augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.Normalize(
                    mean=(0.485,), std=(0.229,), max_pixel_value=255.0
                ),  # Grayscale stats or standard
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]),
                A.Normalize(mean=(0.485,), std=(0.229,), max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )


def create_folds(df, n_folds=5):
    """
    Applies Iterative Stratification to create balanced folds for multi-label data.
    """
    # Prepare X and y
    # X can be just indices
    X = df.index.values.reshape(-1, 1)

    # Construct binary label matrix for stratification
    num_samples = len(df)
    num_classes = Config.NUM_CLASSES
    y = np.zeros((num_samples, num_classes))

    for idx, row in df.iterrows():
        label_str = row["labels"]
        if (
            isinstance(label_str, str)
            and label_str != "?"
            and len(label_str.strip()) > 0
        ):
            indices = [int(x) for x in label_str.split()]
            y[idx, indices] = 1

    # Initialize Iterative Stratification
    k_fold = IterativeStratification(n_splits=n_folds, order=1)

    df["fold"] = -1

    # Generate folds
    # k_fold.split returns indices for train and test sets for each fold
    # We want to assign a fold ID to the test set of each split
    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        df.iloc[val_indices, df.columns.get_loc("fold")] = fold_idx

    return df


def prepare_data(load_cached_data=True):
    """
    Loads metadata, merges train/val for CV, creates folds, and caches the result.
    Returns the dataframe with fold assignments.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "folds.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Creating folds from scratch...")

    # Load provided metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    # Merge them to create a full labeled dataset for 5-fold CV
    # Reset index to ensure unique indices for stratification
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Create Folds
    full_df = create_folds(full_df, n_folds=Config.N_FOLDS)

    # Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    full_df.to_parquet(cache_path, index=False)
    print(f"Folds saved to {cache_path}")

    return full_df


def load_test_data():
    """
    Loads the test metadata.
    """
    return pd.read_csv(Config.TEST_METADATA)
