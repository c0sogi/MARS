import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_transforms(mode="train"):
    """
    Returns the Albumentations transforms for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: Composed transforms.
    """
    # Enforce deterministic behavior via global seed (set_seed should be called in main)

    if mode == "train":
        return A.Compose(
            [
                # Resize to fixed dimensions: 256 (Freq) x 640 (Time)
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH, p=1.0),
                # Horizontal Flip (Time Inversion)
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout (CoarseDropout)
                # Random rectangular masks to simulate noise/occlusion
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
                # Normalization (ImageNet stats) - Critical for pretrained SE-ResNet
                A.Normalize(
                    mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        # Validation / Test / Pseudo-label generation
        return A.Compose(
            [
                # Strict Resize
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH, p=1.0),
                # Normalization
                A.Normalize(
                    mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(p=1.0),
            ]
        )


class BirdDataset(Dataset):
    """
    Custom Dataset for loading Bird Spectrograms.
    Handles mapping from WAV paths to BMP spectrograms, channel replication,
    and label extraction.
    """

    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'val', or 'test'. Used for debugging/logging.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Pre-calculate label column names
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Verify label columns exist if not in test mode (strictly)
        # However, test.csv also has these columns (filled with 0), so we can always extract them.
        missing_cols = [c for c in self.label_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns in dataframe: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Map WAV file path to Spectrogram BMP path
        # Metadata file_path example: "essential_data/src_wavs/PC10_... .wav"
        # Spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension
        wav_rel_path = row["file_path"]
        wav_basename = os.path.basename(wav_rel_path)
        bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)

        # Load Image
        # "Dynamic Loading: Use purely dynamic in-memory loading without disk caching"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should be caught by metadata check, but for safety)
            # Create a black image of roughly correct size to avoid crashing
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Grayscale -> RGB
        # Input Adaptation for ResNet backbone
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transforms provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Extract Labels
        labels = row[self.label_cols].values.astype(np.float32)

        # Sanity check for NaNs in labels (critical for pseudo-labels)
        if np.isnan(labels).any():
            labels = np.nan_to_num(labels, nan=0.0)

        return image, torch.tensor(labels), torch.tensor(rec_id)


def get_dataloaders(extra_train_df=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    Supports concatenating additional training data (e.g., pseudo-labels).

    Args:
        extra_train_df (pd.DataFrame, optional): Additional dataframe to merge
                                                 with the training set (for Stage 3).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Handle Pseudo-labeling / Extra Data
    if extra_train_df is not None:
        # Concatenate original train with extra data
        # Ensure columns match
        common_cols = list(set(train_df.columns) & set(extra_train_df.columns))
        train_df = pd.concat(
            [train_df[common_cols], extra_train_df[common_cols]], axis=0
        )
        train_df = train_df.reset_index(drop=True)
        # Shuffle the combined dataset
        train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(
            drop=True
        )

    # 3. Create Datasets
    train_dataset = BirdDataset(
        train_df, transforms=get_transforms(mode="train"), mode="train"
    )

    val_dataset = BirdDataset(val_df, transforms=get_transforms(mode="val"), mode="val")

    test_dataset = BirdDataset(
        test_df, transforms=get_transforms(mode="test"), mode="test"
    )

    # 4. Create DataLoaders
    # Use Config.NUM_WORKERS and Config.BATCH_SIZE
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for stability in training (esp. with Mixup)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
