import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the Bird Species Classification task.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    transforms_list = []

    # 1. Resize (High-Fidelity Alignment)
    # Resizing to 256x640 to preserve frequency resolution and temporal morphology.
    transforms_list.append(A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH))

    if mode == "train":
        # 2. Horizontal Flip (Time Inversion)
        # Randomly flips the spectrogram horizontally.
        transforms_list.append(A.HorizontalFlip(p=0.5))

        # 3. Unstructured Cutout (CoarseDropout)
        # Randomly masks rectangular regions to encourage the model to use partial features.
        transforms_list.append(
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.5,
            )
        )

    # 4. Normalize (ImageNet stats)
    # Critical for pre-trained models.
    transforms_list.append(
        A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    )

    # 5. ToTensor
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


class BirdDataset(Dataset):
    """
    PyTorch Dataset for loading bird spectrograms.
    Handles dynamic loading, channel replication, and label extraction.
    """

    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Identify label columns (species_0 to species_18)
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Check if label columns exist in the dataframe
        self.has_labels = all(col in self.df.columns for col in self.label_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Construct file path
        # Metadata contains relative path to wav file (e.g., essential_data/src_wavs/file.wav)
        # Spectrograms are in Config.SPECTROGRAM_DIR with .bmp extension
        wav_path = row["file_path"]
        wav_basename = os.path.basename(wav_path)
        bmp_filename = os.path.splitext(wav_basename)[0] + ".bmp"
        img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_filename)

        # Dynamic Loading: Load image from disk
        # Load as grayscale first
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback for missing images (though EDA suggests all exist)
            # Return a black image of correct size
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Convert Grayscale to RGB
        # This repeats the single channel 3 times to match ImageNet pre-trained input expectation
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply Transforms (Augmentation/Normalization)
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        # Extract Labels
        labels = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
        if self.has_labels:
            # Extract values. This works for both binary (0/1) and soft probabilities (0.0-1.0)
            label_vals = row[self.label_cols].values.astype(np.float32)
            labels = torch.tensor(label_vals, dtype=torch.float32)

        return img, labels, rec_id


def worker_init_fn(worker_id):
    """
    Sets the random seed for each DataLoader worker to ensure reproducibility.
    """
    worker_seed = Config.SEED + worker_id
    np.random.seed(worker_seed)


def get_dataloaders(pseudo_labels_df=None):
    """
    Creates and returns DataLoaders for training, validation, and testing.

    Args:
        pseudo_labels_df (pd.DataFrame, optional): A DataFrame containing pseudo-labels for the test set.
                                                   Used during the distillation stages to augment the training data.
                                                   Must contain 'rec_id' and 'species_X' columns.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Debug Mode Handling
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)

        if pseudo_labels_df is not None:
            # Filter pseudo labels to match the debug test set
            test_ids = test_df["rec_id"].unique()
            pseudo_labels_df = pseudo_labels_df[
                pseudo_labels_df["rec_id"].isin(test_ids)
            ]

    # 3. Distillation Logic (Merging Pseudo-Labels)
    if pseudo_labels_df is not None:
        # We need to combine the original training data with the pseudo-labeled test data.

        # Prepare test dataframe for merging
        # Drop existing placeholder species columns in test_df
        cols_to_drop = [c for c in test_df.columns if c.startswith("species_")]
        test_df_clean = test_df.drop(columns=cols_to_drop, errors="ignore")

        # Merge test file paths with pseudo-labels
        # pseudo_labels_df should have [rec_id, species_0, ..., species_18]
        pseudo_train_df = pd.merge(
            test_df_clean, pseudo_labels_df, on="rec_id", how="inner"
        )

        # Concatenate original train and pseudo train
        # Note: train_df has binary labels, pseudo_train_df has soft labels.
        # This is acceptable for BCEWithLogitsLoss.
        train_df = pd.concat([train_df, pseudo_train_df], axis=0, ignore_index=True)

        # Shuffle the combined dataset
        train_df = train_df.sample(frac=1.0, random_state=Config.SEED).reset_index(
            drop=True
        )

    # 4. Create Datasets
    train_dataset = BirdDataset(
        train_df, mode="train", transform=get_transforms(mode="train")
    )

    val_dataset = BirdDataset(val_df, mode="val", transform=get_transforms(mode="val"))

    test_dataset = BirdDataset(
        test_df, mode="test", transform=get_transforms(mode="test")
    )

    # 5. Create DataLoaders
    # pin_memory=True speeds up host-to-device transfer
    # drop_last=True for training to avoid unstable batch norm stats on small last batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
