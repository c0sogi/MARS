import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config


class BirdDataset(Dataset):
    """
    Dataset class for loading bird spectrograms and labels.
    Handles mapping from WAV paths to BMP spectrograms, channel replication,
    normalization, and augmentation.
    """

    def __init__(self, df, transforms=None, img_dir=Config.SPECTROGRAM_DIR):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, species columns).
            transforms (A.Compose): Albumentations transform pipeline.
            img_dir (str): Directory containing the BMP spectrograms.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_dir = img_dir

        # Identify label columns (species_0 to species_18)
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-compute file paths to avoid overhead in __getitem__
        self.image_paths = []
        for rel_path in self.df["file_path"]:
            # rel_path is like "essential_data/src_wavs/PC10_... .wav"
            # We need "PC10_... .bmp" in img_dir
            basename = os.path.basename(rel_path)
            bmp_name = os.path.splitext(basename)[0] + ".bmp"
            full_path = os.path.join(self.img_dir, bmp_name)
            self.image_paths.append(full_path)

        # Pre-compute labels as float32 array
        # If label columns exist, use them. Otherwise (e.g. pure test without pseudo), dummy zeros.
        if self.label_cols:
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df), Config.NUM_CLASSES), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Dynamic Loading: Load image from disk
        # Load as grayscale (H, W)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (should not happen if data is intact)
            # Create a black image of expected size to prevent crash
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Convert Grayscale to RGB (H, W, 3)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Get Label
        label = self.labels[idx]

        # Return dictionary or tuple? Standard is tuple (img, label) for PyTorch
        # Also returning rec_id can be useful for tracking/debugging
        rec_id = self.df.iloc[idx]["rec_id"]

        return image, torch.tensor(label), torch.tensor(rec_id)


def get_transforms(data="train"):
    """
    Returns the Albumentations transform pipeline.

    Args:
        data (str): 'train' or 'valid'/'test'.
    """
    if data == "train":
        return A.Compose(
            [
                # High-Fidelity Resolution
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                # Augmentation: Horizontal Flip (Time Inversion)
                A.HorizontalFlip(p=0.5),
                # Augmentation: Unstructured Cutout (CoarseDropout)
                # Random rectangular masks
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
                # Normalization (ImageNet)
                A.Normalize(
                    mean=Config.IMAGENET_MEAN,
                    std=Config.IMAGENET_STD,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # Resize
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                # Normalization
                A.Normalize(
                    mean=Config.IMAGENET_MEAN,
                    std=Config.IMAGENET_STD,
                ),
                ToTensorV2(),
            ]
        )


def get_dataloaders(
    train_metadata=Config.TRAIN_METADATA,
    val_metadata=Config.VAL_METADATA,
    test_metadata=Config.TEST_METADATA,
    pseudo_labels_df=None,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles merging of pseudo-labels if provided.

    Args:
        train_metadata (str): Path to train CSV.
        val_metadata (str): Path to val CSV.
        test_metadata (str): Path to test CSV.
        pseudo_labels_df (pd.DataFrame, optional): DataFrame containing pseudo-labels for the test set.
            Must contain 'rec_id' and 'species_0'...'species_18'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    df_train = pd.read_csv(train_metadata)
    df_val = pd.read_csv(val_metadata)
    df_test = pd.read_csv(test_metadata)

    # Handle Pseudo-Labels (Student Training Phase)
    if pseudo_labels_df is not None:
        # Ensure pseudo_labels_df has the correct columns
        label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # We only want the test samples that correspond to the pseudo labels
        # Merge pseudo_labels_df with df_test to get file paths + new labels
        # pseudo_labels_df should have [rec_id, species_0, ..., species_18]

        # Drop original zero-filled label columns from df_test
        df_test_clean = df_test.drop(columns=label_cols, errors="ignore")

        # Merge on rec_id
        df_pseudo = pd.merge(df_test_clean, pseudo_labels_df, on="rec_id", how="inner")

        # Concatenate original train and pseudo-labeled test
        # Note: df_train labels are 0/1 integers, df_pseudo labels are floats.
        # Pandas will cast to float, which is compatible with BCEWithLogitsLoss.
        df_train_final = pd.concat([df_train, df_pseudo], axis=0, ignore_index=True)

        print(
            f"Pseudo-labels integrated. New training set size: {len(df_train_final)} "
            f"(Original: {len(df_train)}, Pseudo: {len(df_pseudo)})"
        )

        df_train = df_train_final
    else:
        print(f"Standard training. Training set size: {len(df_train)}")

    # Create Datasets
    train_dataset = BirdDataset(df_train, transforms=get_transforms(data="train"))

    val_dataset = BirdDataset(df_val, transforms=get_transforms(data="valid"))

    test_dataset = BirdDataset(df_test, transforms=get_transforms(data="test"))

    # Worker Init Function for Seeding
    def worker_init_fn(worker_id):
        np.random.seed(Config.SEED + worker_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, test_loader
