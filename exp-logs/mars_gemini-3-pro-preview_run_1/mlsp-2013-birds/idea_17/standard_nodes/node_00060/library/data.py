import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class BirdDataset(Dataset):
    """
    Dataset class for loading bird spectrograms and labels.
    Handles mapping from WAV filenames to BMP spectrograms, channel replication,
    and applying augmentations.
    """

    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Identify label columns (species_0 to species_18)
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Pre-compute file paths to avoid overhead in __getitem__
        self.file_paths = []
        for idx, row in self.df.iterrows():
            # Metadata contains path to wav file, e.g., essential_data/src_wavs/file.wav
            wav_path = row["file_path"]
            wav_name = os.path.basename(wav_path)
            # Spectrograms are .bmp files with the same basename
            bmp_name = os.path.splitext(wav_name)[0] + ".bmp"
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_name)
            self.file_paths.append(full_path)

        # Pre-load labels for training/validation/student modes
        if self.mode in ["train", "val", "student"]:
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        img_path = self.file_paths[idx]

        # Load as grayscale (spectrograms are single channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (safety mechanism)
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Grayscale (H, W) -> RGB (H, W, 3)
        # This adapts the 1-channel spectrogram to the 3-channel ResNet backbone
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Basic fallback transform
            image = cv2.resize(image, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        rec_id = self.df.iloc[idx]["rec_id"]

        if self.mode in ["train", "val", "student"]:
            label = self.labels[idx]
            return image, torch.tensor(label), rec_id
        else:
            # Test mode: return image and ID for submission generation
            return image, rec_id


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.
    """
    if mode == "train" or mode == "student":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout approximation using CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(Config.IMG_HEIGHT * 0.1),
                    max_width=int(Config.IMG_WIDTH * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders(pseudo_labels_df=None):
    """
    Constructs DataLoaders for train, validation, and test sets.

    Args:
        pseudo_labels_df (pd.DataFrame, optional): DataFrame containing pseudo-labels
                                                   for the test set. If provided, it is
                                                   combined with the training data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug Mode: Subset data for rapid iteration
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SUBSET_SIZE)
        val_df = val_df.head(Config.DEBUG_SUBSET_SIZE)
        test_df = test_df.head(Config.DEBUG_SUBSET_SIZE)
        if pseudo_labels_df is not None:
            pseudo_labels_df = pseudo_labels_df.head(Config.DEBUG_SUBSET_SIZE)

    # Handle Pseudo-labels for Student Training
    if pseudo_labels_df is not None:
        # We need to ensure the pseudo_labels_df has the 'file_path' column.
        # Usually, pseudo_labels_df comes from predictions on test_df, so we merge to get paths.
        if "file_path" not in pseudo_labels_df.columns:
            temp_test = test_df[["rec_id", "file_path"]]
            pseudo_labels_df = pd.merge(
                pseudo_labels_df, temp_test, on="rec_id", how="left"
            )

        # Select common columns to align schemas
        label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]
        common_cols = ["rec_id", "file_path"] + label_cols

        train_subset = train_df[common_cols].copy()
        pseudo_subset = pseudo_labels_df[common_cols].copy()

        # Combine original labeled data with pseudo-labeled data
        combined_df = pd.concat(
            [train_subset, pseudo_subset], axis=0, ignore_index=True
        )

        # Create dataset with student transforms
        train_dataset = BirdDataset(
            combined_df, mode="student", transform=get_transforms("student")
        )
    else:
        # Standard supervised training
        train_dataset = BirdDataset(
            train_df, mode="train", transform=get_transforms("train")
        )

    val_dataset = BirdDataset(val_df, mode="val", transform=get_transforms("val"))
    test_dataset = BirdDataset(test_df, mode="test", transform=get_transforms("test"))

    # Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
