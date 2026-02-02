import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import utility for seeding
from library.utils import set_seed


def get_transforms(phase="train", height=256, width=640):
    """
    Returns the Albumentations transform pipeline for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        height (int): Target image height.
        width (int): Target image width.

    Returns:
        A.Compose: The transform pipeline.
    """
    # ImageNet Normalization Constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.HorizontalFlip(p=0.5),  # Random time inversion
                # Unstructured Cutout via CoarseDropout
                A.CoarseDropout(
                    max_holes=8,
                    max_height=int(height * 0.1),
                    max_width=int(width * 0.1),
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.5,
                ),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Deterministic resizing and normalization
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    """
    Custom Dataset for Bird Species Classification.
    Handles dynamic loading of spectrograms, channel replication, and label extraction.
    Supports both hard labels (training) and soft labels (distillation).
    """

    def __init__(self, df, img_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, species columns).
            img_dir (str): Directory containing the spectrogram .bmp files.
            transform (A.Compose): Albumentations transforms.
            is_test (bool): If True, returns dummy labels.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        # Identify label columns
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Determine image path
        # Metadata has 'file_path' pointing to .wav, e.g., essential_data/src_wavs/PC10_... .wav
        # Spectrograms are in supplemental_data/spectrograms/PC10_... .bmp
        wav_path = row["file_path"]
        basename = os.path.basename(wav_path)
        bmp_filename = os.path.splitext(basename)[0] + ".bmp"
        img_path = os.path.join(self.img_dir, bmp_filename)

        # Dynamic Loading
        # Load as grayscale (spectrograms are single channel usually, or mapped)
        # The description says "pixel value for an image", usually grayscale intensity.
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing files (should be caught by EDA, but for safety)
            # Create a black image of expected size
            image = np.zeros((256, 640), dtype=np.uint8)

        # Channel Replication: 1 -> 3 channels to match ImageNet pretrained models
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get Labels
        if self.is_test:
            # Return dummy labels for test set inference
            labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        else:
            # Extract labels from dataframe
            # This works for both binary ground truth and soft pseudo-labels
            label_values = row[self.label_cols].values.astype(np.float32)
            labels = torch.tensor(label_values, dtype=torch.float32)

        return image, labels, torch.tensor(rec_id, dtype=torch.long)


def mixup_batch(data, target, alpha=0.2, device="cuda"):
    """
    Applies Mixup regularization to a batch of data.

    Args:
        data (torch.Tensor): Input batch [B, C, H, W].
        target (torch.Tensor): Target batch [B, NumClasses].
        alpha (float): Mixup alpha parameter.
        device (str): Device.

    Returns:
        mixed_data (torch.Tensor): Mixed inputs.
        mixed_target (torch.Tensor): Mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_data = lam * data + (1 - lam) * data[index]
    mixed_target = lam * target + (1 - lam) * target[index]

    return mixed_data, mixed_target


def create_dataloaders(batch_size=32, pseudo_labels_df=None, num_workers=4, seed=42):
    """
    Creates DataLoaders for train, validation, and test sets.
    Supports merging pseudo-labeled data for distillation.

    Args:
        batch_size (int): Batch size.
        pseudo_labels_df (pd.DataFrame, optional): DataFrame containing pseudo-labels for the test set.
                                                   Must have 'rec_id' and 'species_X' columns.
        num_workers (int): Number of worker threads.
        seed (int): Random seed for reproducibility.

    Returns:
        train_loader, val_loader, test_loader
    """
    set_seed(seed)

    # Paths
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Handle Pseudo-labels (Distillation)
    if pseudo_labels_df is not None:
        # Ensure pseudo_labels_df has the same columns as train_df
        # We need to map the pseudo-labels to the corresponding file paths in test_df

        # 1. Merge pseudo-labels with test_df metadata to get file paths
        # test_df has 'rec_id', 'file_path', etc.
        # pseudo_labels_df has 'rec_id', 'species_0', ...

        # Drop original zero-filled species columns from test_df
        cols_to_drop = [c for c in test_df.columns if c.startswith("species_")]
        test_meta = test_df.drop(columns=cols_to_drop)

        # Merge
        pseudo_train_df = pd.merge(
            test_meta, pseudo_labels_df, on="rec_id", how="inner"
        )

        # 2. Concatenate with original training data
        # Ensure column order matches
        common_cols = train_df.columns.intersection(pseudo_train_df.columns)
        train_df = pd.concat(
            [train_df[common_cols], pseudo_train_df[common_cols]],
            axis=0,
            ignore_index=True,
        )

        # print(f"Distillation: Added {len(pseudo_train_df)} pseudo-labeled samples. New train size: {len(train_df)}")

    # Transforms
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Datasets
    train_dataset = BirdDataset(
        train_df, SPECTROGRAM_DIR, transform=train_transform, is_test=False
    )
    val_dataset = BirdDataset(
        val_df, SPECTROGRAM_DIR, transform=val_transform, is_test=False
    )
    test_dataset = BirdDataset(
        test_df, SPECTROGRAM_DIR, transform=val_transform, is_test=True
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Useful for Mixup stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
