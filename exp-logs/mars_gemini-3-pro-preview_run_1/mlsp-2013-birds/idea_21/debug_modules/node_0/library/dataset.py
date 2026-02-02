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
    Custom Dataset for Bird Species Classification.
    Handles loading BMP spectrograms, preprocessing, and label retrieval (hard or soft).
    """

    def __init__(self, df, img_dir, transform=None, pseudo_labels_dict=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'rec_id', 'file_path', and label columns.
            img_dir (str): Directory containing the spectrogram BMP files.
            transform (albumentations.Compose): Transformations to apply.
            pseudo_labels_dict (dict, optional): Dictionary mapping rec_id to soft label numpy arrays.
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.pseudo_labels_dict = pseudo_labels_dict

        # Pre-identify label columns to avoid overhead in __getitem__
        self.label_cols = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["rec_id"]

        # Construct image path
        # The 'file_path' in metadata is relative to essential_data/src_wavs/
        # But we are loading spectrograms from supplemental_data/spectrograms/
        # We need to extract the basename and change extension to .bmp
        wav_path = row["file_path"]
        basename = os.path.basename(wav_path)
        bmp_name = os.path.splitext(basename)[0] + ".bmp"
        img_path = os.path.join(self.img_dir, bmp_name)

        # Load Image
        # Load as grayscale (cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images: create a black image
            # This ensures the pipeline doesn't crash on a missing file
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Grayscale -> RGB
        # Stack the single channel 3 times
        image = np.stack([image, image, image], axis=-1)

        # Apply Transformations (Resize, Augment, Normalize, ToTensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Get Label
        # Priority: Pseudo-labels (soft) > Metadata labels (hard)
        if self.pseudo_labels_dict is not None and rec_id in self.pseudo_labels_dict:
            label = self.pseudo_labels_dict[rec_id]
            label = torch.tensor(label, dtype=torch.float32)
        else:
            # Extract hard labels from dataframe
            # If it's the test set without pseudo labels, these will be 0s
            label_vals = row[self.label_cols].values.astype(np.float32)
            label = torch.tensor(label_vals, dtype=torch.float32)

        return image, label, rec_id


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the specified mode.

    Args:
        mode (str): 'train' or 'val'/'test'.
    """
    # Resize is applied first to ensure consistent dimensions
    base_transforms = [A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH)]

    if mode == "train":
        # Augmentations for training
        augmentations = [
            # Horizontal Flip (Time Inversion)
            A.HorizontalFlip(p=0.5),
            # Unstructured Cutout (CoarseDropout)
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
        ]
        pipeline = base_transforms + augmentations
    else:
        # Just resize for validation/test
        pipeline = base_transforms

    # Normalization and Tensor Conversion (Always applied)
    pipeline.extend([A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()])

    return A.Compose(pipeline)


def get_dataloaders(config, pseudo_labels_path=None, use_combined_train=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        config (Config): Configuration object.
        pseudo_labels_path (str, optional): Path to parquet/csv file containing pseudo-labels.
        use_combined_train (bool): If True, combines Train + Pseudo-Labeled Test for training (Student phase).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA)
    df_val = pd.read_csv(config.VAL_METADATA)
    df_test = pd.read_csv(config.TEST_METADATA)

    # Debug mode: reduce dataset size
    if config.debug:
        df_train = df_train.head(32)
        df_val = df_val.head(32)
        df_test = df_test.head(32)

    # 2. Handle Pseudo-Labels
    pseudo_labels_dict = None

    if pseudo_labels_path and os.path.exists(pseudo_labels_path):
        try:
            if pseudo_labels_path.endswith(".parquet"):
                df_pseudo = pd.read_parquet(pseudo_labels_path)
            else:
                df_pseudo = pd.read_csv(pseudo_labels_path)

            # Create dictionary: rec_id -> numpy array of probabilities
            # Assuming columns are 'rec_id' and 'species_0'...'species_18'
            label_cols = [c for c in df_pseudo.columns if c.startswith("species_")]

            pseudo_labels_dict = {}
            for _, row in df_pseudo.iterrows():
                rid = int(row["rec_id"])
                probs = row[label_cols].values.astype(np.float32)
                pseudo_labels_dict[rid] = probs

            print(
                f"Loaded {len(pseudo_labels_dict)} pseudo-labels from {pseudo_labels_path}"
            )

        except Exception as e:
            print(f"Error loading pseudo-labels: {e}")

    # 3. Combine Datasets if requested (Student Training)
    if use_combined_train and pseudo_labels_dict is not None:
        # We append the test dataframe to the train dataframe.
        # The Dataset class will handle looking up the labels from pseudo_labels_dict
        # for the test samples, and use hard labels from df_train for training samples.
        print("Combining Labeled Train and Pseudo-Labeled Test sets for training.")
        df_train_combined = pd.concat([df_train, df_test], ignore_index=True)
        train_dataset_df = df_train_combined
    else:
        train_dataset_df = df_train

    # 4. Create Datasets
    train_dataset = BirdDataset(
        df=train_dataset_df,
        img_dir=config.SPECTROGRAM_DIR,
        transform=get_transforms(mode="train"),
        pseudo_labels_dict=pseudo_labels_dict,
    )

    val_dataset = BirdDataset(
        df=df_val,
        img_dir=config.SPECTROGRAM_DIR,
        transform=get_transforms(mode="val"),
        pseudo_labels_dict=None,  # Validation always uses ground truth
    )

    test_dataset = BirdDataset(
        df=df_test,
        img_dir=config.SPECTROGRAM_DIR,
        transform=get_transforms(mode="test"),
        pseudo_labels_dict=None,  # Test set for inference (labels ignored/placeholder)
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
