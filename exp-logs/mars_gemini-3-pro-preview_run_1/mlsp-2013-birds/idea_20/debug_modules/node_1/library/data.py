import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import set_seed


class BirdDataset(Dataset):
    """
    PyTorch Dataset for loading bird spectrograms and labels.
    Supports dynamic loading, channel replication, and soft/hard labels.
    """

    def __init__(self, df, transforms=None, img_dir=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (rec_id, file_path, labels).
            transforms (albumentations.Compose): Transformations to apply.
            img_dir (str): Root directory for images.
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_dir = img_dir

        # Identify label columns (species_0 to species_18)
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        # Check if label columns exist in dataframe
        if all(col in self.df.columns for col in self.label_cols):
            self.labels = self.df[self.label_cols].values.astype(np.float32)
        else:
            # For test set without labels, create dummy zeros
            self.labels = np.zeros((len(self.df), Config.NUM_CLASSES), dtype=np.float32)

        # Pre-compute full image paths to save time in __getitem__
        # The file_path in metadata is relative to input/ (e.g., essential_data/src_wavs/...)
        # But we need spectrograms which are in supplemental_data/spectrograms/
        # The filename is the same base name but with .bmp extension.
        self.image_paths = []
        for rel_path in self.df["file_path"]:
            wav_basename = os.path.basename(rel_path)
            bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
            full_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)
            self.image_paths.append(full_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        # Use IMREAD_GRAYSCALE first
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            # Fallback for missing images (should not happen based on EDA, but for safety)
            # Create a black image of expected size
            image = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Channel Replication: Grayscale -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            # Minimal transform if none provided (Resize + ToTensor)
            # But usually transforms are always provided.
            pass

        # Get Label
        label = self.labels[idx]

        # Return index as well for tracking
        rec_id = self.df.iloc[idx]["rec_id"]

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "rec_id": torch.tensor(rec_id, dtype=torch.long),
        }


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms for the specified phase.

    Args:
        phase (str): 'train' or 'valid'.
    """
    if phase == "train":
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.HorizontalFlip(p=0.5),  # Time inversion
                # Unstructured Cutout (CoarseDropout)
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
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


def get_dataloaders(pseudo_labels_path=None):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        pseudo_labels_path (str, optional): Path to parquet file containing pseudo-labels.
                                            If provided, these are merged with training data.

    Returns:
        dict: Dictionary containing 'train', 'val', 'test' DataLoaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle Pseudo-labels for Student Training
    if pseudo_labels_path and os.path.exists(pseudo_labels_path):
        print(f"Loading pseudo-labels from {pseudo_labels_path}...")
        pseudo_df = pd.read_parquet(pseudo_labels_path)

        # Ensure pseudo_df has the same columns as train_df
        # train_df has: rec_id, file_path, labels_str, species_0...species_18
        # pseudo_df likely has: rec_id, species_0...species_18 (probabilities)

        # We need to map file_path for pseudo-labeled test data.
        # The pseudo-labels correspond to the test set (Fold 1).
        # We can merge pseudo_df with test_df to get file paths.

        # 1. Drop existing zero-labels in test_df
        label_cols = [c for c in test_df.columns if c.startswith("species_")]
        test_df_clean = test_df.drop(columns=label_cols)

        # 2. Merge with pseudo_df on rec_id
        # pseudo_df should contain rec_id and the soft label columns
        combined_test_df = pd.merge(test_df_clean, pseudo_df, on="rec_id", how="inner")

        # 3. Concatenate Train DF (Hard labels) and Combined Test DF (Soft labels)
        # Ensure column order matches

        # For train_df, labels are 0 or 1.
        # For combined_test_df, labels are floats 0.0 to 1.0.

        # Align columns
        common_cols = ["rec_id", "file_path"] + label_cols

        train_subset = train_df[common_cols].copy()
        pseudo_subset = combined_test_df[common_cols].copy()

        final_train_df = pd.concat(
            [train_subset, pseudo_subset], axis=0, ignore_index=True
        )
        print(
            f"Combined Labeled Train ({len(train_subset)}) + Pseudo-Labeled Test ({len(pseudo_subset)})"
        )
        print(f"Total Training Samples: {len(final_train_df)}")

        train_dataset = BirdDataset(
            final_train_df, transforms=get_transforms("train"), img_dir=Config.INPUT_DIR
        )
    else:
        # Standard Training (Teachers)
        print(f"Loading standard training data: {len(train_df)} samples")
        train_dataset = BirdDataset(
            train_df, transforms=get_transforms("train"), img_dir=Config.INPUT_DIR
        )

    # Validation Dataset
    val_dataset = BirdDataset(
        val_df, transforms=get_transforms("valid"), img_dir=Config.INPUT_DIR
    )

    # Test Dataset
    test_dataset = BirdDataset(
        test_df, transforms=get_transforms("valid"), img_dir=Config.INPUT_DIR
    )

    # DataLoaders
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

    return {"train": train_loader, "val": val_loader, "test": test_loader}
