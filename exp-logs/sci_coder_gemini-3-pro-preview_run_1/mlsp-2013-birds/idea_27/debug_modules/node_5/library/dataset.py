import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline based on the mode.

    Args:
        mode (str): 'train' or 'val'/'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if mode == "train":
        return A.Compose(
            [
                # High-Fidelity Resolution
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                # Augmentations
                A.HorizontalFlip(p=0.5),
                # Unstructured Cutout (Random rectangular masks)
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    fill_value=0,
                    p=0.5,
                ),
                # Normalization (ImageNet stats)
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test / Teacher Inference
        return A.Compose(
            [
                A.Resize(height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH),
                A.Normalize(mean=Config.MEAN, std=Config.STD),
                ToTensorV2(),
            ]
        )


class BirdDataset(Dataset):
    def __init__(
        self, metadata_path, mode="train", pseudo_labels_path=None, transforms=None
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV (train.csv, val.csv, test.csv).
            mode (str): 'train', 'val', or 'test'.
            pseudo_labels_path (str, optional): Path to parquet file containing pseudo labels.
                                                Expected format: index is rec_id, cols are probabilities.
            transforms (A.Compose, optional): Albumentations transforms.
        """
        self.mode = mode
        self.transforms = transforms if transforms else get_transforms(mode)

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Load Pseudo Labels if provided
        self.pseudo_labels = None
        if pseudo_labels_path and os.path.exists(pseudo_labels_path):
            try:
                # Load parquet
                df_pseudo = pd.read_parquet(pseudo_labels_path)
                # Ensure we can look up by rec_id
                if "rec_id" in df_pseudo.columns:
                    df_pseudo = df_pseudo.set_index("rec_id")
                self.pseudo_labels = df_pseudo
            except Exception as e:
                print(
                    f"Warning: Failed to load pseudo labels from {pseudo_labels_path}: {e}"
                )
                self.pseudo_labels = None

        # Pre-compute file paths and targets
        self.data_infos = []

        # Identify label columns (species_0 to species_18)
        self.label_cols = [f"species_{i}" for i in range(Config.NUM_CLASSES)]

        for idx, row in self.df.iterrows():
            rec_id = int(row["rec_id"])

            # Map wav path to spectrogram path
            # Metadata file_path: essential_data/src_wavs/filename.wav
            wav_rel_path = row["file_path"]
            wav_basename = os.path.basename(wav_rel_path)
            bmp_basename = os.path.splitext(wav_basename)[0] + ".bmp"
            img_path = os.path.join(Config.SPECTROGRAM_DIR, bmp_basename)

            # Determine targets
            targets = None
            if self.pseudo_labels is not None and rec_id in self.pseudo_labels.index:
                # Use soft targets from pseudo labels
                try:
                    targets = self.pseudo_labels.loc[rec_id].values.astype(np.float32)
                    # Safety check on shape
                    if targets.shape[0] != Config.NUM_CLASSES:
                        targets = row[self.label_cols].values.astype(np.float32)
                except:
                    targets = row[self.label_cols].values.astype(np.float32)
            else:
                # Use hard targets from metadata
                targets = row[self.label_cols].values.astype(np.float32)

            self.data_infos.append(
                {"img_path": img_path, "targets": targets, "rec_id": rec_id}
            )

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]
        img_path = info["img_path"]
        targets = info["targets"]
        rec_id = info["rec_id"]

        # Dynamic Loading
        # Load as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Fallback: Create a black image
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)

        # Input Adaptation: Channel Replication
        # Convert Grayscale to RGB (3 channels) to use ImageNet normalization
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented["image"]

        return img, torch.tensor(targets), rec_id


class CombinedDataset(Dataset):
    """
    Concatenates two datasets (e.g., Labeled Train and Pseudo-Labeled Test).
    """

    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)

    def __len__(self):
        return self.len_a + self.len_b

    def __getitem__(self, idx):
        if idx < self.len_a:
            return self.dataset_a[idx]
        else:
            return self.dataset_b[idx - self.len_a]
