import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.data_processing import load_fragment_slab


class InkDataset(Dataset):
    """
    Dataset for Vesuvius Ink Detection.
    Loads 3D overlapping slabs and 2D binary labels based on generated metadata.
    """

    def __init__(self, mode: str = "train"):
        """
        Args:
            mode: 'train' or 'validation'. Determines which metadata file to load
                  and which augmentations to apply.
        """
        self.mode = mode
        self.is_train = mode == "train"

        # 1. Select Metadata File
        if self.is_train:
            self.metadata_path = os.path.join(Config.METADATA_DIR, "train.csv")
        elif mode == "validation":
            self.metadata_path = os.path.join(Config.METADATA_DIR, "validation.csv")
        else:
            raise ValueError(f"Mode must be 'train' or 'validation', got {mode}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # 2. RAM Caching of Fragments
        # Pre-load full fragment slabs and labels to avoid repeated disk I/O during epochs.
        # The dataset size (a few GBs) fits comfortably in the provided 220 GB RAM.
        self.fragment_slabs = {}
        self.fragment_labels = {}

        unique_fragments = self.df["fragment_id"].unique()

        for frag_id in unique_fragments:
            # Get paths from the first entry for this fragment in the metadata
            frag_row = self.df[self.df["fragment_id"] == frag_id].iloc[0]

            # Use string conversion to ensure consistency (pandas might infer int)
            frag_id_str = str(frag_id)
            volume_path = frag_row["volume_path"]
            label_path = frag_row["label_path"]

            # A. Load Input Slab (3D -> 2D 3-channel MIP)
            # We use the fixed TRAIN_Z_START for both training and validation
            # to evaluate the specific depth configuration.
            # load_fragment_slab handles disk caching (npy files).
            slab = load_fragment_slab(
                fragment_id=frag_id_str,
                volume_path=volume_path,
                z_start=Config.TRAIN_Z_START,
                load_cached_data=True,
            )
            self.fragment_slabs[frag_id] = slab

            # B. Load Ground Truth Label
            full_label_path = os.path.join(Config.INPUT_DIR, label_path)
            if not os.path.exists(full_label_path):
                raise FileNotFoundError(f"Label file not found: {full_label_path}")

            # Load as grayscale (H, W)
            label_img = cv2.imread(full_label_path, cv2.IMREAD_GRAYSCALE)
            if label_img is None:
                raise ValueError(f"Failed to read label image: {full_label_path}")

            # Normalize to binary [0, 1] float32
            label_img = (label_img > 0).astype(np.float32)
            self.fragment_labels[frag_id] = label_img

        # 3. Define Augmentations
        # Strictly geometric augmentations as per requirements.
        if self.is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # RandomRotate90 rotates by 0, 90, 180, 270 degrees
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        else:
            self.transform = A.Compose([ToTensorV2(transpose_mask=True)])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Returns:
            image: Tensor of shape (3, H, W)
            mask: Tensor of shape (1, H, W)
        """
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Retrieve cached full fragment data
        # Note: frag_id in dataframe matches keys in dict (int/str handling done implicitly if types match)
        full_slab = self.fragment_slabs[frag_id]
        full_label = self.fragment_labels[frag_id]

        # Crop the specific patch
        # Slicing handles boundaries gracefully, but metadata ensures validity.
        image_patch = full_slab[y : y + h, x : x + w, :]
        label_patch = full_label[y : y + h, x : x + w]

        # Apply Augmentations
        # Albumentations expects image (H,W,C) and mask (H,W)
        augmented = self.transform(image=image_patch, mask=label_patch)

        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"]

        # Ensure mask has channel dimension (1, H, W) for loss compatibility
        # ToTensorV2 converts 2D mask to (H, W) tensor, so we unsqueeze.
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor
