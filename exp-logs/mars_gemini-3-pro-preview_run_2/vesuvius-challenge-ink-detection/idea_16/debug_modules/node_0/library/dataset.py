import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_image, process_fragment_mips, load_metadata


def get_transforms(split="train"):
    """
    Returns the Albumentations transform pipeline for the specified split.

    Args:
        split (str): 'train', 'validation', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
    """
    if split == "train":
        return A.Compose(
            [
                # Geometric Augmentations only
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Normalize to ensure float32 and consistent range (Identity scaling as data is already [0,1])
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                # No geometric augmentation for validation/test
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )


class InkDataset(Dataset):
    """
    PyTorch Dataset for Vesuvius Ink Detection.

    Handles:
    - Loading 3-channel MIP slabs (Overlapping Stratified Depth Projection).
    - Caching full fragments in memory for fast patch extraction.
    - Geometric augmentations.
    """

    def __init__(self, metadata, split="train", z_start=None, transforms=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing patch coordinates and file paths.
            split (str): 'train', 'validation', or 'test'.
            z_start (int, optional): The starting Z-slice for the projection.
                                     Defaults to Config.TRAIN_Z_START (20).
            transforms (A.Compose, optional): Albumentations transforms.
        """
        self.metadata = metadata
        self.split = split
        self.z_start = z_start if z_start is not None else Config.TRAIN_Z_START
        self.transforms = (
            transforms if transforms is not None else get_transforms(split)
        )

        # Memory Cache for full fragment images
        self.frag_images = {}
        self.frag_labels = {}
        self.frag_masks = {}

        # Identify unique fragments to preload
        if "fragment_id" in metadata.columns:
            fragment_ids = metadata["fragment_id"].unique()
        else:
            fragment_ids = []

        # Pre-load data into memory
        for fid in fragment_ids:
            # Get the first row for this fragment to retrieve paths
            frag_row = metadata[metadata["fragment_id"] == fid].iloc[0]

            # 1. Load/Compute MIPs (3, H, W)
            # process_fragment_mips handles the caching logic (load .npy or compute & save)
            # It returns float32 [0, 1]
            mips = process_fragment_mips(
                fragment_id=str(fid),
                volume_path=frag_row["volume_path"],
                z_start=self.z_start,
                load_cached_data=True,
            )

            # Transpose to (H, W, 3) for Albumentations
            self.frag_images[fid] = np.transpose(mips, (1, 2, 0))

            # 2. Load Label (if available)
            if "label_path" in frag_row and pd.notna(frag_row["label_path"]):
                lbl = load_image(frag_row["label_path"], grayscale=True)
                if lbl is not None:
                    # Binarize and convert to float32
                    self.frag_labels[fid] = (lbl > 0).astype(np.float32)

            # 3. Load Mask (if available)
            if "mask_path" in frag_row and pd.notna(frag_row["mask_path"]):
                msk = load_image(frag_row["mask_path"], grayscale=True)
                if msk is not None:
                    self.frag_masks[fid] = (msk > 0).astype(np.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        fid = row["fragment_id"]

        # Retrieve full fragment data from cache
        full_image = self.frag_images[fid]

        # Determine crop coordinates
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # Crop Image
        # Note: full_image is (H, W, 3)
        image_patch = full_image[y : y + h, x : x + w, :]

        # Prepare Label
        label_patch = None
        if fid in self.frag_labels:
            full_label = self.frag_labels[fid]
            label_patch = full_label[y : y + h, x : x + w]
        else:
            # Create dummy label for inference if missing
            label_patch = np.zeros((h, w), dtype=np.float32)

        # Apply Transforms
        # Albumentations expects image=(H,W,C) and mask=(H,W)
        augmented = self.transforms(image=image_patch, mask=label_patch)

        image_tensor = augmented["image"]  # (3, H, W)
        label_tensor = augmented["mask"]  # (H, W)

        # Unsqueeze label to be (1, H, W) for BCE Loss
        label_tensor = label_tensor.unsqueeze(0)

        return image_tensor, label_tensor


def get_dataset(split="train", z_start=None):
    """
    Factory function to create an InkDataset for a specific split.

    Args:
        split (str): 'train', 'validation', or 'test'.
        z_start (int, optional): Override Z-start index.

    Returns:
        InkDataset: The initialized dataset.
    """
    df = load_metadata(split)
    return InkDataset(df, split=split, z_start=z_start)
