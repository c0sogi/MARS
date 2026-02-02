import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset

from library.config import Config
from library.data_io import read_dicom_robust, resize_image
from library.data_core import (
    get_roi_anchor_indices,
    get_sorted_image_files,
    normalize_independent,
)


def get_transforms(phase: str):
    """
    Returns the Albumentations transform pipeline based on the phase.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Rotate +/- 15 degrees, keep size, pad with 0 (black)
                # This matches the normalized background (0)
                A.Rotate(
                    limit=Config.ROTATION_DEGREES,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5,
                ),
            ]
        )
    return None


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS21 Glioblastoma Classification.

    Implements:
    1. Metadata-based loading.
    2. Raw-Selected ROI Anchor logic (via data_core).
    3. Multi-modality Stacking (4 modalities * 3 slices = 12 channels).
    4. Independent Per-Channel Normalization.
    5. Geometric Augmentations (Train only).
    """

    def __init__(
        self, df: pd.DataFrame, phase: str = "train", load_cached_data: bool = True
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (BraTS21ID, paths, labels).
            phase (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached ROI anchors.
        """
        self.df = df.reset_index(drop=True)
        self.phase = phase
        self.transform = get_transforms(phase)
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

        # Pre-compute or load ROI anchors
        # This ensures we don't re-scan folders every epoch
        # The caching logic is encapsulated in data_core.get_roi_anchor_indices
        self.anchor_map = get_roi_anchor_indices(
            self.df, load_cached_data=load_cached_data
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # ----------------------------------------------------------------------
        # 1. Determine Target
        # ----------------------------------------------------------------------
        if self.phase == "test":
            # For test, we return the ID to map predictions later
            target = brats_id
        else:
            # For train/val, return the binary label
            target = torch.tensor(row["MGMT_value"], dtype=torch.float32)

        # ----------------------------------------------------------------------
        # 2. Determine Slice Indices
        # ----------------------------------------------------------------------
        # Get the anchor calculated from FLAIR
        anchor_idx = self.anchor_map.get(brats_id, 0)

        # Define neighbors based on fixed stride
        stride = Config.STRIDE
        # Order: [Previous, Anchor, Next]
        # We use [Anchor - Stride, Anchor, Anchor + Stride]
        slice_indices = [anchor_idx - stride, anchor_idx, anchor_idx + stride]

        # ----------------------------------------------------------------------
        # 3. Load & Stack Images
        # ----------------------------------------------------------------------
        channels = []

        for mod in self.modalities:
            # Construct full path
            # Metadata paths are relative (e.g., "train/00000/FLAIR")
            mod_path_rel = row[f"path_{mod}"]
            mod_dir = os.path.join(Config.INPUT_DIR, mod_path_rel)

            # Get all files to handle indexing
            files = get_sorted_image_files(mod_dir)
            num_files = len(files)

            for s_idx in slice_indices:
                # Cite solution_lesson_node_00060: Zero Padding vs Edge Clamping
                # Use Zero Padding for out-of-bounds indices to signal physical boundaries
                if num_files > 0 and 0 <= s_idx < num_files:
                    file_path = os.path.join(mod_dir, files[s_idx])

                    # Robust Load (uint16) -> Resize (float32)
                    raw_img = read_dicom_robust(file_path)
                    img = resize_image(raw_img, target_size=Config.IMG_SIZE)
                else:
                    # Zero Padding for out-of-bounds or missing directories
                    img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

                channels.append(img)

        # Stack into tensor: Shape (12, H, W)
        # 4 Modalities * 3 Slices = 12 Channels
        img_tensor = np.stack(channels, axis=0)

        # ----------------------------------------------------------------------
        # 4. Normalization
        # ----------------------------------------------------------------------
        # Apply Independent Min-Max Scaling per channel
        # This preserves local contrast and dynamic range
        img_tensor = normalize_independent(img_tensor)

        # ----------------------------------------------------------------------
        # 5. Augmentation
        # ----------------------------------------------------------------------
        if self.transform:
            # Albumentations expects (H, W, C)
            img_hwc = np.transpose(img_tensor, (1, 2, 0))

            augmented = self.transform(image=img_hwc)["image"]

            # Transpose back to (C, H, W)
            img_tensor = np.transpose(augmented, (2, 0, 1))

        # ----------------------------------------------------------------------
        # 6. Final Conversion
        # ----------------------------------------------------------------------
        return torch.from_numpy(img_tensor).float(), target
