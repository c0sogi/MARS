import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from library.config import (
    INPUT_DIR,
    IMG_SIZE,
    SLICE_STRIDE,
    ROI_DEPTH_MIN,
    ROI_DEPTH_MAX,
    compute_roi_anchors,
)
from library.utils import read_dicom_robust, resize_image, normalize_minmax


class MGMTDataset(Dataset):
    def __init__(self, df, anchor_dict=None, phase="train"):
        self.df = df
        self.anchor_dict = anchor_dict if anchor_dict is not None else {}
        self.phase = phase
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

        # Define augmentations with Reflection Padding as requested
        if self.phase == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # Random rotation +/- 15 degrees with reflection padding
                    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, value=0, p=0.5),
                ]
            )
        else:
            self.transform = None

    def _select_anchor_slice(self, subject_id, flair_path):
        """
        Determines the geometric center using Sum of Intensity on raw FLAIR pixels
        within the 15-85% depth bounds.
        """
        if not os.path.exists(flair_path):
            return 0

        files = sorted(
            [f for f in os.listdir(flair_path) if f.endswith(".dcm")],
            key=lambda x: int(x.split("-")[-1].split(".")[0]) if "-" in x else 0,
        )
        num_slices = len(files)
        if num_slices == 0:
            return 0

        start_idx = int(num_slices * ROI_DEPTH_MIN)
        end_idx = int(num_slices * ROI_DEPTH_MAX)

        # Handle small volumes where range might be invalid
        if start_idx >= end_idx:
            start_idx, end_idx = 0, num_slices

        max_intensity = -1
        best_idx = 0

        # Iterate through the valid range
        for i in range(start_idx, end_idx):
            img_path = os.path.join(flair_path, files[i])
            img = read_dicom_robust(img_path)
            # Use raw pixel sum
            current_intensity = np.sum(img)

            if current_intensity > max_intensity:
                max_intensity = current_intensity
                best_idx = i

        return best_idx

    def _load_volume(self, row, anchor_idx):
        """
        Extracts 5 slices per modality with a stride of 3 using edge clamping.
        Returns a numpy array of shape (20, H, W).
        """
        # Offsets for 5 slices: [-6, -3, 0, +3, +6]
        offsets = [
            -2 * SLICE_STRIDE,
            -1 * SLICE_STRIDE,
            0,
            1 * SLICE_STRIDE,
            2 * SLICE_STRIDE,
        ]

        channels = []

        for mod in self.modalities:
            mod_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])

            # Get sorted file list
            if os.path.exists(mod_path):
                files = sorted(
                    [f for f in os.listdir(mod_path) if f.endswith(".dcm")],
                    key=lambda x: (
                        int(x.split("-")[-1].split(".")[0]) if "-" in x else 0
                    ),
                )
            else:
                files = []

            num_files = len(files)

            for o in offsets:
                target_idx = anchor_idx + o
                # Edge clamping: replicate boundary slices for out-of-bounds
                read_idx = max(0, min(target_idx, num_files - 1))

                if num_files > 0:
                    img_path = os.path.join(mod_path, files[read_idx])
                    img = read_dicom_robust(img_path)
                else:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

                # Resize using Area Interpolation (handled in utils)
                img = resize_image(img, (IMG_SIZE, IMG_SIZE))

                # Normalize [0, 1] per channel
                img = normalize_minmax(img)

                channels.append(img)

        # Stack channels: (20, 224, 224)
        # Order: FLAIR(5), T1w(5), T1wCE(5), T2w(5)
        volume = np.stack(channels, axis=0)
        return volume

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = row["BraTS21ID"]

        # Determine anchor: Use dict if available, else compute fallback
        if subject_id in self.anchor_dict:
            anchor_idx = self.anchor_dict[subject_id]
        else:
            flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
            anchor_idx = self._select_anchor_slice(subject_id, flair_path)

        # Load the dense volume
        volume = self._load_volume(row, anchor_idx)  # Shape: (20, H, W)

        # Apply Augmentations
        if self.transform:
            # Albumentations expects (H, W, C), so we transpose
            volume_hwc = np.transpose(volume, (1, 2, 0))
            augmented = self.transform(image=volume_hwc)["image"]
            # Transpose back to (C, H, W)
            volume = np.transpose(augmented, (2, 0, 1))

        # Convert to Tensor
        tensor = torch.from_numpy(volume).float()

        # Get Target
        target = row["MGMT_value"] if "MGMT_value" in row else 0.5
        return tensor, torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return len(self.df)


def get_dataloader(
    df, batch_size, phase="train", load_cached_anchors=True, num_workers=4
):
    """
    Factory function to create the DataLoader.
    Computes/Loads anchors first to minimize runtime overhead.
    """
    # Attempt to load cached anchors using the config utility
    # Note: If the cache exists but doesn't contain current IDs (e.g. test set),
    # the Dataset's _select_anchor_slice fallback will handle it.
    anchor_dict = compute_roi_anchors(df, load_cached_data=load_cached_anchors)

    dataset = MGMTDataset(df, anchor_dict=anchor_dict, phase=phase)

    shuffle = phase == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
