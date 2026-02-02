import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.SEED)


def get_fragment_mip(
    fragment_id,
    volume_path,
    z_start=Config.Z_START,
    z_end=Config.Z_END,
    load_cached_data=True,
):
    """
    Computes or loads the Maximum Intensity Projection (MIP) for a specific fragment.

    Args:
        fragment_id (str): The ID of the fragment.
        volume_path (str): Relative path to the surface volume directory.
        z_start (int): Starting slice index (inclusive).
        z_end (int): Ending slice index (exclusive).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The 2D MIP image (H, W) with original dtype (uint16).
    """
    cache_filename = f"fragment_{fragment_id}_mip.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            mip = np.load(cache_path)
            return mip
        except Exception as e:
            print(
                f"Error loading cache for fragment {fragment_id}: {e}. Recomputing..."
            )

    # 2. Compute from scratch
    full_volume_dir = os.path.join(Config.INPUT_DIR, volume_path)
    slices = []

    # Iterate through the specified Z-slice range
    for z in range(z_start, z_end):
        slice_filename = f"{z:02d}.tif"
        slice_path = os.path.join(full_volume_dir, slice_filename)

        if os.path.exists(slice_path):
            img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                slices.append(img)

    if not slices:
        raise ValueError(
            f"No valid slices found for fragment {fragment_id} in {full_volume_dir} (range {z_start}-{z_end})"
        )

    # Stack along new axis and take maximum
    stack = np.stack(slices, axis=0)
    mip = np.max(stack, axis=0)

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, mip)

    return mip


def get_transforms(data="train"):
    """
    Returns albumentations transforms for training or validation.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                # Normalize: (x - mean) / std. Since we manually normalize to [0,1], max_pixel_value=1.0
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    return None


class InkDataset(Dataset):
    def __init__(self, metadata_path, transform=None, load_cached_data=True):
        """
        PyTorch Dataset for Vesuvius Ink Detection (MIP approach).

        Args:
            metadata_path (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Transforms to apply.
            load_cached_data (bool): Whether to use cached MIPs.
        """
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Caches for full fragment images to avoid repeated I/O during __getitem__
        self.fragment_mips = {}
        self.fragment_labels = {}

        # Identify unique fragments in this dataset
        unique_fragments = self.df["fragment_id"].unique()

        for frag_id in unique_fragments:
            # Get paths from the first occurrence in metadata
            frag_row = self.df[self.df["fragment_id"] == frag_id].iloc[0]

            # 1. Load MIP
            vol_path = frag_row["volume_path"]
            mip = get_fragment_mip(
                frag_id, vol_path, load_cached_data=self.load_cached_data
            )

            # Normalize MIP to [0, 1] float32 immediately to save processing time later
            # Original data is uint16
            mip = mip.astype(np.float32) / 65535.0
            self.fragment_mips[frag_id] = mip

            # 2. Load Label Mask (if available)
            if "label_path" in frag_row and pd.notna(frag_row["label_path"]):
                label_path = os.path.join(Config.INPUT_DIR, frag_row["label_path"])
                if os.path.exists(label_path):
                    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    # Binarize label (0 or 1)
                    label = (label > 0).astype(np.float32)
                    self.fragment_labels[frag_id] = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        # Retrieve full images from memory
        full_mip = self.fragment_mips[frag_id]

        # Calculate crop coordinates safely
        img_h, img_w = full_mip.shape
        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        # Crop Image
        image_patch = full_mip[y:y_end, x:x_end]

        # Pad if the crop is smaller than target size (e.g. at edges)
        pad_h = h - image_patch.shape[0]
        pad_w = w - image_patch.shape[1]

        if pad_h > 0 or pad_w > 0:
            image_patch = np.pad(
                image_patch,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # Crop Label if available
        mask_patch = None
        if frag_id in self.fragment_labels:
            full_label = self.fragment_labels[frag_id]
            mask_patch = full_label[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                mask_patch = np.pad(
                    mask_patch,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

        # Apply Augmentations
        if self.transform:
            if mask_patch is not None:
                augmented = self.transform(image=image_patch, mask=mask_patch)
                image_tensor = augmented["image"]
                mask_tensor = augmented["mask"]
            else:
                augmented = self.transform(image=image_patch)
                image_tensor = augmented["image"]
        else:
            # Manual conversion if no transform provided
            image_tensor = torch.from_numpy(image_patch).unsqueeze(0)  # Add channel dim
            if mask_patch is not None:
                mask_tensor = torch.from_numpy(mask_patch).float()

        if mask_patch is not None:
            # Ensure mask has channel dimension (1, H, W) for consistency with BCE/Dice losses
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            return image_tensor, mask_tensor
        else:
            return image_tensor
