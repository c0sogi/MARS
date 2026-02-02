import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import (
    INPUT_DIR,
    TILE_SIZE,
    Z_START,
    SLAB_SIZE,
    STRIDE,
    Z_JITTER_RANGE,
    NUM_CHANNELS,
)
from library.utils import min_max_normalize


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


class InkDataset(Dataset):
    def __init__(self, metadata_df, mode="train", transforms=None):
        """
        Dataset for Vesuvius Ink Detection.

        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Transforms to apply.
        """
        self.mode = mode
        self.transforms = transforms
        self.samples = []

        # Pre-process metadata into a list of samples
        if self.mode in ["train", "val"]:
            # Training/Validation: Patches are already defined in the CSV
            self.samples = metadata_df.to_dict("records")
        elif self.mode == "test":
            # Test: We need to generate tiles for the whole fragment
            # The test.csv contains fragment-level info, not patches.
            for _, row in metadata_df.iterrows():
                frag_id = row["fragment_id"]
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                volume_path = row["volume_path"]  # Relative path

                # Load mask to get dimensions
                # We use cv2.IMREAD_GRAYSCALE to get (H, W)
                if not os.path.exists(mask_path):
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                h, w = mask.shape

                # Generate tiles
                # We use a sliding window. To ensure full coverage without padding,
                # if the last tile goes out of bounds, we shift it back.
                y_positions = list(range(0, h - TILE_SIZE, TILE_SIZE))
                if (h % TILE_SIZE) != 0 or not y_positions:
                    y_positions.append(h - TILE_SIZE)

                x_positions = list(range(0, w - TILE_SIZE, TILE_SIZE))
                if (w % TILE_SIZE) != 0 or not x_positions:
                    x_positions.append(w - TILE_SIZE)

                # Ensure non-negative coords for small images (unlikely but safe)
                y_positions = [max(0, y) for y in y_positions]
                x_positions = [max(0, x) for x in x_positions]

                # Remove duplicates (e.g. if image size is exact multiple)
                y_positions = sorted(list(set(y_positions)))
                x_positions = sorted(list(set(x_positions)))

                for y in y_positions:
                    for x in x_positions:
                        self.samples.append(
                            {
                                "fragment_id": frag_id,
                                "x": x,
                                "y": y,
                                "width": TILE_SIZE,
                                "height": TILE_SIZE,
                                "mask_path": row["mask_path"],
                                "volume_path": volume_path,
                                # Test samples don't have label_path or has_ink
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def _get_z_indices(self):
        """
        Generates the list of Z-slice indices for the 3 channels.
        Applies Z-Jitter if in training mode.
        """
        # Determine start index
        start = Z_START
        if self.mode == "train":
            # Apply jitter
            jitter = random.randint(-Z_JITTER_RANGE, Z_JITTER_RANGE)
            start += jitter

        # Define the 3 overlapping slabs
        # Channel 1: [start, start + SLAB_SIZE)
        # Channel 2: [start + STRIDE, start + STRIDE + SLAB_SIZE)
        # Channel 3: [start + 2*STRIDE, start + 2*STRIDE + SLAB_SIZE)

        channels = []
        for i in range(NUM_CHANNELS):
            slab_start = start + (i * STRIDE)
            slab_end = slab_start + SLAB_SIZE

            # Generate range of indices
            indices = list(range(slab_start, slab_end))

            # Clamp indices to valid range [0, 64]
            # Files are 00.tif to 64.tif
            indices = [max(0, min(64, idx)) for idx in indices]
            channels.append(indices)

        return channels

    def _load_volume_patch(self, volume_rel_path, x, y, w, h, z_indices_list):
        """
        Loads the 3-channel volume patch.
        z_indices_list: List of lists, where each sublist contains z-indices for a channel.
        """
        volume_dir = os.path.join(INPUT_DIR, volume_rel_path)

        # Identify all unique Z indices needed to minimize IO
        all_z = sorted(list(set([z for channel in z_indices_list for z in channel])))

        # Cache loaded slices: {z_index: image_patch}
        slice_cache = {}

        for z in all_z:
            slice_filename = f"{z:02d}.tif"
            slice_path = os.path.join(volume_dir, slice_filename)

            if not os.path.exists(slice_path):
                # Fallback: create empty slice if missing
                slice_cache[z] = np.zeros((h, w), dtype=np.uint8)
                continue

            # Load full slice and crop
            # Note: For production with large TIFs, rasterio or memory mapping is better.
            # Given the constraints and standard libraries, cv2 is used.
            # We rely on OS file caching for performance.
            img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                slice_cache[z] = np.zeros((h, w), dtype=np.uint8)
            else:
                # Handle boundary checks for cropping
                img_h, img_w = img.shape
                y_end = min(y + h, img_h)
                x_end = min(x + w, img_w)

                patch = img[y:y_end, x:x_end]

                # Pad if patch is smaller than requested (edge case)
                if patch.shape[0] < h or patch.shape[1] < w:
                    pad_h = h - patch.shape[0]
                    pad_w = w - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="constant")

                slice_cache[z] = patch

        # Construct Channels via MIP (Maximum Intensity Projection)
        channel_images = []
        for indices in z_indices_list:
            # Stack slices for this channel: (D, H, W)
            stack = np.stack([slice_cache[z] for z in indices], axis=0)
            # MIP: Max along depth axis
            mip = np.max(stack, axis=0)
            channel_images.append(mip)

        # Stack channels: (H, W, 3)
        image = np.stack(channel_images, axis=-1)
        return image

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Get Z-indices (with jitter if train)
        z_indices_list = self._get_z_indices()

        # 2. Load Volume (3-channel MIP)
        image = self._load_volume_patch(
            sample["volume_path"],
            sample["x"],
            sample["y"],
            sample["width"],
            sample["height"],
            z_indices_list,
        )

        # 3. Normalize to [0, 1]
        # We normalize the float image. Input is likely uint8 or uint16.
        image = image.astype(np.float32)
        image = min_max_normalize(image)

        # 4. Load Label/Mask (if available)
        mask = None
        if "label_path" in sample and isinstance(sample["label_path"], str):
            label_path = os.path.join(INPUT_DIR, sample["label_path"])
            if os.path.exists(label_path):
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label_img is not None:
                    # Crop
                    x, y, w, h = (
                        sample["x"],
                        sample["y"],
                        sample["width"],
                        sample["height"],
                    )
                    img_h, img_w = label_img.shape
                    y_end = min(y + h, img_h)
                    x_end = min(x + w, img_w)

                    label_patch = label_img[y:y_end, x:x_end]

                    # Pad if necessary
                    if label_patch.shape[0] < h or label_patch.shape[1] < w:
                        pad_h = h - label_patch.shape[0]
                        pad_w = w - label_patch.shape[1]
                        label_patch = np.pad(
                            label_patch, ((0, pad_h), (0, pad_w)), mode="constant"
                        )

                    # Binarize (0 or 1)
                    mask = (label_patch > 0).astype(np.float32)

        # If no label found (e.g. test mode), create dummy mask for transform compatibility
        if mask is None:
            mask = np.zeros((sample["height"], sample["width"]), dtype=np.float32)

        # 5. Augmentations
        if self.transforms:
            # Albumentations expects image in (H, W, C)
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 6. Prepare Output
        # Image is now (3, H, W) tensor due to ToTensorV2
        # Mask is (H, W) tensor

        output = {
            "image": image,
            "fragment_id": sample["fragment_id"],
            "x": sample["x"],
            "y": sample["y"],
        }

        if self.mode in ["train", "val"]:
            # Add channel dim to mask: (1, H, W)
            output["label"] = mask.unsqueeze(0)

        return output
