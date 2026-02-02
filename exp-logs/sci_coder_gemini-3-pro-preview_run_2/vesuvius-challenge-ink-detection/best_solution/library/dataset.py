import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        A.Compose: The transform pipeline.
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
        return A.Compose([ToTensorV2()])


def load_and_process_fragment(fragment_id, volume_dir, load_cached_data=True):
    """
    Loads the 3D volume slices, computes overlapping MIPs, normalizes, and caches the result.

    Args:
        fragment_id (str): The ID of the fragment.
        volume_dir (str): Path to the directory containing .tif slices.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: A (H, W, 3) float32 array containing the processed MIPs.
    """
    cache_filename = f"{fragment_id}_mips.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            mips = np.load(cache_path)
            # Basic validation of shape
            if mips.ndim == 3 and mips.shape[2] == 3:
                return mips
            else:
                print(
                    f"Warning: Cached file {cache_path} has incorrect shape {mips.shape}. Recomputing."
                )
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}. Recomputing.")

    # 2. Compute from scratch
    # Determine the union of all slices needed to minimize file I/O
    all_slices = set()
    for start, end in Config.Z_RANGES:
        all_slices.update(range(start, end))

    min_slice = min(all_slices)
    max_slice = max(all_slices)

    # We load the block of slices into memory.
    # Assuming filenames are formatted as '00.tif', '01.tif', etc.
    # We need to determine the spatial dimensions first.
    first_slice_path = os.path.join(
        Config.INPUT_DIR, volume_dir, f"{min_slice:02d}.tif"
    )
    if not os.path.exists(first_slice_path):
        raise FileNotFoundError(f"Slice file not found: {first_slice_path}")

    ref_img = cv2.imread(first_slice_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        raise ValueError(f"Failed to read image: {first_slice_path}")

    h, w = ref_img.shape

    # Initialize volume block: (D, H, W) where D is the number of slices in the range [min_slice, max_slice]
    # We use a dictionary or a dense array. A dense array is better for MIP calculation.
    # Shift indices so min_slice maps to 0
    depth = (
        max_slice - min_slice + 1
    )  # +1 because max_slice is inclusive in the set logic above?
    # Wait, Config.Z_RANGES uses python range convention (start, end) where end is exclusive?
    # "Channel 1: Slices 20 to 32" -> usually implies range(20, 32) i.e. 20..31.
    # Let's assume standard Python range semantics for Config.Z_RANGES.

    # Re-evaluating set logic with range semantics
    all_indices = set()
    for start, end in Config.Z_RANGES:
        for z in range(start, end):
            all_indices.add(z)

    sorted_indices = sorted(list(all_indices))
    if not sorted_indices:
        raise ValueError("No slices defined in Z_RANGES.")

    # Map global z index to local array index
    z_to_local = {z: i for i, z in enumerate(sorted_indices)}

    volume_block = np.zeros((len(sorted_indices), h, w), dtype=np.uint16)

    for z in sorted_indices:
        path = os.path.join(Config.INPUT_DIR, volume_dir, f"{z:02d}.tif")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                volume_block[z_to_local[z]] = img
            else:
                print(f"Warning: Could not read slice {path}")
        else:
            print(f"Warning: Slice {path} does not exist")

    # Compute MIPs for each channel
    channels = []
    for start, end in Config.Z_RANGES:
        # Identify local indices for this range
        local_idxs = [z_to_local[z] for z in range(start, end) if z in z_to_local]

        if not local_idxs:
            # Should not happen if files exist
            mip = np.zeros((h, w), dtype=np.float32)
        else:
            # Extract slab
            slab = volume_block[local_idxs, :, :]
            # Max Intensity Projection
            mip = np.max(slab, axis=0)

        channels.append(mip)

    # Stack to (H, W, 3)
    mips = np.stack(channels, axis=-1).astype(np.float32)

    # Normalize
    mips = (mips - Config.PIXEL_MIN) / (Config.PIXEL_MAX - Config.PIXEL_MIN)
    mips = np.clip(mips, 0.0, 1.0)

    # Save to cache
    np.save(cache_path, mips)

    return mips


class InkDataset(Dataset):
    def __init__(self, metadata_df, mode="train", load_cached_data=True):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing metadata.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached .npy files for fragments.
        """
        self.mode = mode
        self.transform = get_transforms(mode)
        self.load_cached_data = load_cached_data

        # Data storage
        self.fragment_data = (
            {}
        )  # fragment_id -> {'mips': np.array, 'mask': np.array, 'label': np.array}
        self.samples = (
            []
        )  # List of dicts defining each sample (fragment_id, x, y, w, h)

        # 1. Prepare Fragments
        unique_fragments = metadata_df["fragment_id"].unique()

        for fid in unique_fragments:
            # Get fragment info from the first occurrence in df
            row = metadata_df[metadata_df["fragment_id"] == fid].iloc[0]

            # Load Volume MIPs
            mips = load_and_process_fragment(
                str(fid), row["volume_path"], load_cached_data=self.load_cached_data
            )

            # Load Mask
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Fallback for test if mask might be missing or different path logic?
                # Metadata generation ensures existence, but good to be safe.
                raise ValueError(f"Mask not found at {mask_path}")

            # Load Label (only for train/val)
            label = None
            if mode in ["train", "val"]:
                label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label is None:
                    raise ValueError(f"Label not found at {label_path}")
                # Binarize label
                label = (label > 0).astype(np.float32)

            # Binarize mask
            mask = (mask > 0).astype(np.float32)

            # Store
            self.fragment_data[str(fid)] = {"mips": mips, "mask": mask, "label": label}

        # 2. Prepare Samples (Patches)
        if mode == "test":
            # Generate sliding window patches for test fragments
            for fid in unique_fragments:
                frag_info = self.fragment_data[str(fid)]
                h, w = frag_info["mask"].shape

                # We pad the image to be divisible by TILE_SIZE if necessary,
                # but standard practice is to just tile and handle edges.
                # Here we strictly follow the grid.

                for y in range(0, h, Config.STRIDE):
                    for x in range(0, w, Config.STRIDE):
                        # Ensure we don't go out of bounds by clamping or padding?
                        # The standard inference loop often pads the image.
                        # Here, we will just define the crop.
                        # If the crop extends beyond the image, we will pad in __getitem__.
                        self.samples.append(
                            {
                                "fragment_id": str(fid),
                                "x": x,
                                "y": y,
                                "width": Config.TILE_SIZE,
                                "height": Config.TILE_SIZE,
                            }
                        )
        else:
            # Use provided patches in metadata
            for _, row in metadata_df.iterrows():
                self.samples.append(
                    {
                        "fragment_id": str(row["fragment_id"]),
                        "x": row["x"],
                        "y": row["y"],
                        "width": row["width"],
                        "height": row["height"],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fid = sample["fragment_id"]
        x, y = sample["x"], sample["y"]
        w, h = sample["width"], sample["height"]

        data = self.fragment_data[fid]
        full_mips = data["mips"]
        full_mask = data["mask"]
        full_label = data["label"]

        img_h, img_w = full_mask.shape

        # Calculate crop coordinates with padding handling
        # If crop goes outside, we pad with zeros
        pad_h = max(0, (y + h) - img_h)
        pad_w = max(0, (x + w) - img_w)

        # Actual crop coordinates
        crop_y_end = min(y + h, img_h)
        crop_x_end = min(x + w, img_w)

        # Crop images
        # mips is (H, W, 3)
        mips_crop = full_mips[y:crop_y_end, x:crop_x_end, :]
        mask_crop = full_mask[y:crop_y_end, x:crop_x_end]

        if full_label is not None:
            label_crop = full_label[y:crop_y_end, x:crop_x_end]
        else:
            label_crop = None

        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            mips_crop = np.pad(
                mips_crop,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            mask_crop = np.pad(
                mask_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )
            if label_crop is not None:
                label_crop = np.pad(
                    label_crop,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

        # Augmentations
        if self.transform:
            if label_crop is not None:
                augmented = self.transform(image=mips_crop, mask=label_crop)
                image = augmented["image"]
                label = augmented["mask"]
                # Ensure label has channel dim (1, H, W)
                if label.ndim == 2:
                    label = label.unsqueeze(0)
            else:
                augmented = self.transform(image=mips_crop)
                image = augmented["image"]
                label = torch.zeros((1, h, w), dtype=torch.float32)  # Dummy label
        else:
            # Manual conversion if no transform (should not happen given get_transforms)
            image = torch.from_numpy(mips_crop.transpose(2, 0, 1))
            if label_crop is not None:
                label = torch.from_numpy(label_crop).unsqueeze(0)
            else:
                label = torch.zeros((1, h, w), dtype=torch.float32)

        # We also return the mask to mask out loss/predictions on invalid pixels
        # Resize/Pad mask to tensor
        mask = torch.from_numpy(mask_crop).unsqueeze(0).float()

        return image, label, mask, idx
