import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config, set_seed

# Ensure reproducibility
set_seed(Config.SEED)


def process_and_cache_slab(
    fragment_id, volume_rel_path, z_start, load_cached_data=True
):
    """
    Loads or generates the 3-channel MIP slab for a given fragment and Z-depth.
    Implements the strict caching logic required.

    Args:
        fragment_id (str): ID of the fragment (e.g., '1', 'a').
        volume_rel_path (str): Relative path to the surface_volume directory.
        z_start (int): The starting Z-slice index.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The processed 3-channel slab of shape (H, W, 3).
    """
    # Define cache path
    cache_filename = f"frag_{fragment_id}_slab_{z_start}.npy"
    cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            slab = np.load(cache_path)
            # Basic validation of shape
            if slab.ndim == 3 and slab.shape[2] == 3:
                return slab
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Regenerating...")

    # 2. Process from scratch
    full_volume_path = os.path.join(Config.INPUT_DIR, volume_rel_path)

    # Define channel ranges based on Config
    # Channel 0: z_start -> z_start + MIP_DEPTH
    # Channel 1: z_start + MIP_STRIDE -> z_start + MIP_STRIDE + MIP_DEPTH
    # Channel 2: z_start + 2*MIP_STRIDE -> z_start + 2*MIP_STRIDE + MIP_DEPTH

    channels = []
    offsets = [0, Config.MIP_STRIDE, 2 * Config.MIP_STRIDE]

    # We need to determine image dimensions first
    # Load the first slice to get H, W
    first_slice_path = os.path.join(full_volume_path, f"{z_start:02d}.tif")
    if not os.path.exists(first_slice_path):
        # Fallback for test set if numbering is different or file missing
        # But assuming standard format
        raise FileNotFoundError(f"Slice not found: {first_slice_path}")

    ref_img = cv2.imread(first_slice_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        raise ValueError(f"Could not read image: {first_slice_path}")
    h, w = ref_img.shape

    for offset in offsets:
        start = z_start + offset
        end = start + Config.MIP_DEPTH

        # Load slices for this channel
        slice_imgs = []
        for z in range(start, end):
            path = os.path.join(full_volume_path, f"{z:02d}.tif")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    # Pad with zeros if read fails
                    img = np.zeros((h, w), dtype=np.uint16)
            else:
                # Pad with zeros if out of bounds
                img = np.zeros((h, w), dtype=np.uint16)
            slice_imgs.append(img)

        # Stack and compute MIP (Maximum Intensity Projection)
        stack = np.stack(slice_imgs, axis=0)  # (D, H, W)
        mip = np.max(stack, axis=0)  # (H, W)
        channels.append(mip)

    # Stack channels to (H, W, 3)
    slab = np.stack(channels, axis=-1)

    # Normalize to [0, 1] and convert to float32
    slab = slab.astype(np.float32)
    slab = (slab - Config.PIXEL_MIN) / (Config.PIXEL_MAX - Config.PIXEL_MIN)
    slab = np.clip(slab, 0.0, 1.0)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.save(cache_path, slab)

    return slab


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
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


class InkDataset(Dataset):
    """
    Dataset for Vesuvius Ink Detection.
    Handles loading 3D slabs, cropping, and augmentation.
    """

    def __init__(
        self,
        metadata_path,
        mode="train",
        z_start=Config.Z_START,
        load_cached_data=True,
        limit_size=None,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'.
            z_start (int): Starting Z-slice for the slab.
            load_cached_data (bool): Whether to use cached .npy files.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.z_start = z_start
        self.transform = get_transforms(mode)

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path)

        # Pre-load all fragment slabs into memory
        # This assumes the total size fits in RAM (which it does for this task: ~2-5GB per fragment)
        self.fragment_slabs = {}
        self.fragment_masks = {}
        self.fragment_labels = {}

        unique_fragments = self.df["fragment_id"].unique()

        print(
            f"[{mode.upper()}] Loading data for fragments: {unique_fragments} at Z-start {z_start}..."
        )

        for fid in unique_fragments:
            # Get volume path from the first entry for this fragment
            row = self.df[self.df["fragment_id"] == fid].iloc[0]
            vol_path = row["volume_path"]
            mask_path = row["mask_path"]

            # Load Slab
            self.fragment_slabs[fid] = process_and_cache_slab(
                fid, vol_path, z_start, load_cached_data=load_cached_data
            )

            # Load Mask
            full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
            mask_img = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Binarize
                self.fragment_masks[fid] = (mask_img > 0).astype(np.uint8)
            else:
                # Should not happen based on metadata generation
                raise ValueError(f"Mask not found: {full_mask_path}")

            # Load Label (only for train/val)
            if mode in ["train", "val"] and "label_path" in row:
                label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if label_img is not None:
                    self.fragment_labels[fid] = (label_img > 0).astype(np.float32)
                else:
                    self.fragment_labels[fid] = np.zeros_like(
                        self.fragment_masks[fid], dtype=np.float32
                    )

        # Prepare samples list
        self.samples = []

        if mode == "test":
            # For test, metadata is fragment-level. We need to generate tiles.
            # We iterate over fragments and tile them.
            for fid in unique_fragments:
                h, w = self.fragment_masks[fid].shape
                # Generate tiles
                for y in range(0, h, Config.STRIDE):
                    for x in range(0, w, Config.STRIDE):
                        # Adjust for boundary
                        # We strictly use the tile size. If it goes out of bounds, we handle in __getitem__
                        # But standard practice is to pad or clip.
                        # Here we will just record the coordinate. __getitem__ will handle edge padding if needed.
                        self.samples.append(
                            {
                                "fragment_id": fid,
                                "x": x,
                                "y": y,
                                "width": Config.TILE_SIZE,
                                "height": Config.TILE_SIZE,
                            }
                        )
        else:
            # For train/val, metadata contains patch coordinates
            # Convert dataframe to list of dicts for faster access
            self.samples = self.df.to_dict("records")

        # Debugging limit
        if limit_size is not None:
            self.samples = self.samples[:limit_size]

        print(f"[{mode.upper()}] Dataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fid = sample["fragment_id"]
        x = sample["x"]
        y = sample["y"]
        w = sample["width"]
        h = sample["height"]

        # Retrieve full fragment data
        slab = self.fragment_slabs[fid]  # (H_full, W_full, 3)

        # Calculate crop coordinates with boundary handling
        img_h, img_w, _ = slab.shape

        y_end = min(y + h, img_h)
        x_end = min(x + w, img_w)

        # Crop image
        image_crop = slab[y:y_end, x:x_end, :]

        # Pad if necessary (at the right or bottom edges)
        pad_h = h - (y_end - y)
        pad_w = w - (x_end - x)

        if pad_h > 0 or pad_w > 0:
            image_crop = np.pad(
                image_crop,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # Prepare Label (if exists)
        label_crop = None
        if self.mode in ["train", "val"]:
            full_label = self.fragment_labels[fid]
            l_crop = full_label[y:y_end, x:x_end]
            if pad_h > 0 or pad_w > 0:
                l_crop = np.pad(
                    l_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
                )
            label_crop = l_crop

        # Apply Augmentations
        # Albumentations expects HWC image and HW mask
        if self.mode == "train":
            augmented = self.transform(image=image_crop, mask=label_crop)
            image_tensor = augmented["image"]
            label_tensor = augmented["mask"].unsqueeze(0)  # Add channel dim: (1, H, W)
        elif self.mode == "val":
            augmented = self.transform(image=image_crop, mask=label_crop)
            image_tensor = augmented["image"]
            label_tensor = augmented["mask"].unsqueeze(0)
        else:
            # Test mode
            augmented = self.transform(image=image_crop)
            image_tensor = augmented["image"]
            # Return coordinate info for reconstruction
            return image_tensor, torch.tensor([x, y]), fid

        return image_tensor, label_tensor
