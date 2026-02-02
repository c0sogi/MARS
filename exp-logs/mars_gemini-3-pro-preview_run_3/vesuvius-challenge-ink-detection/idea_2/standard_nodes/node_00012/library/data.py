import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from tqdm import tqdm
from library.config import Config


def load_fragment_data(
    fragment_id, volume_rel_path, mask_rel_path, split, cache_dir, load_cached=True
):
    """
    Loads volume and mask data for a fragment, using caching to speed up access.

    Args:
        fragment_id (str): Unique identifier for the fragment.
        volume_rel_path (str): Relative path to the surface volume directory.
        mask_rel_path (str): Relative path to the binary mask (or None for test).
        split (str): 'train', 'val', or 'test'.
        cache_dir (Path): Directory to store/load cached .npy files.
        load_cached (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (volume_array, mask_array)
               volume_array: (Z, H, W) float32
               mask_array: (H, W) uint8, or None if mask_rel_path is None
    """
    os.makedirs(cache_dir, exist_ok=True)

    vol_cache_path = cache_dir / f"{split}_{fragment_id}_volume.npy"
    mask_cache_path = cache_dir / f"{split}_{fragment_id}_mask.npy"

    # 1. Try to load from cache
    if load_cached and vol_cache_path.exists():
        # Check mask existence if required
        if mask_rel_path is None or mask_cache_path.exists():
            try:
                volume = np.load(
                    vol_cache_path, mmap_mode="r"
                )  # Use mmap to save RAM if needed, though we load to RAM later usually
                # We load fully into RAM for speed during training if RAM permits (220GB is plenty)
                volume = np.array(volume)

                mask = None
                if mask_rel_path is not None:
                    mask = np.load(mask_cache_path)

                return volume, mask
            except Exception as e:
                print(f"Failed to load cache for {fragment_id}: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load Volume
    full_vol_path = Config.INPUT_DIR / volume_rel_path
    if not full_vol_path.exists():
        raise FileNotFoundError(f"Volume path not found: {full_vol_path}")

    # Load all 65 slices
    slices = []
    for z in range(Config.Z_DIM):
        slice_path = full_vol_path / f"{z:02d}.tif"
        if not slice_path.exists():
            # Fallback or error? The dataset should have 65 slices.
            # Assuming standard structure, if missing, fill with zeros or raise error.
            raise FileNotFoundError(f"Slice missing: {slice_path}")

        img = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
        slices.append(img)

    volume = np.stack(slices, axis=0)  # (65, H, W)

    # Save Volume to Cache
    np.save(vol_cache_path, volume)

    # Load Mask
    mask = None
    if mask_rel_path is not None:
        full_mask_path = Config.INPUT_DIR / mask_rel_path
        if full_mask_path.exists():
            mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
            # Binarize
            mask = (mask > 0).astype(np.uint8)
            np.save(mask_cache_path, mask)
        else:
            # If mask path is provided but file missing (shouldn't happen based on metadata)
            pass

    return volume, mask


class InkDataset(Dataset):
    def __init__(self, fragment_row, split, config, load_cached=True):
        """
        Args:
            fragment_row (pd.Series): Row from metadata CSV.
            split (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
            load_cached (bool): Whether to use cached data.
        """
        self.split = split
        self.config = config
        self.fragment_id = str(fragment_row["fragment_id"])

        # Load Data
        # For training/val, we use 'inklabels' as the target mask if available.
        # The 'mask' column in metadata usually refers to the valid fragment area mask.
        # However, the task description says:
        # - train/[id]/inklabels.png --- binary mask of ink vs no-ink labels.
        # - train/[id]/mask.png --- binary mask of which pixels contain data.
        # We need inklabels for training targets.

        target_path = None
        if split in ["train", "val"]:
            target_path = fragment_row.get("inklabels_path", None)

        # We also need the valid pixel mask to avoid sampling background
        self.valid_mask_path = fragment_row.get("mask_path", None)

        # Load Volume and Target
        # Note: We cache the volume and the TARGET mask (inklabels).
        # We handle the 'valid_mask' separately or load it if needed for sampling logic.

        self.volume, self.labels = load_fragment_data(
            self.fragment_id,
            fragment_row["surface_volume_path"],
            target_path,
            split,
            config.CACHE_DIR,
            load_cached,
        )

        # Load valid mask for sampling constraint
        self.valid_mask = None
        if self.valid_mask_path:
            valid_mask_full_path = config.INPUT_DIR / self.valid_mask_path
            if valid_mask_full_path.exists():
                self.valid_mask = cv2.imread(
                    str(valid_mask_full_path), cv2.IMREAD_GRAYSCALE
                )
                self.valid_mask = (self.valid_mask > 0).astype(np.uint8)

        self.h, self.w = self.volume.shape[1], self.volume.shape[2]

        # Pre-calculate grid for validation/test
        if self.split != "train":
            self.coordinates = self._create_grid()
        else:
            # For training, we define length based on area coverage to approximate an epoch
            # Area / (Tile^2) * Overlap_Factor
            n_tiles = (self.h * self.w) / (config.TILE_SIZE**2)
            self.length = int(n_tiles * 2)  # e.g. 2x coverage per epoch

    def _create_grid(self):
        """Creates a list of (y, x) top-left coordinates for sliding window."""
        coords = []
        y_steps = range(0, self.h - self.config.TILE_SIZE + 1, self.config.STRIDE)
        x_steps = range(0, self.w - self.config.TILE_SIZE + 1, self.config.STRIDE)

        # Add the last step if it doesn't cover the edge
        if (self.h - self.config.TILE_SIZE) % self.config.STRIDE != 0:
            y_steps = list(y_steps) + [self.h - self.config.TILE_SIZE]
        if (self.w - self.config.TILE_SIZE) % self.config.STRIDE != 0:
            x_steps = list(x_steps) + [self.w - self.config.TILE_SIZE]

        for y in y_steps:
            for x in x_steps:
                coords.append((y, x))
        return coords

    def __len__(self):
        if self.split == "train":
            return self.length
        return len(self.coordinates)

    def __getitem__(self, idx):
        if self.split == "train":
            # Random sampling
            # Try to find a valid crop (contains some fragment data)
            for _ in range(10):
                y = np.random.randint(0, self.h - self.config.TILE_SIZE + 1)
                x = np.random.randint(0, self.w - self.config.TILE_SIZE + 1)

                # Check if this crop contains valid fragment area
                if self.valid_mask is not None:
                    crop_valid = self.valid_mask[
                        y : y + self.config.TILE_SIZE, x : x + self.config.TILE_SIZE
                    ]
                    if crop_valid.sum() > 0:  # At least some valid pixels
                        break
            # If we fail 10 times, just use the last coordinates
        else:
            # Deterministic sampling
            y, x = self.coordinates[idx]

        # Extract Crops
        # Volume: (65, H, W) -> Crop: (65, Tile, Tile)
        vol_crop = self.volume[
            :, y : y + self.config.TILE_SIZE, x : x + self.config.TILE_SIZE
        ].astype(np.float32)

        # Label: (H, W) -> Crop: (Tile, Tile)
        if self.labels is not None:
            label_crop = self.labels[
                y : y + self.config.TILE_SIZE, x : x + self.config.TILE_SIZE
            ].astype(np.float32)
        else:
            # For test set, return dummy mask
            label_crop = np.zeros(
                (self.config.TILE_SIZE, self.config.TILE_SIZE), dtype=np.float32
            )

        # Normalization
        vol_crop = (vol_crop - self.config.NORM_MEAN) / self.config.NORM_STD

        # Augmentation (Train only)
        if self.split == "train":
            # Random Flip
            if np.random.rand() < 0.5:
                vol_crop = np.flip(vol_crop, axis=2)  # Flip W
                label_crop = np.flip(label_crop, axis=1)
            if np.random.rand() < 0.5:
                vol_crop = np.flip(vol_crop, axis=1)  # Flip H
                label_crop = np.flip(label_crop, axis=0)

            # Random Rotate 90
            k = np.random.randint(0, 4)
            if k > 0:
                vol_crop = np.rot90(vol_crop, k, axes=(1, 2))
                label_crop = np.rot90(label_crop, k, axes=(0, 1))

        # Convert to Tensor
        # Volume: (C, H, W) is standard for PyTorch. Our data is already (Z, H, W).
        vol_tensor = torch.from_numpy(vol_crop.copy())
        label_tensor = torch.from_numpy(label_crop.copy()).unsqueeze(0)  # (1, H, W)

        if self.split == "train":
            return vol_tensor, label_tensor
        else:
            # Return coordinates for reconstruction
            return vol_tensor, label_tensor, torch.tensor([y, x])


def get_dataloaders(config):
    """
    Creates DataLoaders for train, val, and test splits.
    """
    dataloaders = {}

    # Load Metadata
    if config.TRAIN_METADATA_PATH.exists():
        df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
        train_datasets = [
            InkDataset(row, "train", config) for _, row in df_train.iterrows()
        ]
        if train_datasets:
            dataloaders["train"] = DataLoader(
                ConcatDataset(train_datasets),
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )

    if config.VAL_METADATA_PATH.exists():
        df_val = pd.read_csv(config.VAL_METADATA_PATH)
        val_datasets = [InkDataset(row, "val", config) for _, row in df_val.iterrows()]
        if val_datasets:
            dataloaders["val"] = DataLoader(
                ConcatDataset(val_datasets),
                batch_size=config.BATCH_SIZE,
                shuffle=False,  # Deterministic order for validation
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )

    if config.TEST_METADATA_PATH.exists():
        df_test = pd.read_csv(config.TEST_METADATA_PATH)
        test_datasets = [
            InkDataset(row, "test", config) for _, row in df_test.iterrows()
        ]
        if test_datasets:
            # We usually process test fragments one by one for reconstruction,
            # but a single dataloader works if we track fragment IDs.
            # However, for simplicity and standard pipelines, we often return a dict of loaders or a single loader.
            # Given the prompt asks for "predictions for the entire test set", a single loader is fine
            # as long as we can reconstruct. But reconstruction requires knowing which fragment a tile belongs to.
            # InkDataset doesn't return fragment ID in __getitem__.
            # To handle this properly for submission, we should probably return a list of loaders or handle it in the inference loop.
            # Standard approach: Return a single loader, but since we only have 2 test fragments,
            # let's return a dictionary mapping fragment_id -> loader for the test set to make inference easier.

            test_loaders = {}
            for _, row in df_test.iterrows():
                fid = str(row["fragment_id"])
                ds = InkDataset(row, "test", config)
                test_loaders[fid] = DataLoader(
                    ds,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS,
                    pin_memory=True,
                )
            dataloaders["test_dict"] = test_loaders

    return dataloaders
