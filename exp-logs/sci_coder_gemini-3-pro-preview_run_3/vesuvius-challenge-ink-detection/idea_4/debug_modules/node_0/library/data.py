import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from library.config import Config
from library.utils import seed_everything


def load_image(path, grayscale=True):
    """
    Helper to load an image from a path.
    """
    path = str(path)
    if not os.path.exists(path):
        return None
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flags)


def _get_cached_path(fragment_id, suffix):
    """
    Generates the path for the cached .npy file.
    """
    return Config.CACHE_DIR / f"{fragment_id}_{suffix}.npy"


def load_and_process_volume(fragment_id, volume_dir, load_cached=True):
    """
    Loads the 3D surface volume.
    1. Checks for cached .npy file.
    2. If not found or forced, loads TIF slices, normalizes, and caches.
    """
    cache_path = _get_cached_path(fragment_id, "volume")

    if load_cached and cache_path.exists():
        try:
            # Load memory-mapped to save initial load time, though we likely read it all
            # Using mmap_mode='r' allows lazy loading if needed, but we usually need it in RAM for training
            # Given 220GB RAM, we can load fully.
            volume = np.load(cache_path)
            # print(f"Loaded cached volume for fragment {fragment_id}")
            return volume
        except Exception as e:
            print(f"Failed to load cached volume {cache_path}: {e}")

    # Process from scratch
    # print(f"Processing volume for fragment {fragment_id} from {volume_dir}...")

    # Construct paths for 65 slices
    # Assuming filenames are 00.tif, 01.tif, ... 64.tif based on description
    slices = []
    full_vol_dir = Config.INPUT_DIR / volume_dir

    for z in range(Config.Z_DIM):
        slice_path = full_vol_dir / f"{z:02d}.tif"
        if not slice_path.exists():
            # Fallback or error? Description says 65 slices.
            # If missing, we might pad, but usually data is complete.
            # We'll assume zeros if missing to be safe.
            # We need dimensions. Read 00.tif to get shape.
            pass

        img = load_image(slice_path)
        if img is None:
            # If 00.tif is missing, we have a problem.
            # Assuming data integrity based on task description.
            raise FileNotFoundError(f"Slice {slice_path} not found.")
        slices.append(img)

    # Stack: (D, H, W) -> (65, H, W)
    volume = np.stack(slices, axis=0)

    # Normalize
    # (x - mean) / std
    volume = (volume.astype(np.float32) - Config.PIXEL_MEAN) / Config.PIXEL_STD

    # Save to cache
    np.save(cache_path, volume)
    # print(f"Cached volume for fragment {fragment_id} to {cache_path}")

    return volume


def load_and_process_2d(fragment_id, image_path, suffix, load_cached=True):
    """
    Loads a 2D image (mask or label), binarizes it, and caches it.
    """
    if image_path is None:
        return None

    cache_path = _get_cached_path(fragment_id, suffix)

    if load_cached and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cached {suffix} {cache_path}: {e}")

    # Process
    full_path = Config.INPUT_DIR / image_path
    img = load_image(full_path)

    if img is None:
        return None

    # Binarize
    img = (img > 0).astype(np.uint8)

    # Save
    np.save(cache_path, img)

    return img


class InkDataset(Dataset):
    def __init__(self, split, load_cached=True, limit_size=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached (bool): Whether to use cached .npy files.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.patch_size = Config.PATCH_SIZE
        self.z_dim = Config.Z_DIM

        # Load Metadata
        meta_path = Config.METADATA_DIR / f"{split}.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file {meta_path} not found.")

        self.df = pd.read_csv(meta_path)

        # Load Data into Memory
        self.fragments = []

        for _, row in self.df.iterrows():
            fid = str(row["fragment_id"])

            # Load Volume
            vol = load_and_process_volume(fid, row["surface_volume_path"], load_cached)

            # Load Mask (Valid Area)
            mask = load_and_process_2d(fid, row["mask_path"], "mask", load_cached)

            # Load Label (Ink) - Only for train/val
            label = None
            if split in ["train", "val"]:
                label = load_and_process_2d(
                    fid, row["inklabels_path"], "label", load_cached
                )

            self.fragments.append(
                {
                    "id": fid,
                    "volume": vol,  # (65, H, W) float32
                    "mask": mask,  # (H, W) uint8
                    "label": label,  # (H, W) uint8 or None
                }
            )

        # Setup Sampling
        if split == "train":
            # For training, we sample randomly.
            # We define length arbitrarily or based on limit_size.
            # A large number ensures the epoch is long enough to cover data.
            # Given the huge size of fragments, 10,000 samples per epoch is reasonable
            # to get good gradient updates, or we can set it to limit_size.
            self.length = limit_size if limit_size else 2000
        else:
            # For val/test, we use a deterministic grid
            self.grid = self._generate_grid()
            if limit_size:
                self.grid = self.grid[:limit_size]
            self.length = len(self.grid)

    def _generate_grid(self):
        """
        Generates a list of (fragment_index, y, x) coordinates for sliding window.
        Only includes tiles where the mask indicates valid data.
        """
        grid = []
        stride = Config.INFERENCE_STRIDE

        for i, frag in enumerate(self.fragments):
            mask = frag["mask"]
            h, w = mask.shape

            # Generate coordinates
            # Ensure we cover edges
            y_steps = list(range(0, h - self.patch_size, stride))
            if (h - self.patch_size) % stride != 0:
                y_steps.append(h - self.patch_size)

            x_steps = list(range(0, w - self.patch_size, stride))
            if (w - self.patch_size) % stride != 0:
                x_steps.append(w - self.patch_size)

            for y in y_steps:
                for x in x_steps:
                    # Check if this patch contains any valid mask pixels
                    # We only evaluate on valid areas
                    mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]
                    if np.any(mask_patch):
                        grid.append((i, y, x))

        return grid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_grid_item(idx)

    def _get_train_item(self, idx):
        # Random sampling with retry to find valid mask area
        for _ in range(10):  # Try 10 times to find a valid patch
            # Pick random fragment
            frag_idx = np.random.randint(len(self.fragments))
            frag = self.fragments[frag_idx]

            h, w = frag["mask"].shape

            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)

            mask_patch = frag["mask"][y : y + self.patch_size, x : x + self.patch_size]

            # If patch is mostly empty space, skip
            if np.mean(mask_patch) < 0.1:  # Require at least 10% valid pixels
                continue

            # Crop
            vol_patch = frag["volume"][
                :, y : y + self.patch_size, x : x + self.patch_size
            ]
            label_patch = frag["label"][
                y : y + self.patch_size, x : x + self.patch_size
            ]

            # Augmentations
            # 1. Random Flip
            if np.random.rand() > 0.5:
                # Horizontal
                vol_patch = np.flip(vol_patch, axis=2)
                label_patch = np.flip(label_patch, axis=1)
            if np.random.rand() > 0.5:
                # Vertical
                vol_patch = np.flip(vol_patch, axis=1)
                label_patch = np.flip(label_patch, axis=0)

            # 2. Random Rotate
            k = np.random.randint(0, 4)
            if k > 0:
                # rot90 rotates in the plane of the last two axes
                vol_patch = np.rot90(vol_patch, k, axes=(1, 2))
                label_patch = np.rot90(label_patch, k, axes=(0, 1))

            # Convert to tensor
            # Vol: (D, H, W) -> Tensor
            # Label: (H, W) -> (1, H, W) Tensor
            return (
                torch.from_numpy(vol_patch.copy()),
                torch.from_numpy(label_patch.copy()).unsqueeze(0).float(),
            )

        # Fallback if retry fails (unlikely) -> return center crop of first frag
        frag = self.fragments[0]
        h, w = frag["mask"].shape
        y, x = h // 2, w // 2
        vol_patch = frag["volume"][:, y : y + self.patch_size, x : x + self.patch_size]
        label_patch = frag["label"][y : y + self.patch_size, x : x + self.patch_size]
        return (
            torch.from_numpy(vol_patch.copy()),
            torch.from_numpy(label_patch.copy()).unsqueeze(0).float(),
        )

    def _get_grid_item(self, idx):
        frag_idx, y, x = self.grid[idx]
        frag = self.fragments[frag_idx]

        vol_patch = frag["volume"][:, y : y + self.patch_size, x : x + self.patch_size]

        # Prepare label if available, else zeros
        if frag["label"] is not None:
            label_patch = frag["label"][
                y : y + self.patch_size, x : x + self.patch_size
            ]
            label_tensor = torch.from_numpy(label_patch.copy()).unsqueeze(0).float()
        else:
            label_tensor = torch.zeros(
                (1, self.patch_size, self.patch_size), dtype=torch.float32
            )

        # Return coordinates for reconstruction if needed
        # But standard DataLoader expects tensors.
        # We'll return vol, label. The loop can use the sequential order if needed,
        # or we can return metadata as a dict if collate_fn handles it.
        # For simplicity and standard loops: return vol, label.

        return (torch.from_numpy(vol_patch.copy()), label_tensor)


def get_dataloaders(load_cached=True, debug=False):
    """
    Factory function to create training and validation dataloaders.
    """
    limit = Config.DEBUG_SAMPLE_SIZE if debug else None

    # Train Set
    train_ds = InkDataset("train", load_cached=load_cached, limit_size=limit)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set
    # If debug, we might want a smaller val set too
    val_ds = InkDataset("val", load_cached=load_cached, limit_size=limit)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Deterministic order for grid
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(load_cached=True):
    """
    Factory function for the test set loader.
    """
    test_ds = InkDataset("test", load_cached=load_cached)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return test_loader, test_ds
