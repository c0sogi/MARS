import os
import cv2
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from pathlib import Path
from library.config import Config
from library.utils import load_normalization_stats, get_boundary_mask


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)


def load_fragment_data(
    fragment_id: str,
    split: str,
    surface_volume_path: str,
    mask_path: str,
    inklabels_path: str = None,
    load_cached_data: bool = True,
):
    """
    Loads fragment data (volume, mask, label) with caching.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    vol_cache_path = cache_dir / f"{fragment_id}_volume.npy"
    mask_cache_path = cache_dir / f"{fragment_id}_mask.npy"
    label_cache_path = cache_dir / f"{fragment_id}_label.npy"

    # 1. Try loading from cache
    if load_cached_data and vol_cache_path.exists() and mask_cache_path.exists():
        try:
            # Use mmap_mode='r' to avoid immediate RAM spike, though we will likely read it all
            volume = np.load(vol_cache_path)
            mask = np.load(mask_cache_path)
            label = None

            # Check if label is expected and exists
            if inklabels_path:
                if label_cache_path.exists():
                    label = np.load(label_cache_path)
                else:
                    # Label expected but missing in cache -> trigger recompute
                    raise FileNotFoundError("Label cache missing")

            return volume, mask, label
        except Exception as e:
            print(f"Cache load failed for fragment {fragment_id}: {e}. Recomputing...")

    # 2. Compute from scratch
    # Load Mask
    full_mask_path = Config.INPUT_DIR / mask_path
    if not full_mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {full_mask_path}")
    mask = cv2.imread(str(full_mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)

    # Load Label (if exists)
    label = None
    if inklabels_path:
        full_label_path = Config.INPUT_DIR / inklabels_path
        if full_label_path.exists():
            label_img = cv2.imread(str(full_label_path), cv2.IMREAD_GRAYSCALE)
            label = (label_img > 0).astype(np.uint8)

    # Load Volume
    h, w = mask.shape
    volume = np.zeros((Config.Z_DIM, h, w), dtype=np.uint8)

    vol_dir = Config.INPUT_DIR / surface_volume_path
    if not vol_dir.exists():
        raise FileNotFoundError(f"Volume directory not found: {vol_dir}")

    # Load slices
    for z in range(Config.Z_DIM):
        slice_path = vol_dir / f"{z:02d}.tif"
        if slice_path.exists():
            img_slice = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
            if img_slice is not None:
                volume[z] = img_slice

    # 3. Save to cache
    np.save(vol_cache_path, volume)
    np.save(mask_cache_path, mask)
    if label is not None:
        np.save(label_cache_path, label)

    return volume, mask, label


class InkDataset(Dataset):
    """
    Dataset for Training and Validation (Patch-based).
    Implements random sampling from valid mask areas.
    """

    def __init__(
        self,
        split: str = "train",
        samples_per_epoch: int = None,
        load_cached_data: bool = True,
    ):
        self.split = split
        self.samples_per_epoch = (
            samples_per_epoch
            if samples_per_epoch is not None
            else Config.SAMPLES_PER_EPOCH
        )
        self.patch_size = Config.PATCH_SIZE
        self.pad = self.patch_size // 2

        # Load Metadata
        meta_file = Config.METADATA_DIR / f"{split}.csv"
        if not meta_file.exists():
            # Fallback for validation if split file doesn't exist (e.g. single fragment training)
            if split == "val":
                print(
                    f"Warning: {split}.csv not found. Using train.csv subset logic if implemented or failing."
                )
            raise FileNotFoundError(f"Metadata file {meta_file} not found.")

        self.df = pd.read_csv(meta_file)

        # Load Normalization Stats
        train_meta = Config.METADATA_DIR / "train.csv"
        self.mean, self.std = load_normalization_stats(
            train_meta, load_cached_data=load_cached_data
        )

        self.fragments = []
        self.valid_indices = []

        for idx, row in self.df.iterrows():
            fid = str(row["fragment_id"])
            ink_path = (
                row["inklabels_path"]
                if "inklabels_path" in row and pd.notna(row["inklabels_path"])
                else None
            )

            vol, mask, label = load_fragment_data(
                fid,
                split,
                row["surface_volume_path"],
                row["mask_path"],
                ink_path,
                load_cached_data,
            )

            # Pad arrays
            vol_padded = np.pad(
                vol,
                ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode="constant",
                constant_values=0,
            )
            mask_padded = np.pad(
                mask,
                ((self.pad, self.pad), (self.pad, self.pad)),
                mode="constant",
                constant_values=0,
            )

            label_padded = None
            if label is not None:
                label_padded = np.pad(
                    label,
                    ((self.pad, self.pad), (self.pad, self.pad)),
                    mode="constant",
                    constant_values=0,
                )

            # Get valid sampling centers (relative to original image)
            ys, xs = np.where(mask > 0)

            self.fragments.append(
                {"volume": vol_padded, "label": label_padded, "mask": mask_padded}
            )

            # Store indices as (fragment_idx, center_y_padded, center_x_padded)
            # center_y_padded = y_original + pad
            frag_indices = np.stack(
                [
                    np.full_like(ys, len(self.fragments) - 1),
                    ys + self.pad,
                    xs + self.pad,
                ],
                axis=1,
            )
            self.valid_indices.append(frag_indices)

        self.valid_indices = np.concatenate(self.valid_indices, axis=0)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Random sampling
        rand_idx = np.random.randint(len(self.valid_indices))
        frag_idx, cy, cx = self.valid_indices[rand_idx]

        data = self.fragments[frag_idx]

        y1, y2 = cy - self.pad, cy + self.pad
        x1, x2 = cx - self.pad, cx + self.pad

        # Extract crops
        vol_crop = data["volume"][:, y1:y2, x1:x2].astype(np.float32)
        label_crop = data["label"][y1:y2, x1:x2].astype(np.float32)

        # Normalize
        vol_crop = (vol_crop - self.mean) / (self.std + 1e-6)

        # Augmentation (Train only)
        if self.split == "train":
            # Flip
            if random.random() > 0.5:
                vol_crop = np.flip(vol_crop, axis=2)  # W axis
                label_crop = np.flip(label_crop, axis=1)
            if random.random() > 0.5:
                vol_crop = np.flip(vol_crop, axis=1)  # H axis
                label_crop = np.flip(label_crop, axis=0)

            # Rotate
            k = random.randint(0, 3)
            if k > 0:
                vol_crop = np.rot90(vol_crop, k, axes=(1, 2))
                label_crop = np.rot90(label_crop, k, axes=(0, 1))

        # Generate Boundary
        boundary_crop = get_boundary_mask(label_crop)

        # To Tensor
        vol_tensor = torch.from_numpy(vol_crop.copy())
        label_tensor = torch.from_numpy(label_crop.copy()).unsqueeze(0)
        boundary_tensor = torch.from_numpy(boundary_crop.copy()).unsqueeze(0)

        return vol_tensor, {"mask": label_tensor, "boundary": boundary_tensor}


class InferenceDataset(Dataset):
    """
    Dataset for Sliding Window Inference.
    """

    def __init__(
        self,
        split: str = "test",
        fragment_id: str = None,
        load_cached_data: bool = True,
    ):
        self.split = split
        self.patch_size = Config.PATCH_SIZE
        self.stride = Config.INFERENCE_STRIDE
        self.pad = self.patch_size // 2

        meta_file = Config.METADATA_DIR / f"{split}.csv"
        self.df = pd.read_csv(meta_file)

        if fragment_id:
            self.df = self.df[self.df["fragment_id"] == fragment_id]

        train_meta = Config.METADATA_DIR / "train.csv"
        self.mean, self.std = load_normalization_stats(
            train_meta, load_cached_data=load_cached_data
        )

        self.tiles = []

        for _, row in self.df.iterrows():
            fid = str(row["fragment_id"])
            vol, mask, _ = load_fragment_data(
                fid,
                split,
                row["surface_volume_path"],
                row["mask_path"],
                None,
                load_cached_data,
            )

            h, w = mask.shape
            vol_padded = np.pad(
                vol,
                ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode="constant",
                constant_values=0,
            )

            # Sliding window top-left coordinates
            for y in range(0, h, self.stride):
                for x in range(0, w, self.stride):
                    self.tiles.append(
                        {
                            "fragment_id": fid,
                            "volume": vol_padded,
                            "y": y,
                            "x": x,
                            "h": h,
                            "w": w,
                        }
                    )

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        y, x = tile["y"], tile["x"]

        # Center in padded image is y+pad, x+pad
        # Crop from (y+pad)-pad to (y+pad)+pad -> y to y+size

        vol_crop = tile["volume"][
            :, y : y + self.patch_size, x : x + self.patch_size
        ].astype(np.float32)
        vol_crop = (vol_crop - self.mean) / (self.std + 1e-6)

        return torch.from_numpy(vol_crop), {
            "fragment_id": tile["fragment_id"],
            "y": y,
            "x": x,
            "h": tile["h"],
            "w": tile["w"],
        }
