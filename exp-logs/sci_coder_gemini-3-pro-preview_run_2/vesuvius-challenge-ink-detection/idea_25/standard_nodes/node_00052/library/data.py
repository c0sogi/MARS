import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from library.config import Config

# Ensure reproducibility
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


def get_transforms(mode="train"):
    """
    Returns the Albumentations transform pipeline for the specified mode.
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


def load_fragment_slab(fragment_id, volume_dir, z_min, z_max, load_cached_data=True):
    """
    Loads a specific Z-range slab of the fragment volume.
    Caches the result as a .npy file to speed up subsequent loads.

    Args:
        fragment_id (str): ID of the fragment.
        volume_dir (str): Relative path to the volume directory.
        z_min (int): Start slice index (inclusive).
        z_max (int): End slice index (exclusive).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The volume slab of shape (D, H, W) normalized to [0, 1].
    """
    cache_filename = f"frag_{fragment_id}_slab_{z_min}_{z_max}.npy"
    cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(
                f"Loading cached volume for fragment {fragment_id} from {cache_path}..."
            )
            return np.load(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source TIFFs
    print(f"Processing volume for fragment {fragment_id} (Slices {z_min}-{z_max})...")
    slices = []
    for z in range(z_min, z_max):
        filename = f"{z:02d}.tif"
        path = os.path.join(Config.INPUT_DIR, volume_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Slice file not found: {path}")

        # Load image (uint16)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Normalize to [0, 1] float32
        img = img.astype(np.float32) / 65535.0
        slices.append(img)

    # Stack along depth: (D, H, W)
    volume_slab = np.stack(slices, axis=0)

    # 3. Save to cache
    try:
        np.save(cache_path, volume_slab)
        print(f"Cached volume slab to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")

    return volume_slab


class InkDataset(Dataset):
    def __init__(self, mode="train", load_cached_data=True, limit_size=None):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Use cached .npy files if available.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.transforms = get_transforms(mode)

        # Determine the global Z-range required for all modes
        # We need to cover the training range and all inference start points + slab depth
        # Train Range: [TRAIN_Z_MIN, TRAIN_Z_MAX] -> needs up to + SLAB_DEPTH
        # Inference: INFERENCE_Z_STARTS -> needs up to + SLAB_DEPTH

        all_starts = Config.INFERENCE_Z_STARTS + [
            Config.TRAIN_Z_MIN,
            Config.TRAIN_Z_MAX,
        ]
        self.z_min_load = min(all_starts)
        self.z_max_load = max(all_starts) + Config.SLAB_DEPTH

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif mode == "validation":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
        elif mode == "test":
            self.df_frags = pd.read_csv(Config.TEST_METADATA_PATH)
            self.df = self._expand_test_tiles()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if limit_size:
            self.df = self.df.iloc[:limit_size]

        # Pre-load volumes for referenced fragments
        self.volumes = {}
        unique_frags = self.df["fragment_id"].unique()

        for fid in unique_frags:
            # Get volume path
            if mode == "test":
                row = self.df_frags[self.df_frags["fragment_id"] == fid].iloc[0]
            else:
                row = self.df[self.df["fragment_id"] == fid].iloc[0]

            self.volumes[fid] = load_fragment_slab(
                fid,
                row["volume_path"],
                self.z_min_load,
                self.z_max_load,
                load_cached_data,
            )

    def _expand_test_tiles(self):
        """
        Generates a dataframe of tiles for the test fragments.
        """
        tiles = []
        for _, row in self.df_frags.iterrows():
            fid = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Read mask to get dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape

            # Generate non-overlapping tiles (stride = size)
            for y in range(0, h, Config.TILE_SIZE):
                for x in range(0, w, Config.TILE_SIZE):
                    tiles.append(
                        {
                            "fragment_id": fid,
                            "x": x,
                            "y": y,
                            "width": Config.TILE_SIZE,
                            "height": Config.TILE_SIZE,
                            "mask_path": row["mask_path"],
                            "label_path": None,  # No labels in test
                        }
                    )
        return pd.DataFrame(tiles)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = row["fragment_id"]
        x, y = row["x"], row["y"]
        w, h = row["width"], row["height"]

        # 1. Extract Crop from Cached Volume
        vol_slab = self.volumes[fid]  # (D_total, H_frag, W_frag)
        _, vol_h, vol_w = vol_slab.shape

        # Calculate padding if tile goes out of bounds
        pad_h = max(0, (y + h) - vol_h)
        pad_w = max(0, (x + w) - vol_w)

        crop_h = h - pad_h
        crop_w = w - pad_w

        # Crop volume
        crop_vol = vol_slab[:, y : y + crop_h, x : x + crop_w]

        # Pad volume if necessary
        if pad_h > 0 or pad_w > 0:
            crop_vol = np.pad(
                crop_vol,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # 2. Determine Z-Start(s)
        if self.mode == "train":
            # Dynamic sampling for translation invariance
            z_start = random.randint(Config.TRAIN_Z_MIN, Config.TRAIN_Z_MAX)
            z_starts = [z_start]
        elif self.mode == "validation":
            # Fixed center for consistent validation
            z_start = (Config.TRAIN_Z_MIN + Config.TRAIN_Z_MAX) // 2
            z_starts = [z_start]
        else:  # Test
            # Multiple deterministic views for Max-Fusion
            z_starts = Config.INFERENCE_Z_STARTS

        # 3. Process each Z-view
        processed_views = []

        for z_s in z_starts:
            rel_z = z_s - self.z_min_load

            # Extract 12-slice slab
            slab = crop_vol[rel_z : rel_z + Config.SLAB_DEPTH, :, :]

            # Compress to 3 channels via MIP on 4-slice chunks
            # Shape: (12, H, W) -> (3, H, W)
            ch1 = np.max(slab[0:4], axis=0)
            ch2 = np.max(slab[4:8], axis=0)
            ch3 = np.max(slab[8:12], axis=0)

            img_3ch = np.stack([ch1, ch2, ch3], axis=-1)  # (H, W, 3)
            processed_views.append(img_3ch)

        # 4. Handle Labels (Train/Val only)
        mask_tensor = None
        if self.mode in ["train", "validation"]:
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

            # Crop and Pad Label
            crop_lbl = label_img[y : y + crop_h, x : x + crop_w]
            if pad_h > 0 or pad_w > 0:
                crop_lbl = np.pad(
                    crop_lbl,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            # Binarize
            mask_np = (crop_lbl > 0).astype(np.float32)

            # Apply Augmentations (Geometric)
            # We only have 1 view for train/val
            augmented = self.transforms(image=processed_views[0], mask=mask_np)
            img_tensor = augmented["image"]
            mask_tensor = augmented["mask"].unsqueeze(0)  # (1, H, W)

            return img_tensor, mask_tensor

        else:
            # Test Mode: Return stack of views and metadata
            view_tensors = []
            for img in processed_views:
                res = self.transforms(image=img)
                view_tensors.append(res["image"])

            # Stack: (Num_Views, 3, H, W)
            stack = torch.stack(view_tensors, dim=0)

            meta = {
                "fragment_id": fid,
                "x": x,
                "y": y,
                "h": h,  # Original tile size
                "w": w,
                "orig_h": vol_h,  # Full fragment dims for reconstruction
                "orig_w": vol_w,
            }
            return stack, meta
