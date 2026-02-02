import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_volume, load_mask, load_inklabels


def get_normalization_stats(load_cached_data=True):
    """
    Computes or loads global mean and standard deviation from the training set.
    Samples pixels from valid mask areas to estimate global statistics.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    stats_path = Config.WORKING_DIR / "normalization_stats.npy"

    if load_cached_data and stats_path.exists():
        try:
            stats = np.load(stats_path)
            return float(stats[0]), float(stats[1])
        except Exception as e:
            print(f"Failed to load cached stats: {e}. Recomputing...")

    print("Computing global normalization statistics from training data...")

    # Always use training metadata for stats calculation
    if not Config.TRAIN_METADATA_PATH.exists():
        raise FileNotFoundError("Training metadata not found for stats calculation.")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)

    sampled_pixels = []
    # Target roughly 10 million pixels total for robust estimation
    target_sample_count = 10_000_000
    if len(df_train) > 0:
        pixels_per_fragment = target_sample_count // len(df_train)
    else:
        pixels_per_fragment = target_sample_count

    for _, row in df_train.iterrows():
        frag_id = row["fragment_id"]

        # Load data (utilizing utils caching)
        try:
            vol = load_volume(
                frag_id, "train", df_train, load_cached_data=load_cached_data
            )
            mask = load_mask(
                frag_id, "train", df_train, load_cached_data=load_cached_data
            )
        except Exception as e:
            print(f"Skipping fragment {frag_id} for stats due to load error: {e}")
            continue

        # Identify valid pixels
        valid_y, valid_x = np.where(mask > 0)
        n_valid = len(valid_y)

        if n_valid == 0:
            continue

        # Randomly sample indices
        if n_valid > pixels_per_fragment:
            indices = np.random.choice(n_valid, pixels_per_fragment, replace=False)
            sy = valid_y[indices]
            sx = valid_x[indices]
        else:
            sy = valid_y
            sx = valid_x

        # Extract pixel values across all Z-slices for these positions
        # vol is (Z, H, W). Result is (Z, N_samples) flattened to (Z*N_samples,)
        vals = vol[:, sy, sx].flatten()
        sampled_pixels.append(vals)

    if not sampled_pixels:
        print("Warning: No valid pixels sampled. Defaulting to mean=0, std=1.")
        mean, std = 0.0, 1.0
    else:
        all_pixels = np.concatenate(sampled_pixels)
        mean = float(np.mean(all_pixels))
        std = float(np.std(all_pixels))

    # Save to cache
    np.save(stats_path, np.array([mean, std]))
    print(f"Computed stats - Mean: {mean:.4f}, Std: {std:.4f}")

    return mean, std


class InkDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        self.split = split
        self.patch_size = Config.PATCH_SIZE

        # Determine Metadata Path
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA_PATH
        elif split == "test":
            self.metadata_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        if not self.metadata_path.exists():
            # Handle case where split might be empty (e.g. no val set)
            self.df = pd.DataFrame()
        else:
            self.df = pd.read_csv(self.metadata_path)

        # Load Normalization Stats (always from training distribution)
        self.mean, self.std = get_normalization_stats(load_cached_data=load_cached_data)

        # Load Data into Memory
        self.fragments = []
        if not self.df.empty:
            for _, row in self.df.iterrows():
                frag_id = row["fragment_id"]

                # Load full volume and mask
                vol = load_volume(
                    frag_id, split, self.df, load_cached_data=load_cached_data
                )
                mask = load_mask(
                    frag_id, split, self.df, load_cached_data=load_cached_data
                )
                label = load_inklabels(
                    frag_id, split, self.df, load_cached_data=load_cached_data
                )

                self.fragments.append(
                    {
                        "id": frag_id,
                        "volume": vol,  # (Z, H, W)
                        "mask": mask,  # (H, W)
                        "label": label,  # (H, W) or None
                    }
                )

        # Setup Indexing Strategy
        if self.split == "train":
            self.length = Config.TRAIN_SAMPLES_PER_EPOCH
        else:
            # Create deterministic grid for Val/Test
            self.grid = []  # List of (fragment_idx, y, x)
            tile_size = Config.VAL_TILE_SIZE
            stride = Config.VAL_STRIDE

            for f_idx, frag in enumerate(self.fragments):
                _, h, w = frag["volume"].shape

                # Generate top-left coordinates
                ys = list(range(0, h - tile_size + 1, stride))
                # Ensure we cover the last edge if it doesn't align perfectly
                if ys[-1] != h - tile_size:
                    ys.append(h - tile_size)

                xs = list(range(0, w - tile_size + 1, stride))
                if xs[-1] != w - tile_size:
                    xs.append(w - tile_size)

                for y in ys:
                    for x in xs:
                        # Optimization: Only add tiles that contain valid mask
                        # This saves inference time on empty background
                        mask_patch = frag["mask"][y : y + tile_size, x : x + tile_size]
                        if np.any(mask_patch):
                            self.grid.append((f_idx, y, x))

            self.length = len(self.grid)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.split == "train":
            return self._get_train_item(idx)
        else:
            return self._get_val_test_item(idx)

    def _get_train_item(self, idx):
        # 1. Select Random Fragment
        f_idx = random.randint(0, len(self.fragments) - 1)
        frag = self.fragments[f_idx]

        vol = frag["volume"]
        mask = frag["mask"]
        label = frag["label"]

        _, h, w = vol.shape

        # 2. Rejection Sampling for Valid Crop
        # Try to find a crop that contains at least some valid fragment area
        for _ in range(20):
            y = random.randint(0, h - self.patch_size)
            x = random.randint(0, w - self.patch_size)

            mask_patch = mask[y : y + self.patch_size, x : x + self.patch_size]
            # Check if at least 5% of the patch is valid fragment
            if np.mean(mask_patch) > 0.05:
                break

        # 3. Extract Data
        vol_patch = vol[:, y : y + self.patch_size, x : x + self.patch_size]
        label_patch = label[y : y + self.patch_size, x : x + self.patch_size]

        # 4. Augmentations
        # Random Rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            vol_patch = np.rot90(vol_patch, k, axes=(1, 2))
            label_patch = np.rot90(label_patch, k, axes=(0, 1))

        # Random Horizontal Flip
        if random.random() < 0.5:
            vol_patch = np.flip(vol_patch, axis=2)
            label_patch = np.flip(label_patch, axis=1)

        # Random Vertical Flip
        if random.random() < 0.5:
            vol_patch = np.flip(vol_patch, axis=1)
            label_patch = np.flip(label_patch, axis=0)

        # 5. Normalization
        vol_patch = (vol_patch.astype(np.float32) - self.mean) / self.std

        # 6. Convert to Tensor
        # Volume: (Z, H, W)
        # Label: (H, W) -> Float for BCE Loss
        return (
            torch.from_numpy(vol_patch.copy()),
            torch.from_numpy(label_patch.copy()).float(),
        )

    def _get_val_test_item(self, idx):
        f_idx, y, x = self.grid[idx]
        frag = self.fragments[f_idx]

        vol = frag["volume"]
        tile_size = Config.VAL_TILE_SIZE

        # Extract
        vol_patch = vol[:, y : y + tile_size, x : x + tile_size]

        # Normalize
        vol_patch = (vol_patch.astype(np.float32) - self.mean) / self.std

        # Construct output dictionary
        item = {
            "volume": torch.from_numpy(vol_patch.copy()),
            "fragment_id": str(frag["id"]),
            "y": torch.tensor(y, dtype=torch.long),
            "x": torch.tensor(x, dtype=torch.long),
        }

        # Add label if available (for validation)
        if frag["label"] is not None:
            label_patch = frag["label"][y : y + tile_size, x : x + tile_size]
            item["label"] = torch.from_numpy(label_patch.copy()).float()

        return item
