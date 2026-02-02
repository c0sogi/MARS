import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def get_training_transforms():
    """
    Returns a callable that applies random geometric augmentations
    (flips and rotations) to the volume and label patches.
    """

    def transform(volume, label):
        # volume: (D, H, W), label: (H, W)

        # Random Horizontal Flip
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=2)
            label = np.flip(label, axis=1)

        # Random Vertical Flip
        if np.random.rand() < 0.5:
            volume = np.flip(volume, axis=1)
            label = np.flip(label, axis=0)

        # Random 90-degree Rotation
        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))

        return volume, label

    return transform


class InkDataset(Dataset):
    def __init__(self, split, transform=None, cache_data=True, limit=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Augmentation function.
            cache_data (bool): Whether to use/save cached .npy files.
            limit (int, optional): Limit number of fragments (for debugging).
        """
        self.split = split
        self.transform = transform
        self.patch_size = Config.PATCH_SIZE
        self.pad_size = self.patch_size // 2

        # Load metadata
        meta_path = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.metadata = pd.read_csv(meta_path)
        if limit:
            self.metadata = self.metadata.head(limit)

        self.fragments = []
        self.samples = []  # List of (frag_idx, y, x)

        # Ensure working directory exists for cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Load all fragments into memory
        for idx, row in self.metadata.iterrows():
            frag_id = str(row["fragment_id"])

            # 1. Load Data
            vol = self._load_volume(row, cache_data)
            mask = self._load_mask(row, cache_data)

            label = None
            if split in ["train", "val"]:
                label = self._load_label(row, cache_data)

            # 2. Normalize Volume (Z-score per fragment)
            # We compute stats on the whole volume for robustness and speed
            mean = np.mean(vol)
            std = np.std(vol)
            vol = (vol - mean) / (std + 1e-6)

            # 3. Pad Arrays
            # Volume: (D, H, W) -> Pad H, W
            # We use reflect padding for volume to avoid edge artifacts
            vol = np.pad(
                vol,
                (
                    (0, 0),
                    (self.pad_size, self.pad_size),
                    (self.pad_size, self.pad_size),
                ),
                mode="reflect",
            )

            # Mask: (H, W) -> Pad H, W
            # Constant 0 padding for mask
            mask = np.pad(
                mask,
                ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),
                mode="constant",
                constant_values=0,
            )

            if label is not None:
                label = np.pad(
                    label,
                    ((self.pad_size, self.pad_size), (self.pad_size, self.pad_size)),
                    mode="constant",
                    constant_values=0,
                )

            self.fragments.append(
                {"id": frag_id, "vol": vol, "mask": mask, "label": label}
            )

            # 4. Generate Sampling Coordinates
            # Coordinates (y, x) represent the center of the patch in the ORIGINAL image.
            # Due to padding, the patch corresponds to vol[:, y:y+size, x:x+size]

            if split == "test":
                # Grid sampling for inference
                h, w = row["height"], row["width"]
                ys = np.arange(0, h, Config.INFERENCE_STRIDE)
                xs = np.arange(0, w, Config.INFERENCE_STRIDE)

                # Filter grid points to those within the valid mask to save time
                for y in ys:
                    for x in xs:
                        # Check if center is in mask (using padded mask indices)
                        if mask[y + self.pad_size, x + self.pad_size] > 0:
                            self.samples.append((idx, y, x))

            elif split == "val":
                # Deterministic sampling from valid areas
                # np.where returns indices relative to the sliced array (original coords)
                valid_y, valid_x = np.where(
                    mask[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]
                    > 0
                )
                self.samples.extend([(idx, y, x) for y, x in zip(valid_y, valid_x)])

            elif split == "train":
                # Store all valid coords for random sampling
                valid_y, valid_x = np.where(
                    mask[self.pad_size : -self.pad_size, self.pad_size : -self.pad_size]
                    > 0
                )
                self.samples.extend([(idx, y, x) for y, x in zip(valid_y, valid_x)])

        # 5. Finalize Sampling Lists
        if split == "val":
            # Select a fixed, deterministic subset of samples
            rng = np.random.default_rng(Config.SEED)
            if len(self.samples) > Config.VAL_SAMPLE_SIZE:
                indices = rng.choice(
                    len(self.samples), Config.VAL_SAMPLE_SIZE, replace=False
                )
                indices.sort()  # Sort for consistency
                self.samples = [self.samples[i] for i in indices]

    def _load_volume(self, row, cache_data):
        frag_id = str(row["fragment_id"])
        cache_path = os.path.join(Config.WORKING_DIR, f"{frag_id}_volume.npy")

        if cache_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to reload if corrupt

        # Load from TIFs
        vol_path = os.path.join(Config.INPUT_DIR, row["surface_volume_path"])
        slices = []
        for i in range(Config.Z_DIM):
            fname = f"{i:02d}.tif"
            fpath = os.path.join(vol_path, fname)
            if os.path.exists(fpath):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    # Should not happen given EDA, but handle gracefully
                    img = np.zeros((row["height"], row["width"]), dtype=np.uint8)
                slices.append(img)
            else:
                # If a slice is missing, duplicate the last one or use zeros
                if slices:
                    slices.append(slices[-1].copy())
                else:
                    slices.append(
                        np.zeros((row["height"], row["width"]), dtype=np.uint8)
                    )

        volume = np.stack(slices, axis=0).astype(np.float32)

        if cache_data:
            np.save(cache_path, volume)

        return volume

    def _load_mask(self, row, cache_data):
        frag_id = str(row["fragment_id"])
        cache_path = os.path.join(Config.WORKING_DIR, f"{frag_id}_mask.npy")

        if cache_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass

        mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        if cache_data:
            np.save(cache_path, mask)
        return mask

    def _load_label(self, row, cache_data):
        frag_id = str(row["fragment_id"])
        cache_path = os.path.join(Config.WORKING_DIR, f"{frag_id}_label.npy")

        if cache_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass

        label_path = os.path.join(Config.INPUT_DIR, row["inklabels_path"])
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.float32)

        if cache_data:
            np.save(cache_path, label)
        return label

    def __len__(self):
        if self.split == "train":
            # Virtual epoch size
            return Config.STEPS_PER_EPOCH * Config.BATCH_SIZE
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == "train":
            # Random sampling: ignore idx and pick a random valid coordinate
            sample_idx = np.random.randint(len(self.samples))
            frag_idx, y, x = self.samples[sample_idx]
        else:
            frag_idx, y, x = self.samples[idx]

        frag = self.fragments[frag_idx]

        # Extract Patch
        # y, x are coordinates in the ORIGINAL image.
        # In the padded array, the patch starts at y and ends at y + patch_size
        vol_patch = frag["vol"][:, y : y + self.patch_size, x : x + self.patch_size]

        if self.split == "test":
            # Return volume and metadata for reconstruction
            # Meta: [frag_idx, y, x]
            meta = torch.tensor([frag_idx, y, x], dtype=torch.long)
            return torch.from_numpy(vol_patch.copy()), meta

        label_patch = frag["label"][y : y + self.patch_size, x : x + self.patch_size]

        # Augmentation
        if self.split == "train" and self.transform:
            vol_patch, label_patch = self.transform(vol_patch, label_patch)

        # Add channel dimension to label: (H, W) -> (1, H, W)
        label_patch = np.expand_dims(label_patch, axis=0)

        # Return copies to ensure memory continuity (negative strides from flips)
        return torch.from_numpy(vol_patch.copy()), torch.from_numpy(label_patch.copy())
