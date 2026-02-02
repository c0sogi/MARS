import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_volume


class InkDataset(Dataset):
    """
    Dataset for 3D Papyrus Ink Detection.
    Handles volume loading, patch extraction, normalization, and augmentation.
    """

    def __init__(self, split, fragment_ids, samples_per_epoch=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            fragment_ids (list): List of fragment IDs to load.
            samples_per_epoch (int, optional): Number of samples per epoch for training.
        """
        self.split = split
        self.fragment_ids = fragment_ids
        self.patch_size = Config.PATCH_SIZE
        self.half_size = self.patch_size // 2

        # Determine samples per epoch
        if samples_per_epoch is not None:
            self.samples_per_epoch = samples_per_epoch
        else:
            # Default heuristic if not provided
            if Config.DEBUG:
                self.samples_per_epoch = Config.DEBUG_SAMPLE_SIZE
            else:
                # Approximate a reasonable epoch size for training
                self.samples_per_epoch = 10000

        # Storage for loaded data
        self.volumes = []
        self.masks = []
        self.labels = []

        # Valid coordinates for sampling
        # List of tuples: (fragment_index, y_center, x_center)
        self.valid_indices = []

        self._load_data()

    def _load_data(self):
        """Loads volumes and prepares indices."""
        for i, fid in enumerate(self.fragment_ids):
            # Load data using utility function (handles caching)
            vol, mask, label = load_volume(fid, self.split, load_cached_data=True)

            # Pad data to allow sampling at edges
            # Padding is applied to H and W dimensions
            pad_width = (
                (0, 0),
                (self.half_size, self.half_size),
                (self.half_size, self.half_size),
            )
            vol_padded = np.pad(vol, pad_width, mode="constant", constant_values=0)

            # Mask and Label are 2D
            pad_width_2d = (
                (self.half_size, self.half_size),
                (self.half_size, self.half_size),
            )
            mask_padded = np.pad(mask, pad_width_2d, mode="constant", constant_values=0)

            if label is not None:
                label_padded = np.pad(
                    label, pad_width_2d, mode="constant", constant_values=0
                )
            else:
                label_padded = None

            self.volumes.append(vol_padded)
            self.masks.append(mask_padded)
            self.labels.append(label_padded)

            # Generate indices
            if self.split == "train":
                # For training, we want all valid pixel coordinates for random sampling
                ys, xs = np.where(mask_padded > 0)
                # Store as (frag_idx, y, x)
                # Note: ys, xs are already shifted due to padding, so they represent center points in padded img
                coords = np.stack([np.full_like(ys, i), ys, xs], axis=1)
                self.valid_indices.append(coords)
            else:
                # For val/test, generate a deterministic grid
                h, w = mask_padded.shape
                # Grid covers the original valid area (which is now centered in padded image)
                # Start from half_size, go to h-half_size
                y_range = np.arange(
                    self.half_size, h - self.half_size + 1, Config.INFERENCE_STRIDE
                )
                x_range = np.arange(
                    self.half_size, w - self.half_size + 1, Config.INFERENCE_STRIDE
                )

                # Ensure the last tile covers the edge
                if y_range[-1] < h - self.half_size:
                    y_range = np.append(y_range, h - self.half_size)
                if x_range[-1] < w - self.half_size:
                    x_range = np.append(x_range, w - self.half_size)

                grid_y, grid_x = np.meshgrid(y_range, x_range, indexing="ij")
                grid_y = grid_y.flatten()
                grid_x = grid_x.flatten()

                coords = np.stack([np.full_like(grid_y, i), grid_y, grid_x], axis=1)
                self.valid_indices.append(coords)

        # Concatenate all indices into a single lookup array
        self.valid_indices = np.concatenate(self.valid_indices, axis=0)

    def __len__(self):
        if self.split == "train":
            return self.samples_per_epoch
        else:
            return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Determine Coordinate
        if self.split == "train":
            # Random sampling for training
            idx = np.random.randint(0, len(self.valid_indices))

        frag_idx, y, x = self.valid_indices[idx]

        # 2. Extract Patch
        # Coordinates y, x are centers in the padded image
        y_min = y - self.half_size
        y_max = y + self.half_size
        x_min = x - self.half_size
        x_max = x + self.half_size

        volume_patch = self.volumes[frag_idx][:, y_min:y_max, x_min:x_max].copy()

        if self.labels[frag_idx] is not None:
            label_patch = self.labels[frag_idx][y_min:y_max, x_min:x_max].copy()
        else:
            # Create dummy label for test set
            label_patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)

        # 3. Normalization (Global Z-score)
        volume_patch = (volume_patch - Config.PIXEL_MEAN) / Config.PIXEL_STD

        # 4. Augmentation (Train Only)
        if self.split == "train":
            # Intensity Perturbation (Applied in normalized space)
            gain = np.random.uniform(*Config.INTENSITY_SCALE_RANGE)
            offset = np.random.uniform(*Config.INTENSITY_OFFSET_RANGE)
            volume_patch = volume_patch * gain + offset

            # Geometric Augmentation
            # Random Rotate 90
            k = np.random.randint(0, 4)
            volume_patch = np.rot90(volume_patch, k, axes=(1, 2))
            label_patch = np.rot90(label_patch, k, axes=(0, 1))

            # Random Flips
            if np.random.random() < 0.5:
                volume_patch = np.flip(volume_patch, axis=2)  # Flip Width
                label_patch = np.flip(label_patch, axis=1)

            if np.random.random() < 0.5:
                volume_patch = np.flip(volume_patch, axis=1)  # Flip Height
                label_patch = np.flip(label_patch, axis=0)

        # 5. Convert to Tensor
        # Volume: (Z, H, W) -> FloatTensor
        # Label: (H, W) -> (1, H, W) -> FloatTensor
        volume_tensor = torch.from_numpy(np.ascontiguousarray(volume_patch)).float()
        label_tensor = (
            torch.from_numpy(np.ascontiguousarray(label_patch)).float().unsqueeze(0)
        )

        # Return coordinate for reconstruction (y, x are centers in padded space)
        # We convert them back to original image space for reference
        orig_y = y - self.half_size
        orig_x = x - self.half_size
        coord = torch.tensor([frag_idx, orig_y, orig_x], dtype=torch.long)

        return volume_tensor, label_tensor, coord


def get_loaders():
    """
    Creates DataLoaders for training and validation.

    Returns:
        train_loader, val_loader
    """
    # 1. Read Metadata
    if not Config.TRAIN_METADATA_PATH.exists() or not Config.VAL_METADATA_PATH.exists():
        raise FileNotFoundError("Metadata files not found.")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    train_ids = df_train["fragment_id"].astype(str).tolist()
    val_ids = df_val["fragment_id"].astype(str).tolist()

    # 2. Configure Dataset Size
    # If Debug, use small sample size
    train_samples = Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else 8000

    # 3. Instantiate Datasets
    train_dataset = InkDataset(
        split="train", fragment_ids=train_ids, samples_per_epoch=train_samples
    )

    val_dataset = InkDataset(
        split="val", fragment_ids=val_ids, samples_per_epoch=None  # Unused for val
    )

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Deterministic order for validation
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
