import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Identification.
    Loads satellite bands, computes Ash color composite and temporal differences,
    and applies augmentations.
    """

    def __init__(self, metadata, split="train", transform=None):
        self.metadata = metadata
        self.split = split
        self.transform = transform
        self.is_train = split == "train"
        self.is_val = split == "validation"

        # --- Ash Color Scheme Constants ---
        # Bounds for normalization based on physical properties of contrails
        self._ASH_R_MIN = -4.0
        self._ASH_R_MAX = 2.0

        self._ASH_G_MIN = -4.0
        self._ASH_G_MAX = 5.0

        self._ASH_B_MIN = 243.0
        self._ASH_B_MAX = 303.0

    def __len__(self):
        return len(self.metadata)

    def normalize_range(self, data, mn, mx):
        """
        Normalizes data to [0, 1] based on provided min/max bounds.
        """
        return (data - mn) / (mx - mn)

    def get_ash_vector(self, t11, t13, t14, t15):
        """
        Computes the Ash false-color composite from brightness temperatures.

        Args:
            t11, t13, t14, t15: 2D numpy arrays (H, W)

        Returns:
            ash: 3D numpy array (H, W, 3) with values clipped to [0, 1]
        """
        # Red: Split Window Difference (Band 15 - Band 13)
        # Captures optical depth differences
        r = self.normalize_range(t15 - t13, self._ASH_R_MIN, self._ASH_R_MAX)

        # Green: Split Window Difference (Band 14 - Band 11)
        # Captures particle size / phase
        g = self.normalize_range(t14 - t11, self._ASH_G_MIN, self._ASH_G_MAX)

        # Blue: Band 14 (11.2 micron)
        # Captures temperature
        b = self.normalize_range(t14, self._ASH_B_MIN, self._ASH_B_MAX)

        ash = np.stack([r, g, b], axis=-1)
        return np.clip(ash, 0, 1)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # --- Load Band Data ---
        # We need bands 11, 13, 14, 15.
        # Files are H x W x T (T=8).
        # We need index 4 (current labeled frame) and index 3 (previous frame).

        bands_data = {}
        # Bands to load
        required_bands = [11, 13, 14, 15]

        for b in required_bands:
            col_name = f"band_{b}"
            # Path is relative to input directory
            path = os.path.join(Config.INPUT_DIR, row[col_name])

            try:
                # Load the full array.
                # Optimization: In a highly constrained env, we might use mmap_mode='r',
                # but these files are small (~2MB), so direct load is fine.
                full_band = np.load(path).astype(np.float32)
                bands_data[b] = full_band
            except Exception as e:
                # Fallback (should not happen given metadata verification)
                bands_data[b] = np.zeros(
                    (Config.IMG_SIZE, Config.IMG_SIZE, Config.SEQ_LENGTH),
                    dtype=np.float32,
                )

        # Helper to extract specific time step
        def get_band_t(b, t_idx):
            return bands_data[b][..., t_idx]

        t_curr = Config.N_TIMES_BEFORE  # Index 4
        t_prev = Config.N_TIMES_BEFORE - 1  # Index 3

        # --- Feature Engineering ---

        # 1. Compute Ash at t_curr
        ash_curr = self.get_ash_vector(
            get_band_t(11, t_curr),
            get_band_t(13, t_curr),
            get_band_t(14, t_curr),
            get_band_t(15, t_curr),
        )

        # 2. Compute Ash at t_prev
        ash_prev = self.get_ash_vector(
            get_band_t(11, t_prev),
            get_band_t(13, t_prev),
            get_band_t(14, t_prev),
            get_band_t(15, t_prev),
        )

        # 3. Compute Temporal Difference
        ash_diff = ash_curr - ash_prev

        # 4. Construct 6-channel input
        # Shape: (H, W, 6)
        image = np.concatenate([ash_curr, ash_diff], axis=-1).astype(np.float32)

        # --- Load Mask ---
        mask = None
        if self.is_train or self.is_val:
            mask_path = os.path.join(Config.INPUT_DIR, row["human_pixel_masks"])
            try:
                # Shape: (H, W, 1)
                mask = np.load(mask_path).astype(np.float32)
            except:
                mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.float32)

        # --- Augmentation & Normalization ---
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

                # Ensure mask is channel-first (C, H, W)
                # Albumentations usually returns mask as (H, W) or (H, W, C) depending on input
                # ToTensorV2 converts to tensor.
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3:
                    # If shape is (H, W, 1), permute to (1, H, W)
                    if mask.shape[2] == 1:
                        mask = mask.permute(2, 0, 1)
                    # If shape is already (1, H, W), leave it (unlikely from Albumentations default)
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # If no transform (e.g. raw test), ensure tensor format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1)  # H W C -> C H W

        if mask is not None:
            return image, mask
        else:
            # Return dummy label for test set
            return image, torch.tensor(0)


def get_transforms(split="train"):
    """
    Returns Albumentations transforms for the specified split.
    Includes ImageNet-style normalization for the 6-channel input.
    """
    # ImageNet stats extended to 6 channels
    # We apply the same normalization to the raw Ash channels and the Difference channels.
    # While difference channels are residual, standardizing them with the same scale
    # is a robust starting point for transfer learning.
    mean = [0.485, 0.456, 0.406] * 2  # Repeat for 6 channels
    std = [0.229, 0.224, 0.225] * 2  # Repeat for 6 channels

    if split == "train":
        return A.Compose(
            [
                # Geometric Augmentations
                A.ShiftScaleRotate(
                    p=0.5, rotate_limit=15, shift_limit=0.05, scale_limit=0.05
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # Normalization
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
    else:
        # Validation / Test: Only Normalize
        return A.Compose(
            [A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2()]
        )


def get_dataloader(
    split="train",
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders.

    Args:
        split (str): 'train', 'validation', or 'test'.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        debug (bool): If True, subsamples the dataset for debugging.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1. Determine Metadata Path
    if split == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif split == "validation":
        meta_path = Config.VALIDATION_METADATA_PATH
    elif split == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    # 2. Load Metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    df = pd.read_csv(meta_path)

    # 3. Debug Subsampling
    if debug:
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        print(f"DEBUG MODE: Subsampled {split} dataset to {len(df)} samples.")

    # 4. Create Dataset
    transform = get_transforms(split)
    dataset = ContrailDataset(df, split=split, transform=transform)

    # 5. Create DataLoader
    # Shuffle only for training
    shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(
            split == "train"
        ),  # Drop last batch in training to maintain stable batch norm stats
    )

    return loader
