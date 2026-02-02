import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config
from library.utils import seed_everything


class ContrailDataset(Dataset):
    """
    Dataset class for Contrail Identification.
    Loads satellite bands, computes Ash false-color composites, and constructs
    multi-order temporal inputs (Current, Velocity, Acceleration).
    """

    def __init__(
        self, split="train", transform=None, debug=Config.DEBUG, max_samples=None
    ):
        """
        Args:
            split (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            debug (bool): If True, limits dataset size for debugging.
            max_samples (int): Optional limit on number of samples.
        """
        self.split = split
        self.debug = debug

        # Load metadata
        metadata_file = os.path.join(Config.METADATA_DIR, f"{split}.csv")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.df = pd.read_csv(metadata_file)

        # Debug / Subsampling
        if self.debug or (max_samples is not None):
            limit = 100 if (self.debug and max_samples is None) else max_samples
            if limit and limit < len(self.df):
                self.df = self.df.iloc[:limit].reset_index(drop=True)
                # print(f"[{split}] Debug mode: Loaded {len(self.df)} samples.")

        # Define normalization constants for 9 channels
        # Channels 0-2: Ash T (RGB-like) -> Use ImageNet Mean/Std
        # Channels 3-5: Ash T - Ash T-1 (Difference) -> Mean 0, Std 0.225 (Scaled to match backbone variance)
        # Channels 6-8: Ash T-1 - Ash T-2 (Difference) -> Mean 0, Std 0.225
        self.mean = np.array(
            [0.485, 0.456, 0.406] + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0], dtype=np.float32
        )
        self.std = np.array(
            [0.229, 0.224, 0.225] + [0.229, 0.224, 0.225] + [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        # Setup Transform
        if transform is not None:
            self.transform = transform
        else:
            # Default Transforms
            if self.split == "train":
                self.transform = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                        ),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                        ToTensorV2(),
                    ]
                )
            else:
                self.transform = A.Compose(
                    [
                        A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                        ToTensorV2(),
                    ]
                )

    def __len__(self):
        return len(self.df)

    def _get_ash_vector(self, band11, band14, band15):
        """
        Computes the Ash false-color composite.
        Args:
            band11, band14, band15: 2D numpy arrays (H, W)
        Returns:
            ash: 3D numpy array (H, W, 3) normalized to [0, 1]
        """
        # Bounds based on physical properties
        # Band 15 - Band 14
        r = (band15 - band14 - (-6.7)) / (2.6 - (-6.7))

        # Band 14 - Band 11
        g = (band14 - band11 - (-6.0)) / (1.5 - (-6.0))

        # Band 14
        b = (band14 - 243) / (303 - 243)

        ash = np.stack([r, g, b], axis=-1)
        return np.clip(ash, 0, 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        record_id = str(row["record_id"])

        # 1. Load Bands
        # We need bands 11, 14, 15 for time steps t, t-1, t-2
        bands_data = {}
        for b in Config.ASH_BANDS:
            col_name = f"band_{b:02d}"
            # Path in CSV is relative to input dir (e.g. "train/id/band_11.npy")
            path = os.path.join(Config.ROOT_DIR, row[col_name])

            try:
                # Shape: (H, W, T)
                full_band = np.load(path)
            except Exception:
                # Fallback for missing files (robustness)
                full_band = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE, 8), dtype=np.float32
                )

            bands_data[b] = full_band

        # 2. Extract Temporal Slices & Compute Ash
        def get_ash_at_t(time_idx):
            # Handle edge case where time_idx might be out of bounds if file is corrupted
            # Assuming standard structure T=8
            b11 = bands_data[11][:, :, time_idx]
            b14 = bands_data[14][:, :, time_idx]
            b15 = bands_data[15][:, :, time_idx]
            return self._get_ash_vector(b11, b14, b15)

        # Indices: 4 (t), 3 (t-1), 2 (t-2)
        ash_t = get_ash_at_t(Config.TEMPORAL_INDICES[0])
        ash_tm1 = get_ash_at_t(Config.TEMPORAL_INDICES[1])
        ash_tm2 = get_ash_at_t(Config.TEMPORAL_INDICES[2])

        # 3. Compute Temporal Differences
        diff_1 = ash_t - ash_tm1
        diff_2 = ash_tm1 - ash_tm2

        # 4. Construct 9-channel input
        # Shape: (H, W, 9)
        image = np.concatenate([ash_t, diff_1, diff_2], axis=-1).astype(np.float32)

        # 5. Load Mask (if available)
        mask = None
        if self.split != "test":
            mask_path_col = "human_pixel_masks"
            if mask_path_col in row and pd.notna(row[mask_path_col]):
                mask_path = os.path.join(Config.ROOT_DIR, row[mask_path_col])
                try:
                    # Shape (H, W, 1)
                    mask = np.load(mask_path).astype(np.float32)
                    # Squeeze to (H, W) for Albumentations
                    mask = mask.squeeze(-1)
                except:
                    mask = np.zeros(
                        (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                    )
            else:
                mask = np.zeros(
                    (Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
                )

        # 6. Augmentation
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
                # ToTensorV2 converts mask to Tensor (H, W). We need (1, H, W) for training.
                mask = mask.unsqueeze(0).float()
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        result = {"image": image, "record_id": record_id}

        if mask is not None:
            result["mask"] = mask

        return result


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
):
    """
    Factory function to create DataLoaders.
    """
    dataset = ContrailDataset(split=split, debug=debug)

    # Drop last for training to maintain consistent batch statistics
    drop_last = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return loader
