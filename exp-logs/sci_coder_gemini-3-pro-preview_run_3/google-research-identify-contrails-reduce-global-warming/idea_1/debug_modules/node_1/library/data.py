import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class ContrailDataset(Dataset):
    """
    PyTorch Dataset for Contrail Detection.
    Loads infrared bands, applies Ash composite physics, and generates temporal features.
    """

    def __init__(self, metadata_df, split="train", transform=None):
        self.metadata_df = metadata_df
        self.split = split
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Required bands for Ash composite
        self.band_names = ["band_11", "band_13", "band_14", "band_15"]

        # Temporal indices: 4 is the labeled frame, 3 is the frame immediately before
        self.t_current = 4
        self.t_prev = 3

    def __len__(self):
        return len(self.metadata_df)

    def normalize_range(self, data, min_val, max_val):
        """
        Linearly normalizes data to [0, 1] based on provided min/max bounds.
        Clips values falling outside the range.
        """
        return np.clip((data - min_val) / (max_val - min_val), 0, 1)

    def get_ash_composite(self, t11, t13, t14, t15):
        """
        Generates the Ash false-color composite from brightness temperatures.

        Args:
            t11, t13, t14, t15: Numpy arrays of brightness temperatures for respective bands.

        Returns:
            np.ndarray: A (H, W, 3) array with normalized R, G, B channels.
        """
        # Ash Recipe
        # Red: Band 15 - Band 13 (Optical depth proxy)
        # Green: Band 14 - Band 11 (Particle size proxy)
        # Blue: Band 14 (Temperature)

        r = t15 - t13
        g = t14 - t11
        b = t14

        # Normalize using domain-specific bounds
        r_norm = self.normalize_range(r, Config.ASH_RED_MIN, Config.ASH_RED_MAX)
        g_norm = self.normalize_range(g, Config.ASH_GREEN_MIN, Config.ASH_GREEN_MAX)
        b_norm = self.normalize_range(b, Config.ASH_BLUE_MIN, Config.ASH_BLUE_MAX)

        return np.stack([r_norm, g_norm, b_norm], axis=-1)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        record_id = str(row["record_id"])

        # Dictionary to hold raw band data for current and previous time steps
        bands_data = {}

        for b_name in self.band_names:
            # Construct full path from relative path in CSV
            path = os.path.join(self.input_dir, row[b_name])

            try:
                # Load NPY file. Using mmap_mode='r' allows us to slice specific time steps
                # without loading the entire HxWxT array into RAM.
                full_band = np.load(path, mmap_mode="r")

                # Extract specific time steps and convert to float32
                bands_data[f"{b_name}_curr"] = full_band[:, :, self.t_current].astype(
                    np.float32
                )
                bands_data[f"{b_name}_prev"] = full_band[:, :, self.t_prev].astype(
                    np.float32
                )

            except Exception as e:
                # In case of file error, return a zero tensor (should not happen with verified metadata)
                print(f"Error loading {path}: {e}")
                return (
                    torch.zeros(
                        (Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
                    ),
                    torch.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                    record_id,
                )

        # --- Feature Engineering ---

        # 1. Generate Ash Composite for Current Frame (t)
        ash_curr = self.get_ash_composite(
            bands_data["band_11_curr"],
            bands_data["band_13_curr"],
            bands_data["band_14_curr"],
            bands_data["band_15_curr"],
        )  # Shape: (H, W, 3)

        # 2. Generate Ash Composite for Previous Frame (t-1)
        ash_prev = self.get_ash_composite(
            bands_data["band_11_prev"],
            bands_data["band_13_prev"],
            bands_data["band_14_prev"],
            bands_data["band_15_prev"],
        )  # Shape: (H, W, 3)

        # 3. Compute Temporal Difference
        # Captures the change in spectral properties over 10 minutes
        diff = ash_curr - ash_prev  # Shape: (H, W, 3)

        # 4. Construct Input Tensor
        # Concatenate spatial (current) and temporal (difference) features
        # Result: 6 channels
        img = np.concatenate([ash_curr, diff], axis=-1)  # (H, W, 6)

        # Transpose to PyTorch format: (Channels, Height, Width)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).float()

        # --- Load Ground Truth Mask ---
        mask_tensor = torch.zeros(
            (1, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=torch.float32
        )

        if self.split in ["train", "validation"]:
            mask_path = os.path.join(self.input_dir, row["human_pixel_masks"])
            if os.path.exists(mask_path):
                # Load mask: Shape (H, W, 1)
                mask = np.load(mask_path)
                # Transpose to (1, H, W)
                mask = np.transpose(mask, (2, 0, 1))
                mask_tensor = torch.from_numpy(mask).float()

        return img_tensor, mask_tensor, record_id


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
):
    """
    Constructs PyTorch DataLoaders for training, validation, and testing.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        debug (bool): If True, uses a small subset of data for quick testing.
        debug_sample_size (int): Number of samples to use in debug mode.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Apply Debug Filtering
    if debug:
        set_seed(Config.SEED)  # Ensure consistent subsampling
        train_df = train_df.sample(
            n=min(len(train_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), debug_sample_size), random_state=Config.SEED
        ).reset_index(drop=True)
        print(
            f"Debug Mode: Datasets reduced to {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples."
        )

    # Instantiate Datasets
    train_dataset = ContrailDataset(train_df, split="train")
    val_dataset = ContrailDataset(val_df, split="validation")
    test_dataset = ContrailDataset(test_df, split="test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches during training for stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
