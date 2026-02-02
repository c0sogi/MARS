import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Constants for Normalization
# ==========================================
# Bounds for Ash Color Scheme (Brightness Temperatures in Kelvin)
# Derived from domain standards for Contrail/Ash detection
BOUNDS_R = (-6.7, 2.6)  # Band 15 - Band 13
BOUNDS_G = (-6.3, 5.7)  # Band 14 - Band 11
BOUNDS_B = (243, 303)  # Band 13


def normalize_range(data, bounds):
    """
    Normalize data to [0, 1] using provided bounds.
    Clips values outside the bounds.
    """
    return (np.clip(data, bounds[0], bounds[1]) - bounds[0]) / (bounds[1] - bounds[0])


def get_ash_colors(b11, b13, b14, b15):
    """
    Computes the Ash False Color Composite.

    Args:
        b11, b13, b14, b15: 2D numpy arrays (H, W) representing brightness temperatures.

    Returns:
        ash_composite: 3D numpy array (H, W, 3) normalized to [0, 1].
    """
    r = normalize_range(b15 - b13, BOUNDS_R)
    g = normalize_range(b14 - b11, BOUNDS_G)
    b = normalize_range(b13, BOUNDS_B)

    # Stack along last dimension for Albumentations (H, W, C)
    return np.stack([r, g, b], axis=-1)


def get_transforms(mode="train", image_size=256):
    """
    Returns the Albumentations transform pipeline.

    Args:
        mode (str): 'train', 'validation', or 'test'.
        image_size (int): Target image size.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                # Affine transformations only (Rotation, Scale, Shift)
                # Elastic/Grid distortions are excluded to preserve linear features
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5,
                    border_mode=0,  # Constant border (0)
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return A.Compose([ToTensorV2()])


class ContrailDataset(Dataset):
    def __init__(
        self, metadata, input_dir, mode="train", transform=None, n_times_before=4
    ):
        """
        Args:
            metadata (pd.DataFrame): Dataframe containing file paths and metadata.
            input_dir (str): Root directory of the input data.
            mode (str): 'train', 'validation', or 'test'.
            transform (albumentations.Compose): Augmentation pipeline.
            n_times_before (int): Number of frames before the labeled frame.
                                  Labeled frame is at index n_times_before.
        """
        self.metadata = metadata
        self.input_dir = input_dir
        self.mode = mode
        self.transform = transform
        self.n_times_before = n_times_before

        # Check if masks are available in the metadata
        self.has_masks = "human_pixel_masks" in self.metadata.columns

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        record_id = str(row["record_id"])

        # Helper to load a specific band file
        # Paths in metadata are relative (e.g., "train/ID/band_11.npy")
        def load_band(band_name):
            path = os.path.join(self.input_dir, row[band_name])
            return np.load(path)

        # Load required bands (11, 13, 14, 15)
        # Data shape in file: (H, W, T)
        try:
            b11_seq = load_band("band_11")
            b13_seq = load_band("band_13")
            b14_seq = load_band("band_14")
            b15_seq = load_band("band_15")
        except Exception as e:
            # Fallback for corrupted files (should not happen based on validation)
            print(f"Error loading bands for {record_id}: {e}")
            # Return a zero tensor as fallback to avoid crashing
            if self.mode in ["train", "validation"]:
                return torch.zeros((6, 256, 256)), torch.zeros((1, 256, 256))
            return torch.zeros((6, 256, 256)), record_id

        # Determine time indices
        # t_current is the labeled frame
        t_current = self.n_times_before
        # t_prev is the frame immediately before
        t_prev = t_current - 1

        # 1. Generate Ash Composite for t=4 (Current)
        ash_t4 = get_ash_colors(
            b11_seq[..., t_current],
            b13_seq[..., t_current],
            b14_seq[..., t_current],
            b15_seq[..., t_current],
        )  # Shape: (H, W, 3)

        # 2. Generate Ash Composite for t=3 (Previous)
        ash_t3 = get_ash_colors(
            b11_seq[..., t_prev],
            b13_seq[..., t_prev],
            b14_seq[..., t_prev],
            b15_seq[..., t_prev],
        )  # Shape: (H, W, 3)

        # 3. Compute Temporal Difference
        # Channels 4-6: Ash_t4 - Ash_t3
        diff = ash_t4 - ash_t3

        # 4. Construct Final Input Tensor
        # Concatenate along channel axis: (H, W, 6)
        img = np.concatenate([ash_t4, diff], axis=-1)

        # 5. Load Mask (if available and needed)
        mask = None
        if self.has_masks and self.mode in ["train", "validation"]:
            mask_path = os.path.join(self.input_dir, row["human_pixel_masks"])
            mask = np.load(mask_path)  # Shape: (H, W, 1)
            mask = mask.astype(np.float32)

        # 6. Apply Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]

                # Ensure mask is (1, H, W)
                # ToTensorV2 with mask input (H, W, 1) might return (H, W, 1) or (H, W) depending on version
                # We enforce (1, H, W)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                elif mask.ndim == 3 and mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    # Fallback if it became HWC
                    mask = mask.permute(2, 0, 1)
            else:
                augmented = self.transform(image=img)
                img = augmented["image"]

        # Ensure image is float32
        img = img.float()

        if self.mode in ["train", "validation"]:
            return img, mask
        else:
            return img, record_id


def load_metadata(config, mode="train"):
    """
    Loads the appropriate metadata CSV based on the mode.

    Args:
        config (Config): Configuration object.
        mode (str): 'train', 'validation', or 'test'.

    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    if mode == "train":
        path = config.train_metadata_path
    elif mode == "validation":
        path = config.valid_metadata_path
    elif mode == "test":
        path = config.test_metadata_path
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    # Debug mode: sample the dataset
    if config.debug:
        sample_n = (
            config.train_sample_size if mode == "train" else config.valid_sample_size
        )
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=config.seed).reset_index(drop=True)

    return df


def get_dataloader(config, mode="train"):
    """
    Factory function to create a DataLoader.

    Args:
        config (Config): Configuration object.
        mode (str): 'train', 'validation', or 'test'.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    df = load_metadata(config, mode)

    transforms = get_transforms(mode, config.image_size)

    dataset = ContrailDataset(
        metadata=df,
        input_dir=config.input_dir,
        mode=mode,
        transform=transforms,
        n_times_before=config.n_times_before,
    )

    shuffle = mode == "train"
    drop_last = mode == "train"

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
