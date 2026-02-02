import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# Attempt to import pydicom if available, otherwise rely on cv2
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


class BreastCancerDataset(Dataset):
    """
    Dataset class for Breast Cancer Detection using Multi-Contrast Single-Instance Network (MC-SIN).
    Implements the 'Simulated Windowing' input engineering strategy.
    """

    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.mode = mode
        self.transform = transform

        # Pre-compute CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=Config.CLAHE_CLIP_LIMIT, tileGridSize=Config.CLAHE_TILE_GRID_SIZE
        )

        # Pre-compute Gamma Look-Up Table (LUT)
        self.gamma_lut = self._create_gamma_lut(Config.GAMMA_VALUE)

    def _create_gamma_lut(self, gamma):
        """Creates a lookup table for gamma correction."""
        lut = np.empty((1, 256), dtype=np.uint8)
        for i in range(256):
            lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return lut

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Image Loading
        file_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = self._load_image(file_path)

        # 2. Preprocessing & Input Engineering (MC-SIN)
        # Ensure image is 0-255 uint8 for OpenCV operations
        img_uint8 = self._normalize_to_uint8(img)

        # Channel 1: Standard (Linear Normalization)
        ch1 = img_uint8.astype(np.float32) / 255.0

        # Channel 2: Structure (CLAHE)
        ch2_uint8 = self.clahe.apply(img_uint8)
        ch2 = ch2_uint8.astype(np.float32) / 255.0

        # Channel 3: Density (Gamma Correction)
        ch3_uint8 = cv2.LUT(img_uint8, self.gamma_lut)
        ch3 = ch3_uint8.astype(np.float32) / 255.0

        # Stack channels: (H, W, 3)
        img_merged = np.dstack([ch1, ch2, ch3])

        # 3. Resizing
        # Use cv2 for speed, resize to Config.IMG_SIZE (640, 640)
        # Note: cv2.resize expects (width, height)
        img_resized = cv2.resize(img_merged, (Config.IMG_SIZE[1], Config.IMG_SIZE[0]))

        # 4. To Tensor
        # Transpose from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float()

        # 5. Targets & Metadata
        # For Test set, 'cancer' column might not exist
        if "cancer" in row:
            target = torch.tensor(row["cancer"], dtype=torch.float32)
        else:
            target = torch.tensor(-1.0, dtype=torch.float32)  # Dummy for test

        # Construct prediction_id
        if "prediction_id" in row:
            pred_id = row["prediction_id"]
        else:
            # Fallback for train/val if needed for tracking
            pred_id = f"{row['patient_id']}_{row['laterality']}"

        return img_tensor, target, pred_id

    def _load_image(self, path):
        """Robust image loading trying pydicom first, then cv2."""
        img = None

        # Try pydicom if available
        if HAS_PYDICOM:
            try:
                dcm = pydicom.dcmread(path)
                img = dcm.pixel_array

                # Handle Photometric Interpretation (Monochrome1 means white is 0)
                if (
                    hasattr(dcm, "PhotometricInterpretation")
                    and dcm.PhotometricInterpretation == "MONOCHROME1"
                ):
                    img = np.max(img) - img
            except Exception:
                pass

        # Fallback to cv2 (works for some DICOMs or if converted)
        if img is None:
            try:
                # IMREAD_UNCHANGED is critical for 16-bit depth
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            except Exception:
                pass

        # Final fallback: create black image
        if img is None:
            # Create a dummy image of reasonable size
            img = np.zeros((1024, 1024), dtype=np.uint8)

        return img

    def _normalize_to_uint8(self, img):
        """Normalizes any bit-depth image to 0-255 uint8."""
        if img.dtype == np.uint8:
            return img

        # Handle 16-bit or other depths
        img_min = img.min()
        img_max = img.max()

        if img_max > img_min:
            # Linear scaling to 0-255
            img_norm = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img_norm = np.zeros_like(img, dtype=np.float32)

        return img_norm.astype(np.uint8)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Unused in this implementation as we load from metadata CSVs,
                                 but kept for signature compatibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Debug Mode
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Subsampling datasets to {Config.DEBUG_SAMPLE_SIZE} samples."
        )
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Create Datasets
    train_dataset = BreastCancerDataset(train_df, mode="train")
    val_dataset = BreastCancerDataset(val_df, mode="val")
    test_dataset = BreastCancerDataset(test_df, mode="test")

    # 4. Create DataLoaders
    # Train: Shuffle=True for natural distribution sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    # Val/Test: Shuffle=False for sequential evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
