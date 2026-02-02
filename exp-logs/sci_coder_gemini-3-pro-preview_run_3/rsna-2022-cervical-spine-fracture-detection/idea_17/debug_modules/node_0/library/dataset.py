import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pydicom
import albumentations as A
from albumentations.core.composition import ReplayCompose

from library.config import Config


def process_and_cache_scan(study_id, image_dir, load_cached_data=True):
    """
    Loads, processes, and caches a CT scan.

    Steps:
    1. Check cache for .npy file.
    2. If not found or force reload:
       - Load DICOMs.
       - Sort by Z-position.
       - Convert to HU.
       - Apply Bone Window.
       - Center Crop to 224x224.
       - Save as uint8 .npy.
    3. Return the volume.

    Args:
        study_id (str): The StudyInstanceUID.
        image_dir (str): Path to the directory containing DICOM files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: 3D volume of shape (Depth, 224, 224) in uint8 format.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            volume = np.load(cache_path)
            return volume
        except Exception as e:
            print(f"Failed to load cache for {study_id}: {e}. Reprocessing.")
            # Fall through to processing logic

    # 2. Process from scratch
    try:
        # List all DICOM files
        dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
        if not dcm_files:
            # Fallback if no extension
            dcm_files = glob.glob(os.path.join(image_dir, "*"))

        if not dcm_files:
            raise FileNotFoundError(f"No files found in {image_dir}")

        # Read headers to sort by Z position
        # We read just enough metadata to sort
        slices = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(
                    f, stop_before_pixels=False
                )  # Need pixels later anyway
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            raise ValueError(f"No valid DICOMs in {image_dir}")

        # Sort by ImagePositionPatient Z coordinate
        # ImagePositionPatient is [x, y, z]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Extract Pixel Data and Convert to HU
        images = []
        for s in slices:
            # Get raw pixels
            img = s.pixel_array.astype(np.float32)

            # Convert to HU: pixel * slope + intercept
            slope = getattr(s, "RescaleSlope", 1.0)
            intercept = getattr(s, "RescaleIntercept", 0.0)
            img = img * slope + intercept

            images.append(img)

        volume = np.stack(images)  # (Depth, H, W)

        # Apply Bone Window (Level 400, Width 1800)
        # Min = 400 - 900 = -500
        # Max = 400 + 900 = 1300
        center = Config.WINDOW_LEVEL
        width = Config.WINDOW_WIDTH
        lower = center - (width / 2)
        upper = center + (width / 2)

        volume = np.clip(volume, lower, upper)

        # Normalize to 0-255
        volume = (volume - lower) / (upper - lower)
        volume = volume * 255.0
        volume = volume.astype(np.uint8)

        # Center Crop to 224x224
        # Assumes images are usually 512x512. We crop the center.
        _, h, w = volume.shape
        target_size = Config.IMAGE_SIZE

        if h < target_size or w < target_size:
            # Pad if smaller (unlikely for CT, but safe)
            pad_h = max(0, target_size - h)
            pad_w = max(0, target_size - w)
            volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            _, h, w = volume.shape

        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2

        volume = volume[
            :, start_h : start_h + target_size, start_w : start_w + target_size
        ]

        # Save to cache
        np.save(cache_path, volume)

        return volume

    except Exception as e:
        # In case of failure, return a dummy volume to prevent training crash,
        # but print error. Or raise. Raising is better for debugging.
        raise RuntimeError(f"Error processing scan {study_id}: {e}")


class RSNADataset(Dataset):
    """
    PyTorch Dataset for RSNA Cervical Spine Fracture Detection.

    Features:
    - Loads pre-processed cached volumes.
    - Uniformly samples 64 slices.
    - Applies volumetric-consistent augmentations.
    - Returns (64, 3, 224, 224) tensors.
    """

    def __init__(self, subset="train", transform=None, debug=Config.DEBUG):
        """
        Args:
            subset (str): "train", "val", or "test".
            transform (albumentations.Compose): Optional custom transform.
                                                If None, uses default based on subset.
            debug (bool): If True, limits dataset size.
        """
        self.subset = subset
        self.debug = debug

        # Load Metadata
        if subset == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
            self.image_dir_base = (
                Config.INPUT_DIR
            )  # Relative paths in csv include 'train_images/'
        elif subset == "val":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
            self.image_dir_base = Config.INPUT_DIR
        elif subset == "test":
            self.df = pd.read_csv(Config.TEST_METADATA_PATH)
            self.image_dir_base = Config.INPUT_DIR
        else:
            raise ValueError("Subset must be 'train', 'val', or 'test'")

        if self.debug:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Define Augmentations
        # We use ReplayCompose to ensure consistency across the Z-axis (slices)
        if transform:
            self.transform = transform
        elif subset == "train":
            # Train: ShiftScaleRotate
            self.transform = A.ReplayCompose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5,
                        border_mode=0,  # Constant 0 padding
                    )
                ]
            )
        else:
            # Val/Test: No geometric augmentation
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Construct full path to image directory
        # metadata 'image_path' is relative like 'train_images/1.2.3...'
        image_dir = os.path.join(self.image_dir_base, row["image_path"])

        # Load Volume (Depth, H, W)
        # For test set, we might not have cached data yet, so load_cached_data=True
        # will trigger processing if missing.
        volume = process_and_cache_scan(study_id, image_dir, load_cached_data=True)

        # Uniform Sampling of 64 slices
        current_depth = volume.shape[0]
        target_depth = Config.SEQ_LEN

        if current_depth > 0:
            # Generate indices evenly spaced
            indices = np.linspace(0, current_depth - 1, target_depth).astype(int)
            volume = volume[indices]
        else:
            # Handle edge case of empty volume (should not happen with valid data)
            volume = np.zeros(
                (target_depth, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
            )

        # Apply Augmentations
        # volume is (64, 224, 224)
        if self.transform:
            # Apply transform to the first slice to get parameters
            # ReplayCompose requires 'image' kwarg
            res = self.transform(image=volume[0])
            augmented_slices = [res["image"]]
            replay_params = res["replay"]

            # Apply exact same transform to remaining slices
            for i in range(1, target_depth):
                res = A.ReplayCompose.replay(replay_params, image=volume[i])
                augmented_slices.append(res["image"])

            volume = np.stack(augmented_slices)

        # Format for Model
        # Current shape: (64, 224, 224) uint8 [0-255]
        # Target: (64, 3, 224, 224) float32 [0-1]

        volume = volume.astype(np.float32) / 255.0

        # Replicate channels (Grayscale -> RGB)
        # (64, 224, 224) -> (64, 224, 224, 3)
        volume = np.stack([volume, volume, volume], axis=-1)

        # Transpose to (64, 3, 224, 224) for PyTorch (Seq, C, H, W)
        volume = np.transpose(volume, (0, 3, 1, 2))

        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        # Get Targets
        if self.subset != "test":
            # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
            # Order must match RSNALoss expectation
            target_cols = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]
            targets = row[target_cols].values.astype(np.float32)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)
            return volume_tensor, targets_tensor
        else:
            # For test, return row_id base (StudyInstanceUID) to help map predictions later
            # But standard Dataset usually returns inputs.
            # We return dummy targets or just the image.
            # The training loop expects (data, target).
            return volume_tensor, torch.zeros(8)
