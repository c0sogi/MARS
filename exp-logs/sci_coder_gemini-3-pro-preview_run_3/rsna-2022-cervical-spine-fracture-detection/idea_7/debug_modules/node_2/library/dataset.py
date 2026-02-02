import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import pydicom
from library.config import Config
from library.transforms import get_transforms


class CervicalSpineDataset(Dataset):
    """
    Dataset class for RSNA Cervical Spine Fracture Detection.

    Implements the data pipeline for the Multi-Scale Contextualized Instance-MIL Network:
    1. Loads DICOM series.
    2. Applies Bone Windowing (WL=1000, WW=2000).
    3. Constructs 2.5D slices (z-1, z, z+1).
    4. Uniformly samples 64 slices per study.
    5. Caches processed volumes as .npy files for high-throughput training.
    """

    def __init__(self, phase="train", transform=None, n_slices=None):
        self.phase = phase
        self.n_slices = n_slices if n_slices is not None else Config.N_SLICES
        self.transform = transform if transform else get_transforms(phase)

        # Select metadata file and image directory based on phase
        if self.phase == "train":
            self.metadata_path = Config.TRAIN_METADATA_PATH
            self.image_dir = Config.TRAIN_IMAGES_DIR
        elif self.phase == "valid":
            self.metadata_path = Config.VAL_METADATA_PATH
            self.image_dir = (
                Config.TRAIN_IMAGES_DIR
            )  # Validation is a subset of train images
        else:
            self.metadata_path = Config.TEST_METADATA_PATH
            self.image_dir = Config.TEST_IMAGES_DIR

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Target columns (8 labels: Patient Overall + 7 Vertebrae)
        self.target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def window_image(self, dcm):
        """
        Applies bone windowing, resizes, and normalizes the DICOM image.

        Args:
            dcm: pydicom dataset object.

        Returns:
            np.ndarray: Processed image (H, W) in uint8 [0, 255].
        """
        try:
            # Read pixel array
            img = dcm.pixel_array.astype(np.float32)
        except Exception:
            # Fallback for corrupt or unreadable pixels
            return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8)

        # Rescale to Hounsfield Units (HU)
        intercept = getattr(dcm, "RescaleIntercept", 0.0)
        slope = getattr(dcm, "RescaleSlope", 1.0)
        img = img * slope + intercept

        # Apply Bone Window (Center=1000, Width=2000) -> Range [0, 2000]
        # This highlights cortical bone structure relevant for fractures
        center = 1000
        width = 2000
        lower = center - width // 2
        upper = center + width // 2

        img = np.clip(img, lower, upper)

        # Normalize to [0, 1]
        if (upper - lower) != 0:
            img = (img - lower) / (upper - lower)
        else:
            img = img - lower

        # Resize to target dimension (256x256)
        if img.shape[0] != Config.IMAGE_SIZE or img.shape[1] != Config.IMAGE_SIZE:
            img = cv2.resize(
                img,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )

        # Convert to uint8 [0, 255] for storage efficiency
        img = (img * 255).astype(np.uint8)

        return img

    def load_dicom_volume(self, study_id):
        """
        Loads all DICOMs for a study, sorts by Z-position, and returns the 3D volume.
        """
        study_path = os.path.join(self.image_dir, study_id)
        files = glob.glob(os.path.join(study_path, "*.dcm"))

        if not files:
            # If no files found, return a dummy volume
            return np.zeros(
                (self.n_slices, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
            )

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)

                # Determine Z position for sorting
                if hasattr(dcm, "ImagePositionPatient"):
                    z = float(dcm.ImagePositionPatient[2])
                elif hasattr(dcm, "InstanceNumber"):
                    z = float(dcm.InstanceNumber)
                else:
                    z = 0.0

                slices.append((z, dcm))
            except Exception:
                continue

        if not slices:
            return np.zeros(
                (self.n_slices, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.uint8
            )

        # Sort by Z position (Superior to Inferior or vice versa, consistency matters)
        slices.sort(key=lambda x: x[0])

        # Process images
        volume = []
        for _, dcm in slices:
            img = self.window_image(dcm)
            volume.append(img)

        return np.array(volume)  # Shape: (Depth, H, W)

    def process_study(self, study_id, load_cached_data=True):
        """
        Orchestrates loading, 2.5D stacking, sampling, and caching.

        Args:
            study_id (str): The StudyInstanceUID.
            load_cached_data (bool): Whether to attempt loading from disk cache.

        Returns:
            np.ndarray: Sampled 2.5D volume of shape (n_slices, H, W, 3).
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{study_id}.npy")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                cached_data = np.load(cache_path)
                # Validate shape against current config to ensure cache consistency
                # Expected shape: (n_slices, H, W, 3)
                expected_shape = (
                    self.n_slices,
                    Config.IMAGE_SIZE,
                    Config.IMAGE_SIZE,
                    3,
                )
                if cached_data.shape == expected_shape:
                    return cached_data
                # If shape mismatch, ignore cache and recompute
            except Exception:
                # If load fails (e.g. corrupt file), proceed to recompute
                pass

        # 2. Compute from scratch
        volume = self.load_dicom_volume(study_id)
        D = volume.shape[0]

        if D == 0:
            return np.zeros(
                (self.n_slices, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8
            )

        # 3. Uniform Sampling & 2.5D Construction
        # Generate indices for uniform sampling
        indices = np.linspace(0, D - 1, self.n_slices).astype(int)

        sampled_stack = []
        for i in indices:
            # 2.5D Context: [z-1, z, z+1]
            # Handle boundaries by clamping indices
            idx_prev = max(0, i - 1)
            idx_curr = i
            idx_next = min(D - 1, i + 1)

            img_prev = volume[idx_prev]
            img_curr = volume[idx_curr]
            img_next = volume[idx_next]

            # Stack along channel dimension -> (H, W, 3)
            slice_25d = np.stack([img_prev, img_curr, img_next], axis=-1)
            sampled_stack.append(slice_25d)

        sampled_stack = np.array(sampled_stack)  # Shape: (n_slices, H, W, 3)

        # 4. Save to cache
        try:
            np.save(cache_path, sampled_stack)
        except Exception:
            # If saving fails (e.g. disk full), we continue without caching
            pass

        return sampled_stack

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Process/Load data
        # We always attempt to load cache or create it if missing
        volume = self.process_study(study_id, load_cached_data=True)

        # Apply Transforms
        # Input: (N, H, W, 3) uint8
        # Output: (N, 3, H, W) float32 tensor normalized
        if self.transform:
            volume = self.transform(volume)

        # Prepare Targets
        if self.phase != "test":
            # Return 8 labels: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
            labels = row[self.target_cols].values.astype(np.float32)
            return volume, torch.tensor(labels), study_id
        else:
            # Return empty tensor for labels during inference
            return volume, torch.tensor([]), study_id
