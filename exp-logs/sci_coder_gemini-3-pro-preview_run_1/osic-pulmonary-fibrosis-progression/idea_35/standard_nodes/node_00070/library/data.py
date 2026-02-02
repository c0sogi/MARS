import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# Global Constants (from EDA)
# ==========================================
AGE_MEAN = 67.58
AGE_STD = 6.63
PERCENT_MEAN = 76.91
PERCENT_STD = 19.20

# ==========================================
# DICOM Processing Functions
# ==========================================


def get_img_paths(dicom_dir):
    """Returns sorted list of DICOM file paths."""
    return glob.glob(os.path.join(dicom_dir, "*.dcm"))


def load_scan(paths):
    """Loads DICOM slices and sorts them by ImagePositionPatient Z-axis."""
    slices = [pydicom.dcmread(p) for p in paths]

    # Sort by Z position with fallbacks (Cite debug_lesson_10)
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except AttributeError:
            slices.sort(key=lambda x: float(x.InstanceNumber))

    return slices


def get_pixels_hu(scans):
    """Converts raw DICOM pixel data to Hounsfield Units."""
    # Cite debug_lesson_2: Handle missing codecs gracefully
    try:
        image = np.stack([s.pixel_array for s in scans])
    except RuntimeError as e:
        raise RuntimeError(f"Pixel array access failed: {e}")

    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to HU
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def process_volume(volume_hu):
    """Applies Lung Windowing and Normalization."""
    # Lung Window: W=1500, L=-600
    # Range: [-1350, 150]
    min_hu = -600 - (1500 / 2)
    max_hu = -600 + (1500 / 2)

    volume_hu = np.clip(volume_hu, min_hu, max_hu)

    # Normalize to [0, 1]
    volume_norm = (volume_hu - min_hu) / (max_hu - min_hu)
    return volume_norm.astype(np.float32)


def generate_tri_slab(volume, axis=0, overlap=0.15, target_size=224):
    """
    Generates a 3-channel image from a 3D volume using Tri-Slab MIP.

    Args:
        volume: 3D numpy array (D, H, W) normalized to [0, 1]
        axis: 0 for Axial (splitting Depth), 1 for Coronal (splitting Height)
        overlap: Fraction of overlap between slabs
        target_size: Output spatial resolution
    """
    # If Coronal (axis=1), permute so we split along the first dimension
    if axis == 1:
        # Original: (D, H, W) -> Permute to (H, D, W)
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Calculate slab boundaries
    chunk = depth / 3.0
    ov_pixels = int(depth * overlap)

    # Slab 1: 0 to 33% + overlap
    s1_start = 0
    s1_end = int(chunk + ov_pixels)

    # Slab 2: 33% - overlap to 66% + overlap
    s2_start = int(chunk - ov_pixels)
    s2_end = int(2 * chunk + ov_pixels)

    # Slab 3: 66% - overlap to 100%
    s3_start = int(2 * chunk - ov_pixels)
    s3_end = depth

    # Clip boundaries
    s1_end = min(s1_end, depth)
    s2_start = max(0, s2_start)
    s2_end = min(s2_end, depth)
    s3_start = max(0, s3_start)

    # Extract slabs
    slab1 = volume[s1_start:s1_end, :, :]
    slab2 = volume[s2_start:s2_end, :, :]
    slab3 = volume[s3_start:s3_end, :, :]

    # Compute MIP (Maximum Intensity Projection)
    def get_mip(slab):
        if slab.shape[0] == 0:
            return np.zeros((volume.shape[1], volume.shape[2]), dtype=np.float32)
        return np.max(slab, axis=0)

    c1 = get_mip(slab1)
    c2 = get_mip(slab2)
    c3 = get_mip(slab3)

    # Stack to create RGB-like image (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # Resize to target resolution
    img_resized = cv2.resize(
        img, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    return img_resized


def get_patient_images(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Retrieves Axial and Coronal Tri-Slabs for a patient.
    Implements the required caching logic.
    """
    os.makedirs(cache_dir, exist_ok=True)

    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_dicom_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
    paths = get_img_paths(full_dicom_path)

    # Handle edge case with no DICOMs
    if not paths:
        dummy = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return dummy, dummy

    try:
        scans = load_scan(paths)
        vol_hu = get_pixels_hu(scans)
        vol_norm = process_volume(vol_hu)

        # Generate Views
        axial = generate_tri_slab(
            vol_norm, axis=0, overlap=Config.SLAB_OVERLAP, target_size=Config.IMG_SIZE
        )
        coronal = generate_tri_slab(
            vol_norm, axis=1, overlap=Config.SLAB_OVERLAP, target_size=Config.IMG_SIZE
        )

        # Save to cache
        np.save(axial_path, axial)
        np.save(coronal_path, coronal)

        return axial, coronal

    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        dummy = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Cite debug_lesson_8: Cache Fallback Results to Prevent Repeated Failure Overheads
        np.save(axial_path, dummy)
        np.save(coronal_path, dummy)

        return dummy, dummy


# ==========================================
# Augmentation
# ==========================================


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Spatial-only augmentations for training.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ==========================================
# Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, csv_path, mode="train", transform=None):
        """
        Args:
            csv_path: Path to the metadata CSV file.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transform.
        """
        self.mode = mode
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Pre-process tabular features and baseline info
        self._process_tabular()
        self._add_baseline_info()

    def _process_tabular(self):
        # 1. Encode Sex (One-Hot)
        self.df["Sex_Male"] = (
            (self.df["Sex"] == "Male").astype(float)
            if "Sex" in self.df.columns
            else (self.df["Baseline_Sex"] == "Male").astype(float)
        )
        self.df["Sex_Female"] = (
            (self.df["Sex"] == "Female").astype(float)
            if "Sex" in self.df.columns
            else (self.df["Baseline_Sex"] == "Female").astype(float)
        )

        # 2. Encode SmokingStatus (One-Hot)
        col = (
            "SmokingStatus"
            if "SmokingStatus" in self.df.columns
            else "Baseline_SmokingStatus"
        )
        self.df["Smoke_Ex"] = (self.df[col] == "Ex-smoker").astype(float)
        self.df["Smoke_Never"] = (self.df[col] == "Never smoked").astype(float)
        self.df["Smoke_Current"] = (self.df[col] == "Currently smokes").astype(float)

        # 3. Normalize Age and Percent
        if "Age" in self.df.columns:
            self.df["Age_Norm"] = (self.df["Age"] - AGE_MEAN) / AGE_STD
            self.df["Percent_Norm"] = (self.df["Percent"] - PERCENT_MEAN) / PERCENT_STD
        else:
            self.df["Age_Norm"] = (self.df["Baseline_Age"] - AGE_MEAN) / AGE_STD
            self.df["Percent_Norm"] = (
                self.df["Baseline_Percent"] - PERCENT_MEAN
            ) / PERCENT_STD

    def _add_baseline_info(self):
        """
        Ensures every row has Baseline_FVC and Relative_Week.
        For Test set, these already exist.
        For Train/Val, we calculate them.
        """
        if self.mode == "test":
            # In test.csv, 'Predict_Week' is the target week, 'Baseline_Week' is the reference
            self.df["Relative_Week"] = (
                self.df["Predict_Week"] - self.df["Baseline_Week"]
            )
            # Baseline_FVC is already in the columns
        else:
            # For train/val, we define baseline as the measurement at the earliest week
            # Group by Patient to find baseline values
            baselines = self.df.sort_values("Weeks").groupby("Patient").first()

            patient_to_base_fvc = baselines["FVC"].to_dict()
            patient_to_base_week = baselines["Weeks"].to_dict()

            self.df["Baseline_FVC"] = self.df["Patient"].map(patient_to_base_fvc)
            self.df["Baseline_Week"] = self.df["Patient"].map(patient_to_base_week)
            self.df["Relative_Week"] = self.df["Weeks"] - self.df["Baseline_Week"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Load Images (Axial and Coronal)
        axial, coronal = get_patient_images(
            patient_id, dicom_dir, Config.CACHE_DIR, Config.LOAD_CACHED_DATA
        )

        # 2. Apply Augmentations
        if self.transform:
            # Apply transforms. Note: ToTensorV2 converts to tensor.
            # We apply independent spatial augmentation to orthogonal views
            # to act as regularization.
            aug_ax = self.transform(image=axial)["image"]
            aug_cor = self.transform(image=coronal)["image"]
        else:
            # Fallback if no transform provided (should not happen with get_transforms)
            aug_ax = torch.tensor(axial.transpose(2, 0, 1), dtype=torch.float32)
            aug_cor = torch.tensor(coronal.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Construct Tabular Vector (7 dims)
        # Order: Age, Percent, Sex(2), Smoke(3)
        tab_vec = np.array(
            [
                row["Age_Norm"],
                row["Percent_Norm"],
                row["Sex_Male"],
                row["Sex_Female"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
            ],
            dtype=np.float32,
        )

        # 4. Prepare Targets and Metadata
        if self.mode == "test":
            fvc = 0.0  # Dummy target
            patient_week = row["Patient_Week"]
        else:
            fvc = row["FVC"]
            # Construct Patient_Week ID for consistency/debugging
            patient_week = f"{patient_id}_{row['Weeks']}"

        return {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(fvc, dtype=torch.float32),
            "week": torch.tensor(row["Relative_Week"], dtype=torch.float32),
            "baseline_fvc": torch.tensor(row["Baseline_FVC"], dtype=torch.float32),
            "patient_week": patient_week,
        }
