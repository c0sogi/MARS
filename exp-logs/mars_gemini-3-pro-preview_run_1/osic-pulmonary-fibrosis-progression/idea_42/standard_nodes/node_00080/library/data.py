import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import library.config as config
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    TRAIN_DICOM_DIR,
    TEST_DICOM_DIR,
    CACHE_DIR,
    IMAGE_SIZE,
    SEED,
)

# Attempt to import pydicom for DICOM handling
try:
    import pydicom
except ImportError:
    pydicom = None


def get_img_paths(dicom_dir):
    """Returns sorted list of .dcm files in a directory."""
    try:
        files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
        # Sort numerically if possible, else alphanumerically
        files.sort(
            key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else x
        )
        return [os.path.join(dicom_dir, f) for f in files]
    except FileNotFoundError:
        return []


def load_scan(paths):
    """Loads DICOM slices and sorts them by InstanceNumber or Z-position."""
    slices = []
    if not pydicom:
        # Fallback or error if pydicom is strictly missing in environment
        # Given the task involves DICOMs, we assume pydicom is available or we cannot proceed.
        print("Warning: pydicom not found. Returning empty slices.")
        return []

    for p in paths:
        try:
            s = pydicom.dcmread(p)
            slices.append(s)
        except Exception:
            continue

    if not slices:
        return []

    # Sort by InstanceNumber (preferred) or ImagePositionPatient Z
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            pass  # Keep filename sort order

    return slices


def get_pixels_hu(slices):
    """Converts DICOM slices to Hounsfield Units (HU)."""
    # Stack slices
    try:
        image = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    except Exception:
        return np.zeros((len(slices), 512, 512), dtype=np.float32)

    # Apply slope and intercept
    for i, s in enumerate(slices):
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)
        image[i] = slope * image[i] + intercept

    return image


def process_tri_slab(volume, axis=0):
    """
    Splits volume into 3 overlapping slabs along the specified axis,
    computes MIP (Max Intensity Projection), and stacks to RGB.

    Args:
        volume: 3D numpy array (D, H, W)
        axis: 0 for Axial (Depth), 1 for Coronal (Height)
    """
    # If Coronal, transpose to make the slicing axis the first dimension
    if axis == 1:
        # (D, H, W) -> (H, D, W)
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Define overlapping slab ranges
    # Slab 1: 0% - 38%
    # Slab 2: 31% - 69%
    # Slab 3: 62% - 100%
    p1_end = int(depth * 0.38)
    p2_start = int(depth * 0.31)
    p2_end = int(depth * 0.69)
    p3_start = int(depth * 0.62)

    # Handle very small volumes (fewer than 3 slices)
    if depth < 3:
        slab1 = volume
        slab2 = volume
        slab3 = volume
    else:
        slab1 = volume[: max(1, p1_end)]
        slab2 = volume[p2_start:p2_end]
        slab3 = volume[p3_start:]

    # Compute MIP
    # Use -2000 (air) as fallback for empty slabs
    m1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.full_like(volume[0], -2000)
    m2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.full_like(volume[0], -2000)
    m3 = np.max(slab3, axis=0) if slab3.shape[0] > 0 else np.full_like(volume[0], -2000)

    # Stack to 3 channels (H, W, 3)
    img = np.stack([m1, m2, m3], axis=-1)
    return img


def normalize_and_resize(img, size):
    """Applies Lung Window, normalizes to 0-1, and resizes."""
    # Lung Window: WL -600, WW 1500 -> Range [-1350, 150]
    L, W = -600, 1500
    min_hu = L - W // 2
    max_hu = L + W // 2

    img = (img - min_hu) / (max_hu - min_hu)
    img = np.clip(img, 0, 1)

    # Resize
    img = cv2.resize(img, (size, size))

    # Convert to uint8 (0-255) for efficient storage
    img = (img * 255).astype(np.uint8)
    return img


def get_tri_slab_input(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Generates or loads Axial and Coronal Tri-Slab images.
    Returns: (axial, coronal) as (224, 224, 3) uint8 arrays.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Cache
    if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # File corrupted, re-process

    # 2. Process from Scratch
    paths = get_img_paths(dicom_dir)

    # Handle missing data
    if not paths:
        black = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return black, black

    slices = load_scan(paths)
    if not slices:
        black = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return black, black

    vol_hu = get_pixels_hu(slices)

    # Generate Views
    axial = process_tri_slab(vol_hu, axis=0)
    axial = normalize_and_resize(axial, IMAGE_SIZE)

    coronal = process_tri_slab(vol_hu, axis=1)
    coronal = normalize_and_resize(coronal, IMAGE_SIZE)

    # Save to Cache
    np.save(axial_path, axial)
    np.save(coronal_path, coronal)

    return axial, coronal


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Normalization constants for tabular data (approximate dataset stats)
        self.age_mean = 67.0
        self.age_std = 15.0
        self.pct_mean = 77.0
        self.pct_std = 20.0

        # Encode Categorical Features
        # We use Baseline values if available (Train/Val), else direct values (Test)
        # Note: The DF passed here is expected to have 'Baseline_Sex' etc. for Train/Val

        if "Baseline_Sex" in self.df.columns:
            sex_col = "Baseline_Sex"
            smoke_col = "Baseline_SmokingStatus"
        else:
            # Fallback for Test if named differently, though get_dataloaders ensures consistency
            sex_col = "Baseline_Sex" if "Baseline_Sex" in self.df.columns else "Sex"
            smoke_col = (
                "Baseline_SmokingStatus"
                if "Baseline_SmokingStatus" in self.df.columns
                else "SmokingStatus"
            )

        self.df["Sex_enc"] = self.df[sex_col].map({"Male": 0, "Female": 1})
        self.df["Smoking_enc"] = (
            self.df[smoke_col]
            .map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})
            .fillna(0)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial + Coronal)
        # dicom_dir in metadata is relative (e.g., "train/ID..."). Prepend input root.
        dicom_dir = os.path.join("./input", row["dicom_dir"])
        axial, coronal = get_tri_slab_input(
            patient_id, dicom_dir, CACHE_DIR, load_cached=True
        )

        # 2. Apply Augmentations / Normalization
        if self.transform:
            # Albumentations
            res_ax = self.transform(image=axial)["image"]
            res_cor = self.transform(image=coronal)["image"]
            axial = res_ax
            coronal = res_cor
        else:
            # Manual ToTensor (CHW, 0-1 float)
            axial = torch.from_numpy(axial.transpose(2, 0, 1)).float() / 255.0
            coronal = torch.from_numpy(coronal.transpose(2, 0, 1)).float() / 255.0

        # 3. Prepare Tabular Features
        # Normalize Age and Percent
        age = (row["Baseline_Age"] - self.age_mean) / self.age_std
        pct = (row["Baseline_Percent"] - self.pct_mean) / self.pct_std
        sex = row["Sex_enc"]
        smoke = row["Smoking_enc"]

        tab_vec = torch.tensor([age, pct, sex, smoke], dtype=torch.float32)

        # 4. Prepare Time and Targets
        # We need (Weeks - Baseline_Weeks) for the anchored trajectory
        if self.mode in ["train", "val"]:
            base_week = row.get("Baseline_Weeks", 0)
            current_week = row["Weeks"]
            weeks_diff = float(current_week - base_week)

            base_fvc = float(row["Baseline_FVC"])
            target_fvc = float(row["FVC"])

            return (
                axial,
                coronal,
                tab_vec,
                torch.tensor(weeks_diff),
                torch.tensor(base_fvc),
                torch.tensor(target_fvc),
            )

        else:
            # Test Mode
            base_week = row["Baseline_Week"]
            current_week = row["Predict_Week"]
            weeks_diff = float(current_week - base_week)

            base_fvc = float(row["Baseline_FVC"])
            pat_week_id = row["Patient_Week"]

            return (
                axial,
                coronal,
                tab_vec,
                torch.tensor(weeks_diff),
                torch.tensor(base_fvc),
                pat_week_id,
            )


def get_transforms(mode="train"):
    """Returns Albumentations transforms."""
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


def get_dataloaders():
    """
    Prepares DataFrames and returns DataLoaders for Train, Val, and Test.
    Handles merging of baseline statistics for the training set.
    """
    # Load Metadata
    train_df = pd.read_csv(TRAIN_METADATA_PATH)
    val_df = pd.read_csv(VAL_METADATA_PATH)
    test_df = pd.read_csv(TEST_METADATA_PATH)

    if config.DEBUG:
        train_df = train_df.head(config.DEBUG_DATASET_SIZE)
        val_df = val_df.head(config.DEBUG_DATASET_SIZE)
        test_df = test_df.head(config.DEBUG_DATASET_SIZE)

    # Helper to merge baseline info onto history dataframe
    def add_baseline_info(df):
        # Identify baseline row: min(|Weeks|) per patient
        temp = df.copy()
        temp["week_abs"] = temp["Weeks"].abs()
        temp = temp.sort_values(["Patient", "week_abs"])

        # Take first row as baseline
        base = temp.groupby("Patient").first().reset_index()

        # Columns to extract as baseline features
        cols = ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus", "Weeks"]
        base = base[cols]
        base.columns = [
            "Patient",
            "Baseline_FVC",
            "Baseline_Percent",
            "Baseline_Age",
            "Baseline_Sex",
            "Baseline_SmokingStatus",
            "Baseline_Weeks",
        ]

        # Merge back
        merged = pd.merge(df, base, on="Patient", how="left")
        return merged

    # Process Train and Val
    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Test DF already has correct structure from metadata generation

    # Create Datasets
    train_ds = OSICDataset(train_df, mode="train", transform=get_transforms("train"))
    val_ds = OSICDataset(val_df, mode="val", transform=get_transforms("val"))
    test_ds = OSICDataset(test_df, mode="test", transform=get_transforms("val"))

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
