import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# ==========================================
# 1. Helper Functions & Image Processing
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by instance number,
    and returns a list of pydicom datasets.
    """
    if not os.path.exists(path):
        return []

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return []

    # Sort by integer value of filename (e.g., 1.dcm, 2.dcm, 10.dcm)
    # This assumes filename corresponds to instance number/slice position
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            slices.append(ds)
        except Exception:
            continue

    return slices


def get_pixels_hu(slices):
    """
    Converts a list of dicom slices to a numpy array of Hounsfield Units (HU).
    Handles intercept and slope scaling.
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    # The intercept and slope are usually constant per scan, but we check first slice
    intercept = (
        slices[0].RescaleIntercept if hasattr(slices[0], "RescaleIntercept") else -1024
    )
    slope = slices[0].RescaleSlope if hasattr(slices[0], "RescaleSlope") else 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def lung_window_and_normalize(image, level=-600, width=1500):
    """
    Applies lung windowing and normalizes to [0, 1].
    """
    lower = level - width // 2
    upper = level + width // 2

    img_windowed = np.clip(image, lower, upper)
    img_norm = (img_windowed - lower) / (upper - lower)
    return img_norm


def generate_tri_slab(volume, axis=0, num_slabs=3, overlap=0.15, target_size=224):
    """
    Generates 3 overlapping slabs along the specified axis using MIP.

    Args:
        volume: 3D numpy array (Z, Y, X) normalized to [0, 1]
        axis: 0 for Axial (Z-axis split), 1 for Coronal (Y-axis split)
        num_slabs: Number of slabs (channels)
        overlap: Fraction of overlap between slabs
        target_size: Output spatial resolution

    Returns:
        (3, H, W) numpy array
    """
    # If axis is 1 (Coronal), we permute so the split axis is 0
    if axis == 1:
        # (Z, Y, X) -> (Y, Z, X)
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # If depth is too small, just repeat the volume
    if depth < num_slabs:
        # Create 3 identical channels from the max projection of the whole volume
        mip = np.max(volume, axis=0)
        img = np.stack([mip] * num_slabs, axis=0)
    else:
        # Calculate slab parameters
        chunk_size = depth / num_slabs
        overlap_size = int(chunk_size * overlap)

        slabs = []
        for i in range(num_slabs):
            start = max(0, int(i * chunk_size) - overlap_size)
            end = min(depth, int((i + 1) * chunk_size) + overlap_size)

            # Extract slab and compute MIP
            slab_vol = volume[start:end, :, :]
            if slab_vol.shape[0] == 0:
                # Fallback for edge cases
                mip = np.zeros((volume.shape[1], volume.shape[2]))
            else:
                mip = np.max(slab_vol, axis=0)
            slabs.append(mip)

        img = np.stack(slabs, axis=0)  # (3, H, W)

    # Resize spatial dimensions to target_size
    # img is (C, H, W). cv2.resize expects (W, H) and processes channels if interleaved
    # Transpose to (H, W, C) for resizing
    img_t = np.transpose(img, (1, 2, 0))
    img_resized = cv2.resize(
        img_t, (target_size, target_size), interpolation=cv2.INTER_AREA
    )

    # Transpose back to (C, H, W)
    if len(img_resized.shape) == 2:  # If single channel (shouldn't happen with 3 slabs)
        img_final = np.stack([img_resized] * num_slabs, axis=0)
    else:
        img_final = np.transpose(img_resized, (2, 0, 1))

    return img_final.astype(np.float32)


def process_patient_scan(patient_id, dicom_dir, cache_dir, load_cache=True):
    """
    Loads DICOM, processes into Axial and Coronal tri-slabs, and caches result.
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    if load_cache and os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True).item()
        except Exception:
            pass  # Failed to load, recompute

    # Compute from scratch
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
    scans = load_scan(full_path)

    if not scans:
        # Return zeros if no data found
        dummy = np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
        return {"axial": dummy, "coronal": dummy}

    vol_hu = get_pixels_hu(scans)
    vol_norm = lung_window_and_normalize(vol_hu)

    # Generate Views
    axial = generate_tri_slab(
        vol_norm,
        axis=0,
        num_slabs=Config.NUM_SLABS,
        overlap=Config.SLAB_OVERLAP,
        target_size=Config.IMG_SIZE,
    )

    coronal = generate_tri_slab(
        vol_norm,
        axis=1,
        num_slabs=Config.NUM_SLABS,
        overlap=Config.SLAB_OVERLAP,
        target_size=Config.IMG_SIZE,
    )

    data = {"axial": axial, "coronal": coronal}

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, data)

    return data


# ==========================================
# 2. Tabular Preprocessing
# ==========================================


class TabularPreprocessor:
    def __init__(self):
        self.pipeline = ColumnTransformer(
            [
                ("num", StandardScaler(), ["Age", "Percent"]),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["Sex", "SmokingStatus"],
                ),
            ]
        )

    def fit(self, df):
        self.pipeline.fit(df)

    def transform(self, df):
        return self.pipeline.transform(df)


# ==========================================
# 3. Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(self, df, tabular_data, cache_dir, mode="train", transform=None):
        """
        Args:
            df: DataFrame containing patient metadata and targets.
            tabular_data: Preprocessed tabular features (numpy array).
            cache_dir: Directory to store/load processed images.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transform.
        """
        self.df = df.reset_index(drop=True)
        self.tabular_data = tabular_data
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        # Use relative path from metadata
        dicom_dir = row["dicom_dir"]
        img_data = process_patient_scan(
            patient_id, dicom_dir, self.cache_dir, load_cache=True
        )

        axial = img_data["axial"]  # (3, 224, 224)
        coronal = img_data["coronal"]  # (3, 224, 224)

        # 2. Apply Augmentations (Spatial only)
        # Albumentations expects (H, W, C), so we transpose
        if self.transform:
            # Transpose to (H, W, C)
            ax_t = np.transpose(axial, (1, 2, 0))
            co_t = np.transpose(coronal, (1, 2, 0))

            # Apply same transform? Or independent?
            # Independent is better for robustness as views are orthogonal
            aug_ax = self.transform(image=ax_t)["image"]
            aug_co = self.transform(image=co_t)["image"]

            # Transpose back to (C, H, W) is handled by ToTensorV2 usually,
            # but here we might be using custom pipeline or just numpy.
            # If using ToTensorV2, result is Tensor (C, H, W).
            # If just numpy transforms, result is (H, W, C).

            # Let's assume transform includes ToTensorV2
            axial = aug_ax
            coronal = aug_co
        else:
            # Convert to tensor
            axial = torch.tensor(axial, dtype=torch.float32)
            coronal = torch.tensor(coronal, dtype=torch.float32)

        # Ensure Channel-First format (C, H, W)
        # If output is (H, W, C), permute it.
        if isinstance(axial, np.ndarray):
            axial = torch.from_numpy(axial).float()
        if isinstance(coronal, np.ndarray):
            coronal = torch.from_numpy(coronal).float()

        if axial.ndim == 3 and axial.shape[2] == 3:
            axial = axial.permute(2, 0, 1)
        if coronal.ndim == 3 and coronal.shape[2] == 3:
            coronal = coronal.permute(2, 0, 1)

        # 3. Tabular & Target
        tab_vec = torch.tensor(self.tabular_data[idx], dtype=torch.float32)

        # Determine Target and Time Delta
        if self.mode in ["train", "val"]:
            target = float(row["FVC"])
            # For training, we need baseline FVC.
            # In this implementation, we assume the dataframe has 'Baseline_FVC' column
            # populated during preprocessing.
            baseline_fvc = float(row["Baseline_FVC"])
            time_delta = float(row["Weeks"] - row["Baseline_Week"])
        else:
            # Test mode
            target = 0.0  # Dummy
            baseline_fvc = float(row["Baseline_FVC"])
            time_delta = float(row["Predict_Week"] - row["Baseline_Week"])

        return {
            "axial": axial,
            "coronal": coronal,
            "tabular": tab_vec,
            "target": torch.tensor(target, dtype=torch.float32),
            "time_delta": torch.tensor(time_delta, dtype=torch.float32),
            "baseline_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "patient_id": patient_id,
        }


# ==========================================
# 4. Main Data Pipeline
# ==========================================


def prepare_dataframe(df, is_train=True):
    """
    Enriches the dataframe with Baseline information.
    For training/val, finds the baseline visit (Week ~0) and merges it.
    """
    if not is_train:
        # Test dataframe already has Baseline_ columns from metadata script
        return df

    # For train/val, we need to identify baseline for each patient
    # We assume the visit with min absolute weeks is baseline
    df["Abs_Weeks"] = df["Weeks"].abs()

    # Find index of baseline rows
    baseline_indices = df.groupby("Patient")["Abs_Weeks"].idxmin()
    baseline_df = df.loc[baseline_indices].copy()

    # Select relevant columns to merge back
    cols_to_merge = [
        "Patient",
        "FVC",
        "Percent",
        "Age",
        "Sex",
        "SmokingStatus",
        "Weeks",
    ]
    baseline_df = baseline_df[cols_to_merge]

    # Rename to Baseline_
    baseline_df.columns = [
        "Patient",
        "Baseline_FVC",
        "Baseline_Percent",
        "Baseline_Age",
        "Baseline_Sex",
        "Baseline_SmokingStatus",
        "Baseline_Week",
    ]

    # Merge back to original df
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    # Drop helper
    merged_df.drop(columns=["Abs_Weeks"], inplace=True)

    return merged_df


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Val, and Test.
    Handles caching, preprocessing, and augmentations.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Enrich DataFrames with Baseline Info
    train_df = prepare_dataframe(train_df, is_train=True)
    val_df = prepare_dataframe(val_df, is_train=True)
    # Test df is already prepared by metadata script

    # 3. Fit Tabular Preprocessor
    # We fit on Training data only (FULL dataset to capture all categories)
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    if debug:
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        test_df = test_df.head(Config.DEBUG_SIZE)

    # Transform all sets
    train_tab = preprocessor.transform(train_df)
    val_tab = preprocessor.transform(val_df)

    # For test, we must map Baseline columns to the expected feature names
    # The preprocessor expects: Age, Percent, Sex, SmokingStatus
    # In test_df, these are Baseline_Age, Baseline_Percent, Baseline_Sex, Baseline_SmokingStatus
    test_tab_input = test_df.rename(
        columns={
            "Baseline_Age": "Age",
            "Baseline_Percent": "Percent",
            "Baseline_Sex": "Sex",
            "Baseline_SmokingStatus": "SmokingStatus",
        }
    )
    test_tab = preprocessor.transform(test_tab_input)

    # 4. Define Augmentations
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 5. Create Cache Directory
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 6. Create Datasets
    train_ds = LungDataset(
        train_df, train_tab, cache_dir, mode="train", transform=train_transform
    )
    val_ds = LungDataset(
        val_df, val_tab, cache_dir, mode="val", transform=val_transform
    )
    test_ds = LungDataset(
        test_df, test_tab, cache_dir, mode="test", transform=val_transform
    )

    # 7. Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
