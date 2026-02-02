import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# -----------------------------------------------------------------------------
# Helper Functions for Image Processing
# -----------------------------------------------------------------------------


def load_scans(dcm_path):
    """
    Loads DICOM files from a directory and sorts them by InstanceNumber.
    """
    if not os.path.exists(dcm_path):
        return []

    files = [
        os.path.join(dcm_path, f) for f in os.listdir(dcm_path) if f.endswith(".dcm")
    ]
    if not files:
        return []

    scans = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            scans.append(ds)
        except:
            continue

    # Sort by InstanceNumber (Z-position)
    scans.sort(
        key=lambda x: int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
    )
    return scans


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel_array to Hounsfield Units.
    """
    if not scans:
        return np.zeros((1, 224, 224), dtype=np.int16)

    image = np.stack([s.pixel_array for s in scans]).astype(np.float32)

    # Convert to HU
    intercept = (
        scans[0].RescaleIntercept if hasattr(scans[0], "RescaleIntercept") else -1024
    )
    slope = scans[0].RescaleSlope if hasattr(scans[0], "RescaleSlope") else 1

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.float32)

    image += np.float32(intercept)

    # Handle padding (some scanners use -2000 or similar for outside)
    image[image < -1024] = -1024
    return image


def generate_tri_slab(volume, axis_idx, overlap=0.15):
    """
    Generates a 3-channel image using overlapping slabs along the specified axis.
    MIP (Maximum Intensity Projection) is applied to each slab.

    Args:
        volume: 3D numpy array (Z, Y, X)
        axis_idx: 0 for Axial (Z), 1 for Coronal (Y)
        overlap: Fraction of overlap
    """
    # Ensure volume is at least 3D
    if volume.ndim != 3:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    # Move the target axis to the front (0) so we can slice uniformly
    # If axis is 0 (Z), shape is (D, H, W). If axis is 1 (Y), we transpose to (H, D, W)
    if axis_idx == 1:
        # Transpose to make Y the primary axis: (Y, Z, X)
        vol_aligned = np.transpose(volume, (1, 0, 2))
    else:
        vol_aligned = volume

    depth = vol_aligned.shape[0]

    if depth < 3:
        # Fallback for extremely thin volumes: replicate
        mip = np.max(vol_aligned, axis=0)
        return np.stack([mip, mip, mip], axis=-1)

    # Define slab boundaries
    # We want 3 slabs covering the range [0, depth]
    # Slab size approx depth / 3
    slab_size = depth / 3.0
    ov_pixels = int(slab_size * overlap)

    # Calculate indices
    idx1 = int(slab_size) + ov_pixels
    idx2_start = int(slab_size) - ov_pixels
    idx2_end = int(2 * slab_size) + ov_pixels
    idx3_start = int(2 * slab_size) - ov_pixels

    # Clamp indices
    idx1 = min(idx1, depth)
    idx2_start = max(0, idx2_start)
    idx2_end = min(idx2_end, depth)
    idx3_start = max(0, idx3_start)

    # Extract slabs
    slab1 = vol_aligned[0:idx1, :, :]
    slab2 = vol_aligned[idx2_start:idx2_end, :, :]
    slab3 = vol_aligned[idx3_start:, :, :]

    # Compute MIP
    # Handle empty slabs (safety check)
    m1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.zeros_like(vol_aligned[0])
    m2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.zeros_like(vol_aligned[0])
    m3 = np.max(slab3, axis=0) if slab3.shape[0] > 0 else np.zeros_like(vol_aligned[0])

    # Stack channels
    img = np.stack([m1, m2, m3], axis=-1)
    return img


def normalize_and_resize(img, size=224):
    """
    Applies lung windowing, normalization to 0-255, and resizing.
    """
    # Lung Window: Level -600, Width 1500 -> Range [-1350, 150]
    # We use a standard range for lung parenchyma
    L, W = -600, 1500
    lower = L - W // 2
    upper = L + W // 2

    img = np.clip(img, lower, upper)

    # Min-Max normalize to 0-255
    img = (img - lower) / (upper - lower)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Resize
    # Note: img is (H, W, 3)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img


def process_dicom(patient_id, dicom_dir, cache_dir, size=224, load_cached=True):
    """
    Handles the full image pipeline: Check Cache -> Load DICOM -> Process -> Save Cache.
    Returns (axial_img, coronal_img).
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try Load Cache
    if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            ax = np.load(axial_path)
            cor = np.load(coronal_path)
            return ax, cor
        except Exception as e:
            pass  # Fall through to re-processing

    # 2. Process from Scratch
    # Construct full path to DICOMs
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)

    scans = load_scans(full_path)
    if not scans:
        # Fallback: Black images
        ax = np.zeros((size, size, 3), dtype=np.uint8)
        cor = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        vol = get_pixels_hu(scans)

        # Axial (Z-axis is 0)
        ax_raw = generate_tri_slab(vol, axis_idx=0, overlap=Config.SLAB_OVERLAP)
        ax = normalize_and_resize(ax_raw, size=size)

        # Coronal (Y-axis is 1)
        cor_raw = generate_tri_slab(vol, axis_idx=1, overlap=Config.SLAB_OVERLAP)
        cor = normalize_and_resize(cor_raw, size=size)

    # 3. Save Cache
    try:
        np.save(axial_path, ax)
        np.save(coronal_path, cor)
    except Exception:
        pass

    return ax, cor


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, cache=True):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self.cache_dir = Config.CACHE_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-process Tabular Data
        self._prepare_tabular()

    def _prepare_tabular(self):
        # We need to normalize Age, Percent and Encode Sex, SmokingStatus
        # For training, we use Baseline values.

        # Check if Baseline columns exist (Test set structure)
        if "Baseline_Age" in self.df.columns:
            # Test set logic
            self.ages = self.df["Baseline_Age"].values
            self.percents = self.df["Baseline_Percent"].values
            self.sexes = self.df["Baseline_Sex"].values
            self.smokings = self.df["Baseline_SmokingStatus"].values

            # For test, 'Predict_Week' is the target time, 'Baseline_Week' is reference
            self.relative_weeks = (
                self.df["Predict_Week"] - self.df["Baseline_Week"]
            ).values
            self.targets = self.df["FVC"].values  # Placeholder
            self.base_fvcs = self.df["Baseline_FVC"].values

        else:
            # Train/Val logic: Find baseline for each patient
            patient_groups = self.df.groupby("Patient")

            self.ages = np.zeros(len(self.df), dtype=np.float32)
            self.percents = np.zeros(len(self.df), dtype=np.float32)
            self.sexes = np.empty(len(self.df), dtype=object)
            self.smokings = np.empty(len(self.df), dtype=object)
            self.base_fvcs = np.zeros(len(self.df), dtype=np.float32)

            # Pre-calculate baseline info for all patients
            baseline_map = {}
            for pid, group in patient_groups:
                # Find row with min weeks (closest to baseline)
                baseline_idx = group["Weeks"].idxmin()
                baseline_row = group.loc[baseline_idx]

                baseline_map[pid] = {
                    "Age": baseline_row["Age"],
                    "Percent": baseline_row["Percent"],
                    "Sex": baseline_row["Sex"],
                    "SmokingStatus": baseline_row["SmokingStatus"],
                    "FVC": baseline_row["FVC"],
                    "Weeks": baseline_row["Weeks"],
                }

            # Map back to dataframe
            for i, row in self.df.iterrows():
                pid = row["Patient"]
                base = baseline_map[pid]
                self.ages[i] = base["Age"]
                self.percents[i] = base["Percent"]
                self.sexes[i] = base["Sex"]
                self.smokings[i] = base["SmokingStatus"]
                self.base_fvcs[i] = base["FVC"]

            # Relative week = Current Week - Baseline Week
            self.relative_weeks = (
                self.df["Weeks"].values
                - [baseline_map[p]["Weeks"] for p in self.df["Patient"]]
            ).astype(np.float32)
            self.targets = self.df["FVC"].values

        # Normalize/Encode
        # Age: Scale (x - 50) / 50 approx
        self.ages = (self.ages - 50.0) / 50.0

        # Percent: Scale / 100
        self.percents = self.percents / 100.0

        # Sex: Male=0, Female=1
        self.sex_enc = np.array(
            [1 if s == "Female" else 0 for s in self.sexes], dtype=np.float32
        )

        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        smk_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}
        self.smoking_enc = np.array(
            [smk_map.get(s, 0) for s in self.smokings], dtype=np.int64
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]
        dicom_dir = row["dicom_dir"]

        # 1. Load Images
        img_ax, img_cor = process_dicom(
            pid, dicom_dir, self.cache_dir, size=Config.IMG_SIZE, load_cached=self.cache
        )

        # 2. Augmentation
        if self.transform:
            res_ax = self.transform(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor = res_cor["image"]

        # 3. Tabular Feature Construction
        # Smoking One-Hot
        smk_oh = np.zeros(3, dtype=np.float32)
        smk_oh[self.smoking_enc[idx]] = 1.0

        # Tabular Vector: [Age, Sex, Percent, Smoke_Ex, Smoke_Never, Smoke_Current]
        tab_vec = np.concatenate(
            [[self.ages[idx]], [self.sex_enc[idx]], [self.percents[idx]], smk_oh]
        ).astype(np.float32)

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "weeks": torch.tensor(self.relative_weeks[idx], dtype=torch.float32),
            "base_fvc": torch.tensor(self.base_fvcs[idx], dtype=torch.float32),
            "patient_id": pid,
        }


# -----------------------------------------------------------------------------
# Data Loader Setup
# -----------------------------------------------------------------------------


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [A.Normalize(mean=Config.NORM_MEAN, std=Config.NORM_STD), ToTensorV2()]
        )


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_patients = train_df["Patient"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        val_patients = val_df["Patient"].unique()[:5]
        test_patients = test_df["Patient"].unique()[:5]

        train_df = train_df[train_df["Patient"].isin(train_patients)]
        val_df = val_df[val_df["Patient"].isin(val_patients)]
        test_df = test_df[test_df["Patient"].isin(test_patients)]

        print(
            f"DEBUG MODE: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

    # Datasets
    train_ds = OSICDataset(train_df, mode="train", transform=get_transforms("train"))
    val_ds = OSICDataset(val_df, mode="val", transform=get_transforms("val"))
    test_ds = OSICDataset(test_df, mode="test", transform=get_transforms("test"))

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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
