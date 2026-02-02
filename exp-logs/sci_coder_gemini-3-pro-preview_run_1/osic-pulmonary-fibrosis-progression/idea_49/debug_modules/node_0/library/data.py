import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Try importing pydicom, handle if missing
try:
    import pydicom
except ImportError:
    pydicom = None
    print("Warning: pydicom not found. DICOM processing will fail.")

# ==========================================
# Constants & Statistics
# ==========================================
# Normalization constants based on training set statistics (from EDA)
STATS = {
    "Age_mean": 67.58,
    "Age_std": 6.63,
    "Percent_mean": 76.91,
    "Percent_std": 19.20,
    "FVC_mean": 2654.65,
    "FVC_std": 801.70,
}


# ==========================================
# Image Processing Functions
# ==========================================
def get_img_tri_slab(volume, axis=0):
    """
    Generates a Tri-Slab RGB image from a 3D volume using MIP with overlap.
    axis=0: Axial (split Z)
    axis=1: Coronal (split Y)
    """
    if axis == 1:
        # Transpose to (Y, Z, X) so Y becomes the depth dimension
        volume = volume.transpose(1, 0, 2)

    depth = volume.shape[0]
    if depth == 0:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Define slab boundaries with overlap
    # We want 3 slabs covering the depth: 0-33%, 33-66%, 66-100%
    p1 = int(depth * 0.33)
    p2 = int(depth * 0.66)
    overlap_px = int(depth * Config.SLAB_OVERLAP * 0.5)

    # Define ranges with overlap
    r1_start, r1_end = 0, min(depth, p1 + overlap_px)
    r2_start, r2_end = max(0, p1 - overlap_px), min(depth, p2 + overlap_px)
    r3_start, r3_end = max(0, p2 - overlap_px), depth

    # Helper for MIP
    def get_mip(start, end):
        if end <= start:
            return np.zeros_like(volume[0])
        slab = volume[start:end]
        return np.max(slab, axis=0)

    c1 = get_mip(r1_start, r1_end)
    c2 = get_mip(r2_start, r2_end)
    c3 = get_mip(r3_start, r3_end)

    # Stack to RGB (H, W, 3)
    img = np.stack([c1, c2, c3], axis=-1)

    # Resize to target resolution
    img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return img


def process_patient_images(patient_id, dicom_dir, load_cached_data=True):
    """
    Loads DICOMs, generates Axial and Coronal Tri-Slabs, and handles caching.
    """
    # Define cache paths
    cache_path_ax = Config.get_cache_path(patient_id, "axial")
    cache_path_cor = Config.get_cache_path(patient_id, "coronal")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(cache_path_ax)
        and os.path.exists(cache_path_cor)
    ):
        try:
            img_ax = np.load(cache_path_ax)
            img_cor = np.load(cache_path_cor)
            return img_ax, img_cor
        except Exception as e:
            print(f"Error loading cache for {patient_id}: {e}. Reprocessing.")

    # 2. Process from scratch
    full_dicom_path = os.path.join(Config.INPUT_ROOT, dicom_dir)

    # List all dcm files
    files = glob.glob(os.path.join(full_dicom_path, "*.dcm"))

    if not files or pydicom is None:
        # Return blank images if no files or no pydicom
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return blank, blank

    # Read DICOMs
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except:
            continue

    if not slices:
        blank = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return blank, blank

    # Sort by ImagePositionPatient Z (index 2) or InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    # Create 3D volume and Convert to Hounsfield Units
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        img = img * slope + intercept
        images.append(img)

    volume = np.stack(images)  # (Z, Y, X)

    # Normalize Volume (Lung Window: -1000 to 400)
    MIN_HU = -1000.0
    MAX_HU = 400.0
    volume = np.clip(volume, MIN_HU, MAX_HU)
    volume = (volume - MIN_HU) / (MAX_HU - MIN_HU)

    # Generate Tri-Slabs
    img_ax = get_img_tri_slab(volume, axis=0)  # Axial
    img_cor = get_img_tri_slab(volume, axis=1)  # Coronal

    # 3. Save to cache
    try:
        np.save(cache_path_ax, img_ax)
        np.save(cache_path_cor, img_cor)
    except Exception as e:
        print(f"Failed to save cache for {patient_id}: {e}")

    return img_ax, img_cor


# ==========================================
# Dataset Class
# ==========================================
class OSICDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df: DataFrame containing patient data.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transform.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial & Coronal)
        dicom_dir = row["dicom_dir"]
        img_ax, img_cor = process_patient_images(
            patient_id, dicom_dir, load_cached_data=True
        )

        # 2. Apply Augmentations
        if self.transform:
            # Apply transform to both images
            res_ax = self.transform(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor = res_cor["image"]
        else:
            t = ToTensorV2()
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # 3. Tabular Features
        # Normalize inputs using fixed stats
        sex = 0.0 if row["Baseline_Sex"] == "Male" else 1.0

        # One-hot encoding for Smoking
        smk_ex = 1.0 if row["Baseline_SmokingStatus"] == "Ex-smoker" else 0.0
        smk_nv = 1.0 if row["Baseline_SmokingStatus"] == "Never smoked" else 0.0
        smk_cr = 1.0 if row["Baseline_SmokingStatus"] == "Currently smokes" else 0.0

        # Numerical normalization
        age_norm = (row["Baseline_Age"] - STATS["Age_mean"]) / STATS["Age_std"]
        pct_norm = (row["Baseline_Percent"] - STATS["Percent_mean"]) / STATS[
            "Percent_std"
        ]
        fvc_norm = (row["Baseline_FVC"] - STATS["FVC_mean"]) / STATS["FVC_std"]

        tabular = np.array(
            [age_norm, sex, smk_ex, smk_nv, smk_cr, pct_norm, fvc_norm],
            dtype=np.float32,
        )

        # 4. Target & Meta
        if self.mode == "test":
            current_week = row["Predict_Week"]
            base_week = row["Baseline_Week"]
            target = 0.0
        else:
            current_week = row["Weeks"]
            base_week = row["Baseline_Week"]
            target = row["FVC"]

        time_delta = float(current_week - base_week)

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "time_delta": torch.tensor(time_delta, dtype=torch.float32),
            "patient_id": patient_id,
            "week": current_week,
            "baseline_fvc": row["Baseline_FVC"],
        }


# ==========================================
# Data Loading & Preparation
# ==========================================
def prepare_train_dataframe(df):
    """
    Enhances training dataframe by identifying and merging baseline features.
    Baseline is defined as the visit closest to Week 0.
    """
    # Calculate distance to 0 to find baseline visit
    df["dist_to_0"] = df["Weeks"].abs()

    # Sort by patient and distance
    df_sorted = df.sort_values(["Patient", "dist_to_0"])

    # Take first entry per patient as baseline
    baseline_df = df_sorted.groupby("Patient").first().reset_index()

    # Keep relevant columns and rename
    cols_to_keep = ["Patient", "Weeks", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
    baseline_df = baseline_df[cols_to_keep]

    rename_map = {
        "Weeks": "Baseline_Week",
        "FVC": "Baseline_FVC",
        "Percent": "Baseline_Percent",
        "Age": "Baseline_Age",
        "Sex": "Baseline_Sex",
        "SmokingStatus": "Baseline_SmokingStatus",
    }
    baseline_df = baseline_df.rename(columns=rename_map)

    # Merge baseline info back to original df
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    return merged_df


def get_dataloaders(debug=False, batch_size=32, num_workers=4):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare DataFrames (Add Baseline info to Train/Val)
    train_df = prepare_train_dataframe(train_df)
    val_df = prepare_train_dataframe(val_df)
    # test_df already has Baseline columns from metadata generation step

    # 3. Debug Mode
    if debug or Config.DEBUG:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 4. Augmentations (Spatial only)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Resize(Config.IMG_SIZE, Config.IMG_SIZE), ToTensorV2()]
    )

    # 5. Datasets
    train_dataset = OSICDataset(train_df, mode="train", transform=train_transform)
    val_dataset = OSICDataset(val_df, mode="val", transform=val_transform)
    test_dataset = OSICDataset(test_df, mode="test", transform=val_transform)

    # 6. Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
