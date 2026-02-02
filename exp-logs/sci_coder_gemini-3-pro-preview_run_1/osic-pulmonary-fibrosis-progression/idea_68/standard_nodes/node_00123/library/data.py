import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# ==========================================
# 1. Image Processing Functions
# ==========================================


def load_dicom_volume(dicom_dir):
    """
    Loads a DICOM volume from a directory, sorts by slice location,
    and converts to Hounsfield Units (HU).
    """
    if not os.path.exists(dicom_dir):
        return None

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            # Ensure we have image position for sorting
            if not hasattr(ds, "ImagePositionPatient"):
                # Fallback to InstanceNumber if position is missing
                if hasattr(ds, "InstanceNumber"):
                    pos = float(ds.InstanceNumber)
                else:
                    pos = 0
                slices.append((ds, pos))
            else:
                slices.append((ds, float(ds.ImagePositionPatient[2])))
        except:
            continue

    if not slices:
        return None

    # Sort by Z position
    slices.sort(key=lambda x: x[1])
    slices = [s[0] for s in slices]

    # Stack pixel arrays
    try:
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    except:
        return None

    # Convert to HU
    for i, s in enumerate(slices):
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        volume[i] = volume[i] * slope + intercept

    return volume


def process_volume_to_trislabs(volume, img_size=224):
    """
    Generates Fixed Overlapping Orthogonal Tri-Slabs (Axial and Coronal)
    from a 3D HU volume.

    Returns:
        axial_slabs: (224, 224, 3) uint8 array
        coronal_slabs: (224, 224, 3) uint8 array
    """
    # Lung Windowing [-1350, 150]
    # Standard lung window: W=1500, L=-600
    L, W = -600, 1500
    lower, upper = L - W // 2, L + W // 2

    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    volume = np.clip(volume, 0, 1)

    # Volume shape: (Z, Y, X)
    d, h, w = volume.shape

    # --- Axial Processing (Z-axis split) ---
    # Split Z into 3 overlapping slabs
    # Slab 1: 0 to 33% + overlap
    # Slab 2: 33% - overlap to 66% + overlap
    # Slab 3: 66% - overlap to 100%
    # We use roughly 40% chunk size per slab to ensure overlap

    axial_mips = []
    if d < 3:
        # Handle very shallow volumes by duplicating
        mip = np.max(volume, axis=0)
        axial_mips = [mip, mip, mip]
    else:
        # Define slab boundaries
        chunk = d / 3.0
        # Overlap padding (approx 15% of total depth / 3)
        pad = int(d * 0.05)

        starts = [0, int(chunk) - pad, int(2 * chunk) - pad]
        ends = [int(chunk) + pad, int(2 * chunk) + pad, d]

        # Clamp
        starts = [max(0, x) for x in starts]
        ends = [min(d, x) for x in ends]

        for s, e in zip(starts, ends):
            if e <= s:
                slab = volume[s : s + 1, :, :]
            else:
                slab = volume[s:e, :, :]

            # MIP along Z
            mip = np.max(slab, axis=0)  # Shape (Y, X)
            axial_mips.append(mip)

    # Resize and Stack Axial
    axial_resized = []
    for mip in axial_mips:
        res = cv2.resize(mip, (img_size, img_size), interpolation=cv2.INTER_AREA)
        axial_resized.append(res)

    axial_out = np.stack(axial_resized, axis=-1)  # (H, W, 3)
    axial_out = (axial_out * 255).astype(np.uint8)

    # --- Coronal Processing (Y-axis split) ---
    # Coronal view projects through Y (Anterior-Posterior), resulting in (Z, X) image
    # We split Y into 3 overlapping slabs

    coronal_mips = []
    if h < 3:
        mip = np.max(volume, axis=1)  # Max along Y -> (Z, X)
        coronal_mips = [mip, mip, mip]
    else:
        chunk = h / 3.0
        pad = int(h * 0.05)

        starts = [0, int(chunk) - pad, int(2 * chunk) - pad]
        ends = [int(chunk) + pad, int(2 * chunk) + pad, h]

        starts = [max(0, x) for x in starts]
        ends = [min(h, x) for x in ends]

        for s, e in zip(starts, ends):
            if e <= s:
                slab = volume[:, s : s + 1, :]
            else:
                slab = volume[:, s:e, :]

            # MIP along Y
            mip = np.max(slab, axis=1)  # Shape (Z, X)
            coronal_mips.append(mip)

    # Resize and Stack Coronal
    coronal_resized = []
    for mip in coronal_mips:
        # Resize (Z, X) to target size
        # Note: cv2.resize expects (width, height) -> (X, Z)
        res = cv2.resize(mip, (img_size, img_size), interpolation=cv2.INTER_AREA)
        coronal_resized.append(res)

    coronal_out = np.stack(coronal_resized, axis=-1)  # (H, W, 3)
    coronal_out = (coronal_out * 255).astype(np.uint8)

    return axial_out, coronal_out


def cache_patient_images(df, cache_dir, input_root):
    """
    Iterates through patients in dataframe, processes their DICOMs,
    and caches the Tri-Slab images as .npy files.
    """
    unique_patients = df["Patient"].unique()

    print(f"Checking cache for {len(unique_patients)} patients...")

    for patient in unique_patients:
        ax_path = os.path.join(cache_dir, f"{patient}_axial.npy")
        cor_path = os.path.join(cache_dir, f"{patient}_coronal.npy")

        if os.path.exists(ax_path) and os.path.exists(cor_path):
            continue

        # Get DICOM directory
        # Metadata contains relative path in 'dicom_dir'
        rel_path = df[df["Patient"] == patient]["dicom_dir"].iloc[0]
        full_path = os.path.join(input_root, rel_path)

        # Process
        volume = load_dicom_volume(full_path)

        if volume is None:
            # Create blank/zeros if failed (should not happen based on metadata check)
            axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            axial, coronal = process_volume_to_trislabs(volume, Config.IMG_SIZE)

        # Save
        np.save(ax_path, axial)
        np.save(cor_path, coronal)


# ==========================================
# 2. Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        """
        Args:
            df: DataFrame containing patient metadata and targets.
            cache_dir: Directory with cached .npy images.
            transform: Albumentations transforms.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Pre-compute tabular features normalization/mapping
        # Sex: Male=0, Female=1
        self.sex_map = {"Male": 0, "Female": 1}
        # Smoking: Ex-smoker=0, Never smoker=1, Currently smokes=2
        self.smoke_map = {"Ex-smoker": 0, "Never smoker": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient = row["Patient"]

        # 1. Load Images
        ax_path = os.path.join(self.cache_dir, f"{patient}_axial.npy")
        cor_path = os.path.join(self.cache_dir, f"{patient}_coronal.npy")

        try:
            img_ax = np.load(ax_path)
            img_cor = np.load(cor_path)
        except:
            # Fallback
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)

        # 2. Augmentations (Spatial Only)
        # Apply independent augmentations to views to encourage robustness
        if self.transform:
            res_ax = self.transform(image=img_ax)["image"]
            res_cor = self.transform(image=img_cor)["image"]

            # Albumentations ToTensorV2 converts to (C, H, W) and float [0,1] if normalized
            # But we want to handle normalization manually or via ToTensorV2
            # Here we assume ToTensorV2 is part of transform pipeline which normalizes to [0,1]
            img_ax = res_ax
            img_cor = res_cor
        else:
            # Convert to tensor (C, H, W) and normalize [0, 1]
            img_ax = torch.tensor(img_ax, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img_cor = (
                torch.tensor(img_cor, dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

        # 3. Tabular Features
        # Vector: [Age_norm, Percent_norm, Sex_M, Sex_F, Smoke_Ex, Smoke_Nv, Smoke_Cur]
        age = (row["Age"] - 50.0) / 50.0
        pct = row["Percent"] / 100.0

        sex = self.sex_map.get(row["Sex"], 0)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        # One-hot encoding
        sex_oh = [0, 0]
        sex_oh[sex] = 1

        smoke_oh = [0, 0, 0]
        smoke_oh[smoke] = 1

        tab_vec = np.array([age, pct] + sex_oh + smoke_oh, dtype=np.float32)

        # 4. Targets and Time
        # dt = Relative Week
        if self.mode == "test":
            # For test, 'Weeks' in df is the target prediction week
            # 'Baseline_Week' is the reference
            dt = float(row["Weeks"] - row["Baseline_Week"])
            base_fvc = float(row["Baseline_FVC"])
            fvc = 0.0  # Dummy
        else:
            # For train/val, 'Weeks' is already relative to baseline (usually)
            # However, we need to be careful. In train.csv, Weeks is relative to CT.
            # We computed Baseline_FVC as the FVC at Week ~ 0.
            # So dt is just 'Weeks'.
            dt = float(row["Weeks"])
            base_fvc = float(row["Baseline_FVC"])
            fvc = float(row["FVC"])

        return {
            "axial": img_ax,
            "coronal": img_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "dt": torch.tensor(dt, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
        }, torch.tensor(fvc, dtype=torch.float32)


# ==========================================
# 3. Data Loading Pipeline
# ==========================================


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders.
    Handles caching, splitting, and transforms.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Augment Train/Val with Baseline FVC
    # For each patient, find the FVC measurement closest to Week 0
    def add_baseline_fvc(df):
        # Create a mapping of Patient -> Baseline FVC
        # We select the row with min abs(Weeks) for each patient
        # Note: In train.csv, Weeks is relative to CT.
        baseline_map = {}
        for p in df["Patient"].unique():
            p_data = df[df["Patient"] == p]
            # Find index of week closest to 0
            idx = p_data["Weeks"].abs().idxmin()
            baseline_map[p] = p_data.loc[idx, "FVC"]

        df["Baseline_FVC"] = df["Patient"].map(baseline_map)
        return df

    train_df = add_baseline_fvc(train_df)
    val_df = add_baseline_fvc(val_df)

    # 3. Prepare Test DataFrame
    # Test DF columns: Patient_Week, FVC, Confidence, Patient, Predict_Week, Baseline_Week, Baseline_FVC, ...
    # We rename columns to match Train schema for the Dataset class
    test_df_processed = test_df.rename(
        columns={
            "Baseline_Age": "Age",
            "Baseline_Sex": "Sex",
            "Baseline_SmokingStatus": "SmokingStatus",
            "Baseline_Percent": "Percent",
            "Predict_Week": "Weeks",  # Target week
        }
    )

    # 4. Debugging: Subsample if configured
    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
    if Config.MAX_VAL_SAMPLES:
        val_df = val_df.iloc[: Config.MAX_VAL_SAMPLES]

    # 5. Caching
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    if not load_cached_data:
        # Force re-process
        pass  # Logic handled inside cache function (it checks existence, we could clear dir but safer to just overwrite or check)

    # Process images for all unique patients in all sets
    # Combine unique patients
    all_patients_df = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]].drop_duplicates(),
            val_df[["Patient", "dicom_dir"]].drop_duplicates(),
            test_df_processed[["Patient", "dicom_dir"]].drop_duplicates(),
        ]
    )

    # Run caching
    cache_patient_images(all_patients_df, Config.CACHE_DIR, Config.INPUT_ROOT)

    # 6. Transforms
    # Spatial-only augmentations, no intensity changes
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=Config.MEAN, std=Config.STD), ToTensorV2()]
    )

    # 7. Create Datasets
    train_dataset = OSICDataset(
        train_df, Config.CACHE_DIR, transform=train_transform, mode="train"
    )
    val_dataset = OSICDataset(
        val_df, Config.CACHE_DIR, transform=val_transform, mode="val"
    )
    test_dataset = OSICDataset(
        test_df_processed, Config.CACHE_DIR, transform=val_transform, mode="test"
    )

    # 8. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
