import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import pydicom for DICOM handling
try:
    import pydicom
except ImportError:
    # Fallback or error handling if strictly not available,
    # though standard for this task type.
    pydicom = None

from library.config import Config
from library.utils import Logger

# ==========================================
# Helper Functions
# ==========================================


def load_dicom_scan(path):
    """
    Loads a DICOM scan from a directory, sorts by InstanceNumber,
    and converts to a Hounsfield Unit (HU) numpy array.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files.")

    if not os.path.exists(path):
        return None

    # List all dcm files
    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return None

    # Read and sort
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except:
            continue

    if not slices:
        return None

    # Sort by ImagePositionPatient Z or InstanceNumber
    # InstanceNumber is usually reliable for this dataset
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except:
            pass  # Keep file order if sorting fails

    # Extract pixels and convert to HU
    image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Convert to HU
    # slope * pixel + intercept
    for i, s in enumerate(slices):
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        image[i] = image[i] * slope + intercept

    return image


def generate_tri_slab_mip(volume, orientation="axial"):
    """
    Generates a 3-channel image using Fixed Overlapping Orthogonal Tri-Slabs.

    Args:
        volume (np.ndarray): 3D HU array (D, H, W).
        orientation (str): 'axial' or 'coronal'.

    Returns:
        np.ndarray: (224, 224, 3) normalized RGB-like image.
    """
    # 1. Standardize Orientation
    if orientation == "coronal":
        # Axial is (Z, Y, X). Coronal view requires looking from Y.
        # Transpose to (Y, Z, X) -> (Depth, Height, Width) for the slab logic
        vol = volume.transpose(1, 0, 2)
    else:
        vol = volume

    depth = vol.shape[0]

    # Handle small volumes by repeating
    if depth < 3:
        vol = np.repeat(vol, 3, axis=0)
        depth = vol.shape[0]

    # 2. Define Slab Boundaries with Overlap
    # We want 3 slabs covering [0, 1] range.
    # Base splits: 0-0.33, 0.33-0.66, 0.66-1.0
    # Overlap 15% of a slab's width. Slab width is D/3.
    # Overlap pixels = 0.15 * (D/3) = 0.05 * D

    overlap = int(0.05 * depth)
    p1 = int(depth / 3)
    p2 = int(2 * depth / 3)

    # Define ranges
    # Slab 1: 0 to p1 + overlap
    s1_start, s1_end = 0, min(depth, p1 + overlap)

    # Slab 2: p1 - overlap to p2 + overlap
    s2_start, s2_end = max(0, p1 - overlap), min(depth, p2 + overlap)

    # Slab 3: p2 - overlap to end
    s3_start, s3_end = max(0, p2 - overlap), depth

    # 3. Compute MIPs
    # Lung Windowing before MIP?
    # Standard lung window: W=1500, L=-600 -> [-1350, 150].
    # However, for fibrosis, we want to capture high density.
    # Let's clip to a reasonable range [-1000, 400] then normalize.

    vol_clipped = np.clip(vol, -1000, 400)
    # Normalize to 0-1
    vol_norm = (vol_clipped + 1000) / 1400.0

    # Extract slabs
    slab1 = vol_norm[s1_start:s1_end, :, :]
    slab2 = vol_norm[s2_start:s2_end, :, :]
    slab3 = vol_norm[s3_start:s3_end, :, :]

    # MIP (Max Intensity Projection) along the depth axis of the slab
    # If a slab is empty (shouldn't happen), use zeros
    mip1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.zeros(vol_norm.shape[1:])
    mip2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.zeros(vol_norm.shape[1:])
    mip3 = np.max(slab3, axis=0) if slab3.shape[0] > 0 else np.zeros(vol_norm.shape[1:])

    # Stack to 3 channels
    img = np.stack([mip1, mip2, mip3], axis=-1)  # (H, W, 3)

    # 4. Resize
    img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

    return img_resized


def process_patient_images(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Loads or processes patient images (Axial and Coronal).
    Uses caching to avoid re-processing.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # Try loading from cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            img_axial = np.load(axial_path)
            img_coronal = np.load(coronal_path)
            return img_axial, img_coronal
        except:
            pass  # Corrupt file, re-process

    # Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
    volume = load_dicom_scan(full_path)

    if volume is None:
        # Fallback for missing data: create black image
        img_axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        img_coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
    else:
        img_axial = generate_tri_slab_mip(volume, orientation="axial")
        img_coronal = generate_tri_slab_mip(volume, orientation="coronal")

    # Save to cache
    np.save(axial_path, img_axial)
    np.save(coronal_path, img_coronal)

    return img_axial, img_coronal


# ==========================================
# Dataset Class
# ==========================================


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, cache_dir=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient info.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            cache_dir (str): Directory to store/load cached images.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir

        # Pre-compute normalization stats for tabular data (approximate from training set)
        # Age: Mean~67, Range 50-90. Scale to roughly [-1, 1] or [0, 1]
        self.age_mean = 67.0
        self.age_std = 10.0

        # FVC: Mean~2700, Std~800.
        self.fvc_mean = 2700.0
        self.fvc_std = 1000.0

        # Percent: Mean~77, Std~20
        self.pct_mean = 77.0
        self.pct_std = 20.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        # 'dicom_dir' in metadata is relative path e.g. "train/ID..."
        dicom_dir = row["dicom_dir"]

        img_axial, img_coronal = process_patient_images(
            patient_id, dicom_dir, self.cache_dir, load_cached_data=True
        )

        # 2. Apply Augmentations (Spatial Only)
        if self.transform:
            # Apply same transform to both? Or independent?
            # Independent is fine for dual-stream.
            aug_ax = self.transform(image=img_axial)["image"]
            aug_cor = self.transform(image=img_coronal)["image"]
        else:
            # Convert to tensor manually if no transform
            aug_ax = torch.tensor(img_axial.transpose(2, 0, 1), dtype=torch.float32)
            aug_cor = torch.tensor(img_coronal.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Process Tabular Data
        # Features: [Weeks, Age, Sex, Smoking, Base_FVC, Base_Percent]

        # Weeks (Relative)
        if self.mode == "test":
            # test.csv has 'Predict_Week' and 'Baseline_Week'
            weeks = row["Predict_Week"] - row["Baseline_Week"]
            base_fvc = row["Baseline_FVC"]
            base_pct = row["Baseline_Percent"]
            age = row["Baseline_Age"]
            sex = row["Baseline_Sex"]
            smoking = row["Baseline_SmokingStatus"]
        else:
            # train/val: we computed relative weeks in preprocessing
            weeks = row["Relative_Weeks"]
            base_fvc = row["Baseline_FVC"]
            base_pct = row["Baseline_Percent"]
            age = row["Age"]
            sex = row["Sex"]
            smoking = row["SmokingStatus"]

        # Normalize Features
        # Weeks: Scale? It can be negative. Let's keep it raw or scale by 100.
        # Model expects raw-ish values or normalized?
        # BBSL-Net uses MLP, so normalization helps.
        feat_weeks = weeks / 100.0
        feat_age = (age - self.age_mean) / self.age_std
        feat_base_fvc = (base_fvc - self.fvc_mean) / self.fvc_std
        feat_base_pct = (base_pct - self.pct_mean) / self.pct_std

        # Categorical Encoding
        # Sex: Male=0, Female=1
        feat_sex = 1.0 if sex == "Female" else 0.0

        # Smoking: Ex=0, Never=1, Current=2
        if smoking == "Ex-smoker":
            feat_smoke = 0.0
        elif smoking == "Never smoked":
            feat_smoke = 1.0
        else:
            feat_smoke = 2.0

        # One-hot for smoking (3 dims)
        smoke_oh = [0.0, 0.0, 0.0]
        smoke_oh[int(feat_smoke)] = 1.0

        # Assemble Feature Vector
        # [Weeks, Age, Sex, Smoke_0, Smoke_1, Smoke_2, Base_FVC, Base_Percent]
        # Total dims: 1 + 1 + 1 + 3 + 1 + 1 = 8
        tabular = torch.tensor(
            [
                feat_weeks,
                feat_age,
                feat_sex,
                smoke_oh[0],
                smoke_oh[1],
                smoke_oh[2],
                feat_base_fvc,
                feat_base_pct,
            ],
            dtype=torch.float32,
        )

        # 4. Target
        if self.mode != "test":
            target = torch.tensor(row["FVC"], dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)  # Dummy

        # 5. Metadata
        meta = {
            "Patient_Week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{weeks}"
            ),
            "Base_FVC": base_fvc,
            "Weeks": weeks,
        }

        return {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": tabular,
            "target": target,
            "meta": meta,
        }


# ==========================================
# Data Preparation Logic
# ==========================================


def preprocess_train_data(df):
    """
    Prepares training data by identifying baseline features and calculating relative weeks.
    """
    # Group by Patient to find baseline
    # Baseline is defined as the visit with min Weeks (usually 0 or negative)
    patient_groups = df.groupby("Patient")

    processed_rows = []

    for pid, group in patient_groups:
        # Find baseline row
        # Sort by Weeks
        group = group.sort_values("Weeks")
        baseline_row = group.iloc[0]

        base_week = baseline_row["Weeks"]
        base_fvc = baseline_row["FVC"]
        base_pct = baseline_row["Percent"]

        # Add baseline info to all rows
        group["Baseline_Week"] = base_week
        group["Baseline_FVC"] = base_fvc
        group["Baseline_Percent"] = base_pct
        group["Relative_Weeks"] = group["Weeks"] - base_week

        processed_rows.append(group)

    return pd.concat(processed_rows, ignore_index=True)


def get_dataloaders(debug=False):
    """
    Main entry point to get DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Debug Mode
    if debug:
        train_df = train_df.head(Config.BATCH_SIZE * 2)
        val_df = val_df.head(Config.BATCH_SIZE)
        test_df = test_df.head(Config.BATCH_SIZE)

    # 3. Preprocess Tabular Data
    print("Preprocessing tabular data...")
    train_df = preprocess_train_data(train_df)
    val_df = preprocess_train_data(val_df)
    # Test data is already structured with Baseline columns in metadata generation

    # 4. Define Transforms
    # Spatial only, no intensity changes
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD), ToTensorV2()]
    )

    # 5. Create Datasets
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_dataset = OSICDataset(
        train_df, mode="train", transform=train_transform, cache_dir=Config.CACHE_DIR
    )
    val_dataset = OSICDataset(
        val_df, mode="val", transform=val_transform, cache_dir=Config.CACHE_DIR
    )
    test_dataset = OSICDataset(
        test_df, mode="test", transform=val_transform, cache_dir=Config.CACHE_DIR
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
