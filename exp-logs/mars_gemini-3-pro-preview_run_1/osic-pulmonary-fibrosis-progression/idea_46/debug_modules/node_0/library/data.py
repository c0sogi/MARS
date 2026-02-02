import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config

# Try importing pydicom, handle case if not present (though essential for this task)
try:
    import pydicom
except ImportError:
    print("WARNING: pydicom not found. DICOM processing will fail.")

# ==========================================
# Constants & Statistics
# ==========================================
# Stats from EDA for normalization
STATS = {
    "Age_mean": 67.58,
    "Age_std": 6.63,
    "Percent_mean": 76.91,
    "Percent_std": 19.20,
}

# Lung Window Settings
LUNG_WIN_WIDTH = 1500
LUNG_WIN_LEVEL = -600

# ==========================================
# Helper Functions
# ==========================================


def get_img_seq(dicom_dir):
    """
    Loads DICOM files from a directory, sorts them by slice location,
    and returns a list of pydicom objects.
    """
    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        return []

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dicom_dir, f))
            slices.append(ds)
        except Exception:
            continue

    # Sort by ImagePositionPatient Z coordinate
    # If missing, fall back to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def window_image(img, window_center, window_width):
    """Applies CT windowing to the image."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def get_hu_pixels(slices):
    """
    Converts list of pydicom slices to a 3D numpy array of Hounsfield Units.
    Returns shape (Depth, Height, Width).
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to HU
    # The intercept is usually -1024, so air is approx -1000
    for n, s in enumerate(slices):
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            image[n] = slope * image[n].astype(np.float64)
            image[n] = image[n].astype(np.int16)

        image[n] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def generate_tri_slab(volume, axis=0, overlap=0.15, target_size=224):
    """
    Generates a 3-channel image using Fixed Overlapping Tri-Slabs via MIP.

    Args:
        volume (np.array): 3D HU volume (D, H, W).
        axis (int): 0 for Axial (split D), 1 for Coronal (split H/Y).
        overlap (float): Fraction of overlap.
        target_size (int): Output spatial resolution.

    Returns:
        np.array: (target_size, target_size, 3) normalized [0,1] image.
    """
    # If generating Coronal (axis=1), we need to permute so the split axis is 0
    # Original: (Z, Y, X) -> (0, 1, 2)
    # Axial: Split Z (0), project -> (Y, X)
    # Coronal: Split Y (1), project -> (Z, X)

    if axis == 1:
        # Permute to (Y, Z, X) so we can treat Y as the depth dimension to split
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Define slab boundaries
    # We want 3 slabs covering [0, depth]
    # Slab 1: 0 to 1/3 + overlap
    # Slab 2: 1/3 - overlap to 2/3 + overlap
    # Slab 3: 2/3 - overlap to 1

    p1 = int(depth / 3)
    p2 = int(depth * 2 / 3)
    ov = int(depth * overlap)

    s1_start, s1_end = 0, p1 + ov
    s2_start, s2_end = max(0, p1 - ov), p2 + ov
    s3_start, s3_end = max(0, p2 - ov), depth

    # Extract slabs
    slab1 = volume[s1_start:s1_end, :, :]
    slab2 = volume[s2_start:s2_end, :, :]
    slab3 = volume[s3_start:s3_end, :, :]

    # Handle edge case of single slice or empty
    if slab1.shape[0] == 0:
        slab1 = volume
    if slab2.shape[0] == 0:
        slab2 = volume
    if slab3.shape[0] == 0:
        slab3 = volume

    # Compute MIP (Maximum Intensity Projection)
    # MIP is taken along the new depth axis (0)
    mip1 = np.max(slab1, axis=0)
    mip2 = np.max(slab2, axis=0)
    mip3 = np.max(slab3, axis=0)

    # Stack to channels
    img = np.stack([mip1, mip2, mip3], axis=-1)

    # Windowing (Lung)
    img = window_image(img, LUNG_WIN_LEVEL, LUNG_WIN_WIDTH)

    # Normalize to [0, 1]
    img = (img - (LUNG_WIN_LEVEL - LUNG_WIN_WIDTH // 2)) / LUNG_WIN_WIDTH
    img = np.clip(img, 0, 1)

    # Resize
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    return img.astype(np.float32)


def process_patient_scans(dicom_dir, patient_id, cache_dir):
    """
    Orchestrates the loading, processing, and caching of patient scans.
    Returns (axial_img, coronal_img).
    """
    os.makedirs(cache_dir, exist_ok=True)

    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # Check cache
    if os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to re-processing

    # Load DICOMs
    slices = get_img_seq(dicom_dir)
    if not slices:
        # Return black images if no dicoms found (should not happen based on EDA)
        empty = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        return empty, empty

    # Convert to Volume
    vol = get_hu_pixels(slices)  # (Z, Y, X)

    # Generate Views
    # Axial: Split Z (axis 0)
    axial = generate_tri_slab(vol, axis=0, target_size=Config.IMG_SIZE)

    # Coronal: Split Y (axis 1)
    coronal = generate_tri_slab(vol, axis=1, target_size=Config.IMG_SIZE)

    # Save to cache
    np.save(axial_path, axial)
    np.save(coronal_path, coronal)

    return axial, coronal


def prepare_training_data(df):
    """
    Augments the training/validation dataframe with Baseline information.
    For each patient, finds the visit closest to Week 0 and uses its attributes as baseline.
    """
    # Ensure we don't overwrite if already exists
    if "Baseline_FVC" in df.columns:
        return df

    df = df.copy()
    patients = df["Patient"].unique()

    baseline_data = []

    for p in patients:
        p_data = df[df["Patient"] == p]
        # Find row with min absolute weeks
        idx_min = p_data["Weeks"].abs().idxmin()
        baseline_row = p_data.loc[idx_min]

        baseline_data.append(
            {
                "Patient": p,
                "Baseline_Week": baseline_row["Weeks"],
                "Baseline_FVC": baseline_row["FVC"],
                "Baseline_Percent": baseline_row["Percent"],
                "Baseline_Age": baseline_row["Age"],
                "Baseline_Sex": baseline_row["Sex"],
                "Baseline_SmokingStatus": baseline_row["SmokingStatus"],
            }
        )

    baseline_df = pd.DataFrame(baseline_data)

    # Merge back
    df = pd.merge(df, baseline_df, on="Patient", how="left")
    return df


# ==========================================
# Dataset Class
# ==========================================


class PulmonaryDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient info.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Pre-compute tabular features normalization
        # We use Baseline features for consistency
        self.df["Norm_Age"] = (self.df["Baseline_Age"] - STATS["Age_mean"]) / STATS[
            "Age_std"
        ]
        self.df["Norm_Percent"] = (
            self.df["Baseline_Percent"] - STATS["Percent_mean"]
        ) / STATS["Percent_std"]

        # Encode Sex (Male: 0, Female: 1)
        self.df["Enc_Sex"] = self.df["Baseline_Sex"].apply(
            lambda x: 1 if x == "Female" else 0
        )

        # Encode Smoking (One Hot: Ex-smoker, Never smoked, Currently smokes)
        # Order: [Ex, Never, Current]
        self.df["Smoke_Ex"] = (self.df["Baseline_SmokingStatus"] == "Ex-smoker").astype(
            int
        )
        self.df["Smoke_Never"] = (
            self.df["Baseline_SmokingStatus"] == "Never smoked"
        ).astype(int)
        self.df["Smoke_Current"] = (
            self.df["Baseline_SmokingStatus"] == "Currently smokes"
        ).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Cached or Processed)
        # Construct full path to DICOM directory
        dicom_dir = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        axial, coronal = process_patient_scans(dicom_dir, patient_id, Config.CACHE_DIR)

        # 2. Apply Augmentations
        if self.transform:
            # Albumentations requires images to be passed as named arguments
            # We apply the same spatial transform to both if possible, or separate?
            # Usually we want independent or consistent?
            # Since they are orthogonal views, independent spatial augs are acceptable
            # but consistent might be better. However, they are different coordinate systems.
            # We will apply transforms independently for simplicity and robustness.
            res_ax = self.transform(image=axial)["image"]
            res_cor = self.transform(image=coronal)["image"]
            axial = res_ax
            coronal = res_cor
        else:
            # Convert to tensor manually if no transform
            axial = torch.tensor(axial.transpose(2, 0, 1), dtype=torch.float32)
            coronal = torch.tensor(coronal.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Tabular Features
        # Vector: [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Percent]
        tab_vec = np.array(
            [
                row["Norm_Age"],
                row["Enc_Sex"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["Norm_Percent"],
            ],
            dtype=np.float32,
        )

        # 4. Target & Meta
        # For Test set, FVC might be dummy, but we need it for format consistency
        target_fvc = row["FVC"]
        current_week = row["Predict_Week"] if "Predict_Week" in row else row["Weeks"]

        # Meta info for Anchored Trajectory Logic
        baseline_fvc = row["Baseline_FVC"]
        baseline_week = row["Baseline_Week"]

        return {
            "img_axial": axial,
            "img_coronal": coronal,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "weeks": torch.tensor(current_week, dtype=torch.float32),
            "base_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "base_week": torch.tensor(baseline_week, dtype=torch.float32),
            "patient_id": patient_id,
        }


# ==========================================
# Data Loaders
# ==========================================


def get_dataloaders(debug=False):
    """
    Creates and returns DataLoaders for train, val, and test sets.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Data (Add Baseline info to Train/Val)
    train_df = prepare_training_data(train_df)
    val_df = prepare_training_data(val_df)

    # Debug Mode: Subset data
    if debug or Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        # Keep test full usually, or subset if really debugging pipeline
        # test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Define Transforms
    # Spatial only, no intensity changes
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 4. Create Datasets
    train_ds = PulmonaryDataset(train_df, mode="train", transform=train_transform)
    val_ds = PulmonaryDataset(val_df, mode="val", transform=val_transform)
    test_ds = PulmonaryDataset(test_df, mode="test", transform=val_transform)

    # 5. Create Loaders
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
