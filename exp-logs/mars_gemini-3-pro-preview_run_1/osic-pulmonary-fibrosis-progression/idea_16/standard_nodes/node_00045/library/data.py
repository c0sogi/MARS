import os
import cv2
import glob
import numpy as np
import pandas as pd
import pydicom
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

from library.config import Config
from library.utils import seed_everything


def get_img_custom_code(path):
    """
    Loads all DICOM files from a directory, sorts them by SliceLocation/InstanceNumber,
    converts to Hounsfield Units, applies Lung Windowing, and returns a 3D volume.
    """
    if not os.path.exists(path):
        return None

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return None

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure necessary attributes exist
            if not hasattr(dcm, "PixelData"):
                continue
            slices.append(dcm)
        except:
            continue

    if not slices:
        return None

    # Sort slices
    # Try sorting by ImagePositionPatient Z, then SliceLocation, then InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

    # Load pixel data and stack
    image_stack = []
    for s in slices:
        try:
            img = s.pixel_array.astype(np.float32)
        except RuntimeError:
            continue

        # Convert to Hounsfield Units (HU)
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)
        img = img * slope + intercept

        image_stack.append(img)

    if not image_stack:
        return None

    vol = np.stack(image_stack, axis=0)  # (Z, Y, X)

    # Lung Windowing: WL -600, WW 1500 -> Min -1350, Max 150
    # Standard lung window
    L, W = -600, 1500
    lower = L - W // 2
    upper = L + W // 2

    vol = np.clip(vol, lower, upper)

    # Normalize to [0, 255]
    vol = (vol - lower) / (upper - lower)
    vol = (vol * 255).astype(np.uint8)

    return vol


def generate_tri_slab(vol, axis=0, img_size=224):
    """
    Generates a 3-channel image using Fixed Overlapping Tri-Slabs via MIP.

    Args:
        vol: 3D numpy array (Z, Y, X)
        axis: 0 for Axial (splitting Z), 1 for Coronal (splitting Y)
        img_size: Output spatial resolution

    Returns:
        numpy array of shape (img_size, img_size, 3)
    """
    if vol is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Determine the dimension to split
    # If axis=0 (Axial), we split Z (dim 0). The projection plane is (Y, X).
    # If axis=1 (Coronal), we split Y (dim 1). The projection plane is (Z, X).

    split_dim = axis
    N = vol.shape[split_dim]

    # Define Slab Boundaries (0-33%, 33-66%, 66-100%) with 15% overlap
    # Chunk size
    chunk = N / 3.0
    overlap = chunk * 0.15

    # Indices
    idx_0 = 0
    idx_1 = int(chunk + overlap)

    idx_2 = int(chunk - overlap)
    idx_3 = int(2 * chunk + overlap)

    idx_4 = int(2 * chunk - overlap)
    idx_5 = N

    # Ensure indices are within bounds and valid
    idx_1 = max(idx_1, 1)
    idx_2 = max(idx_2, 0)
    idx_3 = max(idx_3, idx_2 + 1)
    idx_4 = max(idx_4, 0)

    ranges = [(idx_0, idx_1), (idx_2, idx_3), (idx_4, idx_5)]
    channels = []

    for start, end in ranges:
        if start >= end:
            # Fallback for very small volumes
            start = 0
            end = N

        # Slice the volume
        if axis == 0:
            slab = vol[start:end, :, :]
            # MIP along Z
            projection = np.max(slab, axis=0)
        else:
            slab = vol[:, start:end, :]
            # MIP along Y
            projection = np.max(slab, axis=1)

        # Resize to target resolution
        # cv2.resize expects (width, height) -> (X, Y)
        # If axis=0, projection is (Y, X).
        # If axis=1, projection is (Z, X).
        resized = cv2.resize(
            projection, (img_size, img_size), interpolation=cv2.INTER_AREA
        )
        channels.append(resized)

    # Stack to RGB (H, W, 3)
    img = np.stack(channels, axis=-1)
    return img


def process_patient(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Handles caching logic. Checks for existing .npy files.
    If missing or forced, loads DICOM, generates Axial/Coronal slabs, and saves.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # Check cache
    if load_cached_data and os.path.exists(axial_path) and os.path.exists(coronal_path):
        return

    # Process
    full_dicom_path = os.path.join(Config.DICOM_ROOT, dicom_dir)
    vol = get_img_custom_code(full_dicom_path)

    # Generate Axial (Axis 0)
    axial_img = generate_tri_slab(vol, axis=0, img_size=Config.IMG_SIZE)

    # Generate Coronal (Axis 1)
    coronal_img = generate_tri_slab(vol, axis=1, img_size=Config.IMG_SIZE)

    # Save
    np.save(axial_path, axial_img)
    np.save(coronal_path, coronal_img)


class LungDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Pre-process Tabular Data
        # Handle schema differences (Train vs Test) - Cite debug_lesson_9
        sex_col = "Sex" if "Sex" in self.df.columns else "Baseline_Sex"
        smoking_col = (
            "SmokingStatus"
            if "SmokingStatus" in self.df.columns
            else "Baseline_SmokingStatus"
        )
        age_col = "Age" if "Age" in self.df.columns else "Baseline_Age"

        # Mapping for categorical variables
        # Sex: Male=0, Female=1
        self.df["Sex_Code"] = (
            self.df[sex_col].map({"Male": 0, "Female": 1}).fillna(0).astype(float)
        )

        # SmokingStatus: Ex-smoker=0, Never smoked=1, Currently smokes=2
        self.df["Smoking_Code"] = (
            self.df[smoking_col]
            .map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})
            .fillna(0)
            .astype(float)
        )

        # Normalize Numerical Features (Simple Min-Max or standard scaling approximation based on typical values)
        # Age: approx 30-90. Scale to 0-1 roughly.
        self.df["Age_Norm"] = (self.df[age_col] - 65.0) / 15.0

        # Percent: approx 50-100.
        # If 'Percent' column exists (it should in train/val).
        # In test, we might need to use 'Baseline_Percent'.
        if "Percent" in self.df.columns:
            self.df["Percent_Norm"] = (self.df["Percent"] - 80.0) / 20.0
        elif "Baseline_Percent" in self.df.columns:
            self.df["Percent_Norm"] = (self.df["Baseline_Percent"] - 80.0) / 20.0
        else:
            self.df["Percent_Norm"] = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Load Images
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        try:
            img_axial = np.load(axial_path)
            img_coronal = np.load(coronal_path)
        except (FileNotFoundError, ValueError):
            # Fallback if cache is missing/corrupt (should not happen if processed correctly)
            img_axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_coronal = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8
            )

        # Apply Augmentations
        if self.transform:
            # Apply same transform logic or independent?
            # Independent is fine as they are different views.
            res_ax = self.transform(image=img_axial)
            img_axial = res_ax["image"]

            res_cor = self.transform(image=img_coronal)
            img_coronal = res_cor["image"]
        else:
            # Just to tensor
            t = ToTensorV2()
            img_axial = t(image=img_axial)["image"]
            img_coronal = t(image=img_coronal)["image"]

        # Normalize images to [0, 1] for Neural Net (uint8 -> float)
        img_axial = img_axial.float() / 255.0
        img_coronal = img_coronal.float() / 255.0

        # Prepare Tabular Vector
        # [Age, Sex, Smoking, Percent]
        tab_vec = torch.tensor(
            [
                row["Age_Norm"],
                row["Sex_Code"],
                row["Smoking_Code"],
                row["Percent_Norm"],
            ],
            dtype=torch.float32,
        )

        # Prepare Target and Time
        if self.mode in ["train", "val"]:
            # Target is FVC
            target = torch.tensor(row["FVC"], dtype=torch.float32)
            # Time is Weeks (relative to baseline)
            weeks = torch.tensor(row["Weeks"], dtype=torch.float32)
        else:
            # Test mode
            target = torch.tensor(0.0, dtype=torch.float32)  # Dummy
            # Calculate relative week: Predict_Week - Baseline_Week
            # In metadata/test.csv, 'Predict_Week' is the target week, 'Baseline_Week' is usually 0
            # But strictly it is Predict_Week - Baseline_Week
            rel_week = row["Predict_Week"] - row["Baseline_Week"]
            weeks = torch.tensor(rel_week, dtype=torch.float32)

        return {
            "img_axial": img_axial,
            "img_coronal": img_coronal,
            "tabular": tab_vec,
            "target": target,
            "weeks": weeks,
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{int(weeks.item())}"
            ),
        }


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare data and return dataloaders.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # 1. Prepare Cache (Process Images)
    # Collect all unique patients and their dicom dirs
    all_patients = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]],
            val_df[["Patient", "dicom_dir"]],
            test_df[["Patient", "dicom_dir"]],
        ]
    ).drop_duplicates(subset=["Patient"])

    print(f"Preparing cache for {len(all_patients)} patients in {Config.CACHE_DIR}...")

    for _, row in all_patients.iterrows():
        process_patient(
            row["Patient"],
            row["dicom_dir"],
            Config.CACHE_DIR,
            load_cached_data=load_cached_data,
        )

    # 2. Define Transforms
    # Spatial-only augmentation for training
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            ToTensorV2(),
        ]
    )

    # No augmentation for val/test
    val_transform = None

    # 3. Create Datasets
    train_dataset = LungDataset(
        train_df, Config.CACHE_DIR, transform=train_transform, mode="train"
    )
    val_dataset = LungDataset(
        val_df, Config.CACHE_DIR, transform=val_transform, mode="val"
    )
    test_dataset = LungDataset(
        test_df, Config.CACHE_DIR, transform=val_transform, mode="test"
    )

    # 4. Create DataLoaders
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
