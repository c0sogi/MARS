import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config

# ==========================================
# 1. Helper Functions for DICOM Processing
# ==========================================


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by InstanceNumber,
    and converts them to a numpy 3D array in Hounsfield Units.
    """
    slices = [pydicom.dcmread(s) for s in glob.glob(os.path.join(path, "*.dcm"))]
    if not slices:
        return None

    # Sort by Instance Number to reconstruct the 3D volume correctly
    slices.sort(key=lambda x: int(x.InstanceNumber))

    try:
        slice_thickness = np.abs(
            slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
        )
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    # Convert to Hounsfield Units (HU)
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    intercept = slices[0].RescaleIntercept
    slope = slices[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return image


def get_tri_slab_mip(volume, axis=0):
    """
    Generates a 3-channel image using Fixed Overlapping Tri-Slabs.

    Args:
        volume: 3D numpy array (D, H, W)
        axis: 0 for Axial (split along D), 1 for Coronal (split along H)

    Returns:
        3-channel numpy array (H, W, 3) or (D, W, 3) depending on projection.
    """
    # If generating Coronal (axis=1), we transpose to make the split axis the first dimension
    # Original: (Z, Y, X). Coronal view looks from Front (X-Z plane), iterating over Y.
    # We want the resulting image to be (Z, X).
    if axis == 1:
        volume = volume.transpose(1, 0, 2)  # Now (Y, Z, X)

    depth = volume.shape[0]

    # Define slab boundaries with 15% overlap
    # Points: 0, 1/3, 2/3, 1
    # Overlap radius: 7.5% of depth
    p1 = int(depth / 3)
    p2 = int(2 * depth / 3)
    r = int(depth * 0.075)  # Total overlap is 15%

    # Slab ranges
    s1_start, s1_end = 0, p1 + r
    s2_start, s2_end = max(0, p1 - r), min(depth, p2 + r)
    s3_start, s3_end = max(0, p2 - r), depth

    # Handle edge case for very small volumes
    if depth < 3:
        s1_end = depth
        s2_start, s2_end = 0, depth
        s3_start = 0

    # Extract slabs
    slab1 = volume[s1_start:s1_end, :, :]
    slab2 = volume[s2_start:s2_end, :, :]
    slab3 = volume[s3_start:s3_end, :, :]

    # Compute MIP (Maximum Intensity Projection)
    # If slab is empty (shouldn't happen with logic above), use zeros
    mip1 = np.max(slab1, axis=0) if slab1.size > 0 else np.zeros(volume.shape[1:])
    mip2 = np.max(slab2, axis=0) if slab2.size > 0 else np.zeros(volume.shape[1:])
    mip3 = np.max(slab3, axis=0) if slab3.size > 0 else np.zeros(volume.shape[1:])

    # Stack to RGB
    # Shape becomes (H, W, 3)
    img = np.stack([mip1, mip2, mip3], axis=-1)

    return img


def resize_image(img, size=224):
    """
    Resizes image to target resolution.
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize_hu(img):
    """
    Normalize Hounsfield Units to roughly [0, 1] or [-1, 1] range for neural nets.
    Lung window: -1000 to 400.
    """
    min_bound = -1000.0
    max_bound = 400.0
    img = (img - min_bound) / (max_bound - min_bound)
    img = np.clip(img, 0, 1)
    return img


def process_patient(patient_id, dicom_dir, cache_dir):
    """
    Loads DICOM, generates Axial and Coronal Tri-Slabs, resizes, and saves to cache.
    """
    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # If both exist, skip
    if os.path.exists(axial_path) and os.path.exists(coronal_path):
        return

    full_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
    volume = load_scan(full_path)

    if volume is None:
        # Create dummy black images if load fails
        dummy = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
        np.save(axial_path, dummy)
        np.save(coronal_path, dummy)
        return

    # Generate Axial Tri-Slab
    axial_img = get_tri_slab_mip(volume, axis=0)
    axial_img = normalize_hu(axial_img)
    axial_img = resize_image(axial_img, Config.IMG_SIZE)

    # Generate Coronal Tri-Slab
    coronal_img = get_tri_slab_mip(volume, axis=1)
    coronal_img = normalize_hu(coronal_img)
    coronal_img = resize_image(coronal_img, Config.IMG_SIZE)

    # Save as float32
    np.save(axial_path, axial_img.astype(np.float32))
    np.save(coronal_path, coronal_img.astype(np.float32))


def process_and_cache_data(df, cache_dir, load_cached_data=True):
    """
    Iterates over unique patients in dataframe and processes their images.
    """
    os.makedirs(cache_dir, exist_ok=True)

    unique_patients = df[["Patient", "dicom_dir"]].drop_duplicates()

    if not load_cached_data:
        print(f"Reprocessing all {len(unique_patients)} patients (Cache ignored)...")
    else:
        print(f"Checking cache for {len(unique_patients)} patients...")

    for _, row in unique_patients.iterrows():
        pid = row["Patient"]
        d_dir = row["dicom_dir"]

        # Check if files exist
        f1 = os.path.join(cache_dir, f"{pid}_axial.npy")
        f2 = os.path.join(cache_dir, f"{pid}_coronal.npy")

        if not load_cached_data or not (os.path.exists(f1) and os.path.exists(f2)):
            process_patient(pid, d_dir, cache_dir)


# ==========================================
# 2. Dataset Class
# ==========================================


class LungDataset(Dataset):
    def __init__(
        self, df, cache_dir, transform=None, tabular_transform=None, is_train=True
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.tabular_transform = tabular_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # Load Images
        axial_path = os.path.join(self.cache_dir, f"{pid}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{pid}_coronal.npy")

        try:
            img_ax = np.load(axial_path)
            img_cor = np.load(coronal_path)
        except FileNotFoundError:
            # Fallback
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Apply Augmentations
        # Note: We apply transforms independently or consistently?
        # Standard albumentations apply to one image. We need to apply to both.
        # Since they are orthogonal views, independent spatial augmentation is acceptable/beneficial.

        if self.transform:
            res_ax = self.transform(image=img_ax)
            img_ax = res_ax["image"]

            res_cor = self.transform(image=img_cor)
            img_cor = res_cor["image"]
        else:
            # Basic to tensor
            t = ToTensorV2()
            img_ax = t(image=img_ax)["image"]
            img_cor = t(image=img_cor)["image"]

        # Prepare Tabular Data
        # Features: Age, Sex, SmokingStatus, Percent, Baseline_FVC, Relative_Week
        # We assume df has these columns processed

        # Extract raw values
        age = float(row["Baseline_Age"])
        sex = 0.0 if row["Baseline_Sex"] == "Male" else 1.0

        ss = row["Baseline_SmokingStatus"]
        if ss == "Ex-smoker":
            smoke = 0.0
        elif ss == "Never smoked":
            smoke = 1.0
        else:
            smoke = 2.0  # Currently smokes

        percent = float(row["Baseline_Percent"])
        base_fvc = float(row["Baseline_FVC"])

        # Calculate relative week
        # For train/val: Weeks - Baseline_Week
        # For test: Predict_Week - Baseline_Week
        if "Predict_Week" in row:
            rel_week = float(row["Predict_Week"] - row["Baseline_Week"])
        else:
            rel_week = float(row["Weeks"] - row["Baseline_Week"])

        # Tabular vector construction
        # Order: [Age, Sex, Smoking, Percent, Baseline_FVC, Rel_Week]
        tab_vec = np.array(
            [age, sex, smoke, percent, base_fvc, rel_week], dtype=np.float32
        )

        if self.tabular_transform:
            # We only scale continuous variables: Age (0), Percent (3), Base_FVC (4), Rel_Week (5)
            # Sex (1) and Smoke (2) are categorical/ordinal
            # To simplify, we'll assume tabular_transform handles the vector or we do manual scaling in get_dataloaders
            # Here we just return the vector, scaling happens before or we assume pre-scaled in DF?
            # Better: Scale in __getitem__ using provided scaler params or scaler object?
            # We'll assume the scaler is applied to specific indices if passed,
            # but usually sklearn scaler expects 2D array.
            # Let's apply scaler manually using mean/scale provided in tabular_transform dict

            if isinstance(self.tabular_transform, dict):
                means = self.tabular_transform["mean"]
                scales = self.tabular_transform["scale"]
                # Indices to scale: Age, Percent, Base_FVC, Rel_Week
                # Indices in vector: 0, 3, 4, 5
                for i, idx_in_vec in enumerate([0, 3, 4, 5]):
                    tab_vec[idx_in_vec] = (tab_vec[idx_in_vec] - means[i]) / scales[i]

        # Target
        if "FVC" in row and self.is_train:
            target = float(row["FVC"])
        else:
            target = 0.0  # Dummy for test

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{pid}_{int(rel_week)}"
            ),
        }


# ==========================================
# 3. Data Pipeline & Loader
# ==========================================


def enrich_metadata_with_baseline(df, is_test=False):
    """
    Ensures the dataframe has Baseline_FVC, Baseline_Percent, etc.
    For Train/Val, we find the visit at Week ~0.
    For Test, it's already provided.
    """
    if is_test:
        return df

    # For Train/Val, we need to find baseline for each patient
    # Baseline is defined as the visit with min(abs(Weeks))

    df["abs_weeks"] = df["Weeks"].abs()
    df = df.sort_values(["Patient", "abs_weeks"])

    # Extract baseline rows
    baseline_df = df.groupby("Patient").first().reset_index()

    # Select columns to merge back
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

    # Rename to Baseline_...
    rename_map = {
        "FVC": "Baseline_FVC",
        "Percent": "Baseline_Percent",
        "Age": "Baseline_Age",
        "Sex": "Baseline_Sex",
        "SmokingStatus": "Baseline_SmokingStatus",
        "Weeks": "Baseline_Week",
    }
    baseline_df = baseline_df.rename(columns=rename_map)

    # Merge back to original df
    # We drop original static columns from df to avoid confusion/duplication if they exist
    # But Age/Sex/Smoking are static per patient usually. FVC/Percent change.

    df_out = pd.merge(df, baseline_df, on="Patient", how="left")

    return df_out


def get_dataloaders(load_cached_data=True):
    """
    Main entry point. Loads CSVs, processes images, prepares DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
        val_df = val_df.iloc[: Config.MAX_TRAIN_SAMPLES]

    # 2. Enrich Metadata (Train/Val)
    train_df = enrich_metadata_with_baseline(train_df)
    val_df = enrich_metadata_with_baseline(val_df)
    # Test df already has Baseline columns from metadata generation

    # 3. Process & Cache Images
    # Combine all unique patients to process
    all_patients = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]],
            val_df[["Patient", "dicom_dir"]],
            test_df[["Patient", "dicom_dir"]],
        ]
    )
    process_and_cache_data(all_patients, Config.CACHE_DIR, load_cached_data)

    # 4. Fit Scalers for Tabular Data
    # We scale: Age, Percent, Baseline_FVC, Relative_Week
    # Calculate Relative Week for stats
    train_rel_week = train_df["Weeks"] - train_df["Baseline_Week"]

    # Gather data for fitting
    # [Age, Percent, Baseline_FVC, Rel_Week]
    data_to_fit = np.stack(
        [
            train_df["Baseline_Age"].values,
            train_df["Baseline_Percent"].values,
            train_df["Baseline_FVC"].values,
            train_rel_week.values,
        ],
        axis=1,
    )

    scaler = StandardScaler()
    scaler.fit(data_to_fit)

    tabular_stats = {"mean": scaler.mean_, "scale": scaler.scale_}

    # 5. Define Augmentations
    # Spatial only: Flips, Shifts, Rotations. No intensity changes.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 6. Create Datasets
    train_dataset = LungDataset(
        train_df,
        Config.CACHE_DIR,
        transform=train_transform,
        tabular_transform=tabular_stats,
        is_train=True,
    )

    val_dataset = LungDataset(
        val_df,
        Config.CACHE_DIR,
        transform=val_transform,
        tabular_transform=tabular_stats,
        is_train=True,  # Has targets
    )

    test_dataset = LungDataset(
        test_df,
        Config.CACHE_DIR,
        transform=val_transform,
        tabular_transform=tabular_stats,
        is_train=False,  # No targets
    )

    # 7. Create Loaders
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
