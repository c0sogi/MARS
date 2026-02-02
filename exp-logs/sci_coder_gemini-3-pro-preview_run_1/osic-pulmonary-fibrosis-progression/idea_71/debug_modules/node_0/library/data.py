import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom

from library.config import Config

# Ensure cache directory exists
os.makedirs(Config.CACHE_DIR, exist_ok=True)


def load_scan_tri_slab(patient_id, dicom_dir, load_cached_data=True):
    """
    Generates Fixed Overlapping Axial and Coronal Tri-Slab images.

    Args:
        patient_id (str): Unique Patient ID (used for caching).
        dicom_dir (str): Path to the directory containing DICOM files.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (axial_img, coronal_img)
               Both are numpy arrays of shape (224, 224, 3), float32, range [0, 1].
    """
    cache_path_ax = os.path.join(Config.CACHE_DIR, f"{patient_id}_axial.npy")
    cache_path_cor = os.path.join(Config.CACHE_DIR, f"{patient_id}_coronal.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        if os.path.exists(cache_path_ax) and os.path.exists(cache_path_cor):
            try:
                ax = np.load(cache_path_ax)
                cor = np.load(cache_path_cor)
                return ax, cor
            except Exception:
                pass  # Fallback to processing if load fails

    # 2. Load and Process DICOMs
    # Initialize blank images in case of failure
    blank_img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    if not os.path.exists(dicom_dir):
        return blank_img, blank_img

    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not files:
        return blank_img, blank_img

    # Read DICOMs
    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure pixel array is present
            if hasattr(dcm, "pixel_array"):
                slices.append(dcm)
        except:
            continue

    if not slices:
        return blank_img, blank_img

    # Sort slices
    # Try ImagePositionPatient[2] (Z-axis), fallback to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass  # Keep original order

    # Create 3D Volume and Convert to HU
    images = []
    for s in slices:
        img_2d = s.pixel_array.astype(np.float32)
        # Apply Rescale Slope/Intercept if available
        if hasattr(s, "RescaleSlope") and hasattr(s, "RescaleIntercept"):
            slope = float(s.RescaleSlope)
            intercept = float(s.RescaleIntercept)
            img_2d = img_2d * slope + intercept
        images.append(img_2d)

    if not images:
        return blank_img, blank_img

    volume = np.stack(images)  # Shape: (Depth, Height, Width)

    # Helper function for Tri-Slab MIP
    def get_tri_slab_mip(vol_data):
        # vol_data shape: (Depth, H, W)
        D = vol_data.shape[0]
        if D < 1:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Define slab boundaries with overlap
        # Slab 1: 0% - 40%
        # Slab 2: 30% - 70%
        # Slab 3: 60% - 100%
        if D == 1:
            slab1 = slab2 = slab3 = vol_data[0]
        else:
            s1_end = int(D * 0.40) + 1
            s2_start = int(D * 0.30)
            s2_end = int(D * 0.70) + 1
            s3_start = int(D * 0.60)

            # Clamp indices
            s1_end = min(s1_end, D)
            s2_start = max(0, min(s2_start, D - 1))
            s2_end = min(max(s2_end, s2_start + 1), D)
            s3_start = max(0, min(s3_start, D - 1))

            # Compute MIPs (max over depth axis)
            slab1 = np.max(vol_data[0:s1_end], axis=0)
            slab2 = np.max(vol_data[s2_start:s2_end], axis=0)
            slab3 = np.max(vol_data[s3_start:], axis=0)

        # Stack to RGB (H, W, 3)
        img = np.stack([slab1, slab2, slab3], axis=-1)
        return img

    # 3. Axial Tri-Slab (View along Z-axis)
    axial_mip = get_tri_slab_mip(volume)

    # 4. Coronal Tri-Slab (View along Y-axis)
    # Transpose volume to (Y, Z, X) so Y becomes the depth dimension
    # Standard DICOM: (Z, Y, X).
    vol_cor = volume.transpose(1, 0, 2)
    coronal_mip = get_tri_slab_mip(vol_cor)

    # 5. Post-Processing (Resize & Normalize)
    def post_process(img):
        # Resize to Config.IMG_SIZE (224x224)
        img_resized = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR
        )

        # Windowing (Lung Window: -1000 to 400 HU)
        min_hu = -1000.0
        max_hu = 400.0
        img_clipped = np.clip(img_resized, min_hu, max_hu)

        # Normalize to [0, 1]
        img_norm = (img_clipped - min_hu) / (max_hu - min_hu)
        return img_norm.astype(np.float32)

    axial_final = post_process(axial_mip)
    coronal_final = post_process(coronal_mip)

    # 6. Save to Cache
    np.save(cache_path_ax, axial_final)
    np.save(cache_path_cor, coronal_final)

    return axial_final, coronal_final


class LungDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, cache_data=True):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.cache_data = cache_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Construct full DICOM path
        # Metadata 'dicom_dir' is relative (e.g., "train/ID...")
        dicom_path = os.path.join(Config.INPUT_DIR, row["dicom_dir"])

        # Load Images
        img_ax, img_cor = load_scan_tri_slab(
            patient_id, dicom_path, load_cached_data=self.cache_data
        )

        # Apply Augmentations
        # Albumentations expects (H, W, C) input
        if self.transform:
            # Apply transform independently to both views
            # Note: We use the same pipeline, but random parameters might differ per call
            # This is acceptable as views are physically distinct planes
            res_ax = self.transform(image=img_ax)["image"]
            res_cor = self.transform(image=img_cor)["image"]
        else:
            t = ToTensorV2()
            res_ax = t(image=img_ax)["image"]
            res_cor = t(image=img_cor)["image"]

        # Process Tabular Data
        # 1. Age (Standardized)
        age_raw = float(row["Age"]) if "Age" in row else float(row["Baseline_Age"])
        age = (age_raw - Config.TABULAR_MEAN["Age"]) / Config.TABULAR_STD["Age"]

        # 2. Percent (Standardized)
        # Use 'Percent' for train, 'Baseline_Percent' for test
        if "Percent" in row:
            pct_raw = float(row["Percent"])
        elif "Baseline_Percent" in row:
            pct_raw = float(row["Baseline_Percent"])
        else:
            pct_raw = Config.TABULAR_MEAN["Percent"]
        pct = (pct_raw - Config.TABULAR_MEAN["Percent"]) / Config.TABULAR_STD["Percent"]

        # 3. Sex (Encoded)
        sex_str = row["Sex"] if "Sex" in row else row["Baseline_Sex"]
        sex = float(Config.SEX_MAP.get(sex_str, 0))

        # 4. SmokingStatus (Encoded)
        smoke_str = (
            row["SmokingStatus"]
            if "SmokingStatus" in row
            else row["Baseline_SmokingStatus"]
        )
        smoke = float(Config.SMOKING_MAP.get(smoke_str, 0))

        # Construct tabular tensor
        tabular = torch.tensor([age, pct, sex, smoke], dtype=torch.float32)

        # Target FVC
        if "FVC" in row:
            target = torch.tensor(row["FVC"], dtype=torch.float32)
        else:
            target = torch.tensor(0.0, dtype=torch.float32)

        # Weeks
        if "Weeks" in row:
            week = float(row["Weeks"])
        elif "Predict_Week" in row:
            week = float(row["Predict_Week"])
        else:
            week = 0.0

        # Baseline Information (for inference logic)
        base_week = float(row["Baseline_Week"]) if "Baseline_Week" in row else 0.0
        base_fvc = float(row["Baseline_FVC"]) if "Baseline_FVC" in row else 0.0

        return {
            "image_axial": res_ax,
            "image_coronal": res_cor,
            "tabular": tabular,
            "target": target,
            "week": torch.tensor(week, dtype=torch.float32),
            "base_week": torch.tensor(base_week, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "patient_id": patient_id,
        }


def add_baseline_info(df):
    """
    Augments the dataframe with Baseline_FVC and Baseline_Week for training data.
    Assumes baseline is the visit closest to Week 0.
    """
    if "Baseline_FVC" in df.columns:
        return df

    baselines = []
    # Group by patient to find their specific baseline
    for pid, group in df.groupby("Patient"):
        # Find index of row with Week closest to 0
        idx = (group["Weeks"] - 0).abs().idxmin()
        base_fvc = group.loc[idx, "FVC"]
        base_week = group.loc[idx, "Weeks"]
        baselines.append(
            {"Patient": pid, "Baseline_FVC": base_fvc, "Baseline_Week": base_week}
        )

    base_df = pd.DataFrame(baselines)
    # Merge baseline info back to original dataframe
    df = pd.merge(df, base_df, on="Patient", how="left")
    return df


def get_dataloaders(debug=False):
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Add Baseline Info to training/validation sets
    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(50)

    # Define Transforms (Spatial Only)
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # Instantiate Datasets
    train_dataset = LungDataset(
        train_df, Config.INPUT_DIR, transform=train_transform, cache_data=True
    )
    val_dataset = LungDataset(
        val_df, Config.INPUT_DIR, transform=val_transform, cache_data=True
    )

    # Instantiate DataLoaders
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader():
    test_df = pd.read_csv(Config.TEST_CSV)

    test_transform = A.Compose([ToTensorV2()])

    test_dataset = LungDataset(
        test_df, Config.INPUT_DIR, transform=test_transform, cache_data=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
