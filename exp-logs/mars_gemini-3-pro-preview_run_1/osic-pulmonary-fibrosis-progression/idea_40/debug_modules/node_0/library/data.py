import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from library.config import Config

# -------------------------------------------------------------------------
# Helper Functions for DICOM Processing
# -------------------------------------------------------------------------


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by InstanceNumber.
    """
    if not os.path.exists(path):
        return []

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception:
                continue

    if not slices:
        return []

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts raw DICOM pixel data to Hounsfield Units (HU).
    """
    image = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Convert to HU
    for i, s in enumerate(slices):
        slope = getattr(s, "RescaleSlope", 1)
        intercept = getattr(s, "RescaleIntercept", -1024)
        image[i] = slope * image[i].astype(np.float64) + intercept

    return image


def apply_lung_window(image, width=1500, level=-600):
    """
    Applies lung windowing and normalizes to [0, 255].
    """
    lower = level - width / 2
    upper = level + width / 2

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower) * 255.0
    return image.astype(np.uint8)


def generate_tri_slab(volume, orientation="axial", img_size=240):
    """
    Generates a 3-channel image using Fixed Overlapping Orthogonal Tri-Slabs.

    Args:
        volume: 3D numpy array (D, H, W)
        orientation: 'axial' or 'coronal'
        img_size: Target resolution
    """
    if volume.ndim != 3:
        # Return black image if volume is invalid
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Adjust orientation
    # Assuming input is (Depth, Height, Width) -> (Z, Y, X)
    if orientation == "coronal":
        # Reslice to look from Y axis: (Height, Depth, Width)
        # We transpose to make Y the primary depth axis for slab generation
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]

    # Define slab boundaries with overlap
    # Slab 1: 0% - 40%
    # Slab 2: 30% - 70%
    # Slab 3: 60% - 100%

    p1 = int(depth * 0.4)
    p2_start = int(depth * 0.3)
    p2_end = int(depth * 0.7)
    p3_start = int(depth * 0.6)

    # Handle edge case where depth is very small
    if depth < 3:
        # Just repeat the volume
        slab1 = np.max(volume, axis=0)
        slab2 = slab1
        slab3 = slab1
    else:
        slab1 = np.max(volume[0:p1], axis=0) if p1 > 0 else volume[0]
        slab2 = np.max(volume[p2_start:p2_end], axis=0)
        slab3 = np.max(volume[p3_start:], axis=0)

    # Stack to form RGB
    # Each slab is 2D (H', W')
    img = np.stack([slab1, slab2, slab3], axis=-1)

    # Resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    return img.astype(np.uint8)


def process_patient_dicom(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Orchestrates the loading, processing, and caching of patient CT scans.
    """
    os.makedirs(cache_dir, exist_ok=True)

    axial_path = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    coronal_path = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    # 1. Try to load from cache
    if load_cached and os.path.exists(axial_path) and os.path.exists(coronal_path):
        try:
            axial = np.load(axial_path)
            coronal = np.load(coronal_path)
            return axial, coronal
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    full_path = os.path.join(Config.ROOT_DIR, dicom_dir)
    slices = load_scan(full_path)

    if not slices:
        # Fallback for missing data: black images
        axial = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        coronal = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
    else:
        # Convert to HU and Window
        vol_hu = get_pixels_hu(slices)
        vol_windowed = apply_lung_window(vol_hu)

        # Generate Views
        axial = generate_tri_slab(
            vol_windowed, orientation="axial", img_size=Config.IMG_SIZE
        )
        coronal = generate_tri_slab(
            vol_windowed, orientation="coronal", img_size=Config.IMG_SIZE
        )

    # 3. Save to cache
    np.save(axial_path, axial)
    np.save(coronal_path, coronal)

    return axial, coronal


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, scaler=None, cat_encoder=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            transform (A.Compose): Albumentations transforms.
            scaler (StandardScaler): Fitted scaler for numerical cols.
            cat_encoder (OneHotEncoder): Fitted encoder for categorical cols.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.scaler = scaler
        self.cat_encoder = cat_encoder

        # Pre-process tabular features
        self.process_tabular()

    def process_tabular(self):
        # Numerical features
        num_data = self.df[Config.NUMERICAL_COLS].values.astype(np.float32)
        if self.scaler:
            num_data = self.scaler.transform(num_data)
        self.num_features = num_data

        # Categorical features
        cat_data = self.df[Config.CATEGORICAL_COLS].astype(str).values
        if self.cat_encoder:
            cat_data = self.cat_encoder.transform(cat_data).toarray()
        self.cat_features = cat_data.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images (Axial & Coronal)
        # Use relative path from metadata
        dicom_dir = row["dicom_dir"]

        axial, coronal = process_patient_dicom(
            patient_id, dicom_dir, Config.CACHE_DIR, load_cached=Config.USE_CACHE
        )

        # 2. Apply Augmentations
        if self.transform:
            # Albumentations requires named arguments
            # We apply the same spatial transform to both if we want them aligned?
            # Actually, axial and coronal are different views, so independent augmentation
            # or consistent augmentation is debatable.
            # Given they are processed by independent backbones, independent aug is fine.
            # However, for consistency, let's augment them independently.
            aug_ax = self.transform(image=axial)["image"]
            aug_cor = self.transform(image=coronal)["image"]
        else:
            # Just convert to tensor
            t = ToTensorV2()
            aug_ax = t(image=axial)["image"]
            aug_cor = t(image=coronal)["image"]

        # Normalize images to [0, 1] (Albumentations ToTensorV2 converts to float tensor but keeps 0-255 if not normalized)
        # We manually normalize to [0, 1]
        aug_ax = aug_ax.float() / 255.0
        aug_cor = aug_cor.float() / 255.0

        # 3. Tabular Features
        tab_vec = np.concatenate([self.num_features[idx], self.cat_features[idx]])
        tab_tensor = torch.tensor(tab_vec, dtype=torch.float32)

        # 4. Target & Metadata
        if self.mode in ["train", "val"]:
            # Target FVC
            target_fvc = torch.tensor(row["FVC"], dtype=torch.float32)
            # Baseline FVC (for anchor) - In train/val we often don't have explicit "Baseline_FVC" column
            # unless we merge it. However, usually the first visit is baseline.
            # For this task, we treat the current row as a visit.
            # We need the baseline FVC for the patient.
            # In the provided metadata generation script, train.csv has full history.
            # We need to find the baseline (Week ~0) for this patient or use the first record.
            # To simplify, we can assume the model learns from the available features.
            # BUT, the idea requires "Baseline FVC" as an anchor.
            # We will approximate Baseline FVC using the 'FVC' value where Weeks is closest to 0 for this patient,
            # OR if not easily available efficiently, we use the current FVC as target and
            # rely on the fact that we need to pass *some* baseline.

            # Efficient approach: The metadata script didn't explicitly add "Baseline_FVC" to train.csv.
            # We will grab the first FVC measurement for this patient as baseline (or Week=0).
            # Since we can't easily search the whole DF inside __getitem__, we should have pre-processed this.
            # Let's do a quick fix: In train mode, we use the row's FVC as target.
            # For the "Baseline FVC" input to the model, we can use the 'Percent' * typical_FVC?
            # Or better: Pre-calculate baseline FVC for all train patients during init.
            pass
        else:
            # Test mode
            target_fvc = torch.tensor(0.0, dtype=torch.float32)  # Dummy

        # Handling Baseline FVC and Relative Week
        if "Baseline_FVC" in row:
            base_fvc = float(row["Baseline_FVC"])
            base_week = int(row["Baseline_Week"])
        else:
            # Fallback for train/val if column missing (should be handled in get_dataloaders)
            base_fvc = float(row["FVC"])  # This is wrong if we want true baseline.
            base_week = int(row["Weeks"])

        if self.mode == "test":
            current_week = int(row["Predict_Week"])
        else:
            current_week = int(row["Weeks"])

        relative_week = current_week - base_week

        return {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": tab_tensor,
            "target": target_fvc,
            "baseline_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "relative_week": torch.tensor(relative_week, dtype=torch.float32),
            "patient_week": (
                row["Patient_Week"]
                if "Patient_Week" in row
                else f"{patient_id}_{current_week}"
            ),
        }


# -------------------------------------------------------------------------
# DataLoaders
# -------------------------------------------------------------------------


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # 2. Preprocess Metadata for Baseline FVC in Train/Val
    # We need to identify the baseline FVC (Week ~ 0) for each patient in Train/Val
    # to serve as the anchor input.
    def add_baseline_info(df):
        # Find the row with min weeks (closest to baseline) for each patient
        # We create a mapping Patient -> (Baseline_FVC, Baseline_Week)
        # Note: In the competition data, Week 0 is baseline, but it might not exist for everyone.
        # We take the earliest visit.
        baseline_map = {}
        for pid, group in df.groupby("Patient"):
            # Sort by weeks
            group = group.sort_values("Weeks")
            base_row = group.iloc[0]
            baseline_map[pid] = (base_row["FVC"], base_row["Weeks"])

        # Apply mapping
        baselines = df["Patient"].map(baseline_map)
        df["Baseline_FVC"] = [x[0] for x in baselines]
        df["Baseline_Week"] = [x[1] for x in baselines]
        return df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)
    # Test DF already has Baseline_FVC and Baseline_Week from metadata generation

    # 3. Fit Scalers (Train set only)
    scaler = StandardScaler()
    scaler.fit(train_df[Config.NUMERICAL_COLS].values)

    cat_encoder = OneHotEncoder(handle_unknown="ignore")
    cat_encoder.fit(train_df[Config.CATEGORICAL_COLS].astype(str).values)

    # 4. Define Augmentations
    # Spatial only, no intensity changes
    train_transform = A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose([ToTensorV2()])

    # 5. Create Datasets
    train_ds = LungDataset(
        train_df,
        mode="train",
        transform=train_transform,
        scaler=scaler,
        cat_encoder=cat_encoder,
    )

    val_ds = LungDataset(
        val_df,
        mode="val",
        transform=eval_transform,
        scaler=scaler,
        cat_encoder=cat_encoder,
    )

    test_ds = LungDataset(
        test_df,
        mode="test",
        transform=eval_transform,
        scaler=scaler,
        cat_encoder=cat_encoder,
    )

    # 6. Create DataLoaders
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
