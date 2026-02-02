import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import seed_everything

# Constants for Normalization
MEAN_AGE = 67.0
STD_AGE = 7.0
MEAN_PERCENT = 77.0
STD_PERCENT = 20.0
MEAN_FVC = 2700.0
STD_FVC = 800.0


def load_dicom_volume(path):
    """
    Reads a directory of DICOM files and returns a numpy 3D array (D, H, W).
    Sorts by InstanceNumber or SliceLocation.
    """
    if not os.path.exists(path):
        return np.zeros((10, 224, 224), dtype=np.float32)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((10, 224, 224), dtype=np.float32)

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(os.path.join(path, f))
            slices.append(dcm)
        except:
            continue

    if not slices:
        return np.zeros((10, 224, 224), dtype=np.float32)

    # Sort slices
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except:
            pass  # Keep original order if sorting fails

    # Stack images
    images = []
    for s in slices:
        try:
            img = s.pixel_array.astype(np.float32)
            # Intercept/Slope adjustment if present
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = img * slope + intercept
            images.append(img)
        except:
            continue

    if not images:
        return np.zeros((10, 224, 224), dtype=np.float32)

    volume = np.stack(images)  # (D, H, W)
    return volume


def generate_tri_slab(volume, axis=0, target_size=(224, 224)):
    """
    Generates a 3-channel image using Fixed Overlapping Orthogonal Tri-Slabs.

    Args:
        volume: 3D numpy array (D, H, W) or (D, Y, X)
        axis: 0 for Axial (split D), 1 for Coronal (split H/Y)
        target_size: tuple (H, W) for resizing

    Returns:
        numpy array (H, W, 3) normalized to 0-1
    """
    # Ensure volume is at least 3D
    if volume.ndim != 3:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

    # If Coronal (axis=1), we transpose so the split axis is dimension 0
    if axis == 1:
        # Original: (D, H, W). We want to split H.
        # Transpose to (H, D, W) so we can reuse logic
        volume = np.transpose(volume, (1, 0, 2))

    depth = volume.shape[0]
    if depth < 3:
        # Handle edge case with very few slices by repeating
        # Just return resized MIP of whole volume repeated
        mip = np.max(volume, axis=0)
        mip = cv2.resize(mip, target_size)
        # Normalize
        mi_min, mi_max = -1000, 400
        mip = np.clip(mip, mi_min, mi_max)
        mip = (mip - mi_min) / (mi_max - mi_min)
        return np.stack([mip] * 3, axis=-1)

    # Define slab boundaries with overlap
    # Slabs: 0-38%, 28-71%, 61-100% (Approx 33% core + overlap)
    p1 = int(depth * 0.38)
    p2_start = int(depth * 0.28)
    p2_end = int(depth * 0.71)
    p3_start = int(depth * 0.61)

    slab1 = volume[:p1, :, :]
    slab2 = volume[p2_start:p2_end, :, :]
    slab3 = volume[p3_start:, :, :]

    # Compute MIPs
    # If slab is empty (shouldn't happen with logic above but safety check), take whole volume
    m1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.max(volume, axis=0)
    m2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.max(volume, axis=0)
    m3 = np.max(slab3, axis=0) if slab3.shape[0] > 0 else np.max(volume, axis=0)

    # Resize
    m1 = cv2.resize(m1, target_size)
    m2 = cv2.resize(m2, target_size)
    m3 = cv2.resize(m3, target_size)

    # Stack
    img = np.stack([m1, m2, m3], axis=-1)  # (H, W, 3)

    # Normalize (Lung Window: -1000 to 400 approx)
    # Standard lung window center -600, width 1500 -> -1350 to 150
    # Let's use a broad range to capture density: -1000 to 600
    lower = -1000
    upper = 600
    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    return img.astype(np.float32)


def preprocess_and_cache(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Loads DICOM, generates Axial and Coronal Tri-Slabs, and caches them.
    """
    input_root = "./input"
    full_dicom_path = os.path.join(input_root, dicom_dir)

    os.makedirs(cache_dir, exist_ok=True)

    f_ax = os.path.join(cache_dir, f"{patient_id}_axial.npy")
    f_cor = os.path.join(cache_dir, f"{patient_id}_coronal.npy")

    if load_cached_data and os.path.exists(f_ax) and os.path.exists(f_cor):
        try:
            img_ax = np.load(f_ax)
            img_cor = np.load(f_cor)
            return img_ax, img_cor
        except:
            pass  # Fallback to re-process

    # Process
    vol = load_dicom_volume(full_dicom_path)

    # Axial (Axis 0)
    img_ax = generate_tri_slab(vol, axis=0)
    # Coronal (Axis 1)
    img_cor = generate_tri_slab(vol, axis=1)

    # Save
    np.save(f_ax, img_ax)
    np.save(f_cor, img_cor)

    return img_ax, img_cor


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        mode="train",
        transform=None,
        cache_dir="./working/idea_65/",
        load_cache=True,
    ):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.load_cache = load_cache

        # Pre-process Tabular Features
        # Sex: Male=0, Female=1
        self.df["Sex_Code"] = self.df["Sex"].map({"Male": 0, "Female": 1})

        # Smoking: Ex-smoker, Never smoker, Currently smokes
        # One-hot encoding manually to ensure fixed order
        self.df["Smoke_Ex"] = (self.df["SmokingStatus"] == "Ex-smoker").astype(float)
        self.df["Smoke_Never"] = (self.df["SmokingStatus"] == "Never smoker").astype(
            float
        )
        self.df["Smoke_Cur"] = (self.df["SmokingStatus"] == "Currently smokes").astype(
            float
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Images
        # Use dicom_dir from metadata
        dicom_dir = row["dicom_dir"]
        img_ax, img_cor = preprocess_and_cache(
            pid, dicom_dir, self.cache_dir, self.load_cache
        )

        # 2. Augmentations (Spatial Only)
        if self.transform:
            # Albumentations expects HWC uint8 or float
            # Our images are float 0-1.
            res_ax = self.transform(image=img_ax)["image"]
            res_cor = self.transform(image=img_cor)["image"]
            img_ax = res_ax
            img_cor = res_cor
        else:
            # To Tensor
            img_ax = torch.tensor(img_ax.transpose(2, 0, 1), dtype=torch.float32)
            img_cor = torch.tensor(img_cor.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Tabular Data
        # Features: [Age_norm, Sex, Smoke_Ex, Smoke_Never, Smoke_Cur, Percent_norm, Base_FVC_norm]
        # Note: We use Baseline info for the "Context" and "Prior".
        # For train/val, the row contains the visit info, but we need the BASELINE info for the input features.
        # The dataframe passed to this class should ideally have Baseline columns merged.

        if "Baseline_Age" in row:
            # Test set structure or merged train structure
            age = row["Baseline_Age"]
            pct = row["Baseline_Percent"]
            base_fvc = row["Baseline_FVC"]
            # Sex/Smoking are usually constant per patient, but use Baseline_ prefix if available
            sex = row["Sex_Code"]  # Assuming Sex doesn't change
            s_ex = row["Smoke_Ex"]
            s_nev = row["Smoke_Never"]
            s_cur = row["Smoke_Cur"]
        else:
            # Fallback for standard train row if not pre-merged (though we will merge in get_dataloaders)
            age = row["Age"]
            pct = row["Percent"]
            # For training, we need to pass the Baseline FVC as a feature,
            # but the row FVC is the target.
            # We assume 'Base_FVC' column exists (created in get_dataloaders).
            base_fvc = row["Base_FVC"]
            sex = row["Sex_Code"]
            s_ex = row["Smoke_Ex"]
            s_nev = row["Smoke_Never"]
            s_cur = row["Smoke_Cur"]

        # Normalize
        age_norm = (age - MEAN_AGE) / STD_AGE
        pct_norm = (pct - MEAN_PERCENT) / STD_PERCENT
        base_fvc_norm = (base_fvc - MEAN_FVC) / STD_FVC

        tabular = torch.tensor(
            [age_norm, sex, s_ex, s_nev, s_cur, pct_norm, base_fvc_norm],
            dtype=torch.float32,
        )

        # 4. Target and Time
        if self.mode in ["train", "val"]:
            target_fvc = row["FVC"]
            # Weeks relative to baseline
            # If 'Base_Week' exists, subtract. Else assume 'Weeks' is already relative (dataset desc says Weeks is relative)
            # In train.csv, Weeks is relative to baseline CT.
            weeks = row["Weeks"]
        else:
            # Test mode
            target_fvc = 0.0  # Dummy
            # For test, we predict at 'Predict_Week'
            weeks = row["Predict_Week"] - row["Baseline_Week"]  # Relative time

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": tabular,
            "target": torch.tensor(target_fvc, dtype=torch.float32),
            "weeks": torch.tensor(weeks, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "patient_id": pid,
        }


def get_dataloaders(batch_size=16, num_workers=2, load_cache=True, debug=False):
    """
    Prepares DataFrames and returns DataLoaders.
    """
    metadata_dir = "./metadata"
    train_df = pd.read_csv(os.path.join(metadata_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(metadata_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(metadata_dir, "test.csv"))

    if debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:20]
        test_df = test_df.iloc[:20]

    # --- Preprocessing for Training Data ---
    # We need to identify the Baseline FVC for each patient in Train/Val
    # Baseline is the visit where Weeks is closest to 0 (or min weeks)
    # We create a 'Base_FVC' column for every row of that patient

    def add_baseline_info(df):
        # Group by patient, find row with min abs(Weeks)
        # Create a mapping Patient -> Base_FVC
        baseline_map = {}
        for pid, group in df.groupby("Patient"):
            # Find visit closest to week 0
            idx_min = group["Weeks"].abs().idxmin()
            base_fvc = group.loc[idx_min, "FVC"]
            baseline_map[pid] = base_fvc

        df["Base_FVC"] = df["Patient"].map(baseline_map)
        return df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # Test df already has 'Baseline_FVC' from metadata generation

    # --- Augmentations ---
    # Spatial only: Flips, Shifts, Rotations. No intensity changes.
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
            A.Normalize(
                mean=(0, 0, 0), std=(1, 1, 1)
            ),  # Just to ensure float32 conversion if needed, but we do manual normalization in generate
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), ToTensorV2()]
    )

    # --- Datasets ---
    train_ds = OSICDataset(
        train_df, mode="train", transform=train_transform, load_cache=load_cache
    )
    val_ds = OSICDataset(
        val_df, mode="val", transform=val_transform, load_cache=load_cache
    )
    test_ds = OSICDataset(
        test_df, mode="test", transform=val_transform, load_cache=load_cache
    )

    # --- Loaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
