import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom; strict dependency for DICOM reading
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config
from library.utils import seed_everything

# Dataset Statistics for Normalization
MEAN_AGE = 67.0
STD_AGE = 7.0
MEAN_PERCENT = 77.0
STD_PERCENT = 20.0


def get_img(path):
    """
    Loads a DICOM file, converts to Hounsfield Units (HU), applies Lung Windowing,
    and normalizes to 0-255 uint8.
    """
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files.")

    d = pydicom.dcmread(path)

    # Convert to Hounsfield Units (HU)
    intercept = getattr(d, "RescaleIntercept", -1024)
    slope = getattr(d, "RescaleSlope", 1)

    img = d.pixel_array.astype(np.float32)
    img = img * slope + intercept

    # Apply Lung Window: Width 1500, Level -600
    # Range: [-1350, 150]
    window_center = -600
    window_width = 1500
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    # Normalize to 0-1 then scale to 0-255
    img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)

    return img


def generate_tri_slab(volume, axis_idx, target_size=(224, 224)):
    """
    Generates a 3-channel Tri-Slab MIP image from a 3D volume.
    Splits the specified axis into 3 overlapping slabs and computes MIP for each.

    Args:
        volume: 3D numpy array (D, H, W)
        axis_idx: Axis to collapse (0 for Axial, 1 for Coronal)
        target_size: Tuple (H, W) for resizing
    """
    depth = volume.shape[axis_idx]

    # Define overlapping slab ranges (approx 0-40%, 30-70%, 60-100%)
    starts = [0.0, 0.30, 0.60]
    ends = [0.40, 0.70, 1.0]

    channels = []

    for s, e in zip(starts, ends):
        idx_start = int(s * depth)
        idx_end = int(e * depth)

        # Ensure valid slice range
        idx_end = max(idx_end, idx_start + 1)
        idx_end = min(idx_end, depth)

        # Slice and MIP
        if axis_idx == 0:
            # Axial: Split Depth (0), MIP along Depth
            slab = volume[idx_start:idx_end, :, :]
            mip = np.max(slab, axis=0)
        elif axis_idx == 1:
            # Coronal: Split Height (1), MIP along Height
            slab = volume[:, idx_start:idx_end, :]
            mip = np.max(slab, axis=1)
        else:
            # Fallback (Sagittal)
            slab = volume[:, :, idx_start:idx_end]
            mip = np.max(slab, axis=2)

        channels.append(mip)

    # Stack to create (H, W, 3) image
    img = np.stack(channels, axis=-1)

    # Resize to fixed resolution
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=Config.CACHE_DIR):
        self.df = df.copy()
        self.mode = mode
        self.cache_dir = cache_dir

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Preprocess metadata to unify Train/Test structures
        self._prepare_metadata()

        # Define Augmentations (Spatial only, no intensity changes)
        if self.mode == "train" and Config.USE_AUGMENTATION:
            self.transforms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _prepare_metadata(self):
        """
        Unifies column names and computes Baseline features for the training set.
        """
        # 1. Standardize Target Week and Baseline Info
        if "Predict_Week" in self.df.columns:
            # Test Set: Has Predict_Week, Baseline_FVC, etc.
            self.df["Weeks"] = self.df["Predict_Week"]
        else:
            # Train Set: Has Weeks, FVC. Needs Baseline extraction.
            # Group by Patient to find the visit closest to Week 0 (Baseline)
            patients = self.df["Patient"].unique()
            baseline_rows = []

            for p in patients:
                p_df = self.df[self.df["Patient"] == p]
                # Find row with minimum absolute week number
                idx = p_df["Weeks"].abs().idxmin()
                row = p_df.loc[idx]

                baseline_rows.append(
                    {
                        "Patient": p,
                        "Baseline_Week": row["Weeks"],
                        "Baseline_FVC": row["FVC"],
                        "Baseline_Percent": row["Percent"],
                        "Baseline_Age": row["Age"],
                        "Baseline_Sex": row["Sex"],
                        "Baseline_SmokingStatus": row["SmokingStatus"],
                    }
                )

            baseline_df = pd.DataFrame(baseline_rows)
            self.df = pd.merge(self.df, baseline_df, on="Patient", how="left")

        # 2. Compute Time Delta (Input to parametric model)
        self.df["Time_Delta"] = self.df["Weeks"] - self.df["Baseline_Week"]

        # 3. Encode Categorical Features
        # Sex: Male=0, Female=1
        self.df["Sex_Enc"] = (
            self.df["Baseline_Sex"].map({"Male": 0, "Female": 1}).fillna(0)
        )

        # Smoking: One-Hot Encoding
        self.df["Smoke_Ex"] = (self.df["Baseline_SmokingStatus"] == "Ex-smoker").astype(
            float
        )
        self.df["Smoke_Never"] = (
            self.df["Baseline_SmokingStatus"] == "Never smoked"
        ).astype(float)
        self.df["Smoke_Current"] = (
            self.df["Baseline_SmokingStatus"] == "Currently smokes"
        ).astype(float)

        # 4. Normalize Numerical Features
        self.df["Age_Norm"] = (self.df["Baseline_Age"] - MEAN_AGE) / STD_AGE
        self.df["Percent_Norm"] = (
            self.df["Baseline_Percent"] - MEAN_PERCENT
        ) / STD_PERCENT

    def __len__(self):
        return len(self.df)

    def _process_images(self, patient_id, dicom_dir):
        """
        Handles image loading, processing, and caching.
        """
        cache_ax = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cache_cor = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try Loading from Cache
        if os.path.exists(cache_ax) and os.path.exists(cache_cor):
            try:
                img_ax = np.load(cache_ax)
                img_cor = np.load(cache_cor)
                return img_ax, img_cor
            except Exception:
                pass  # Cache corrupt, recompute

        # 2. Process from Scratch
        full_path = os.path.join(Config.INPUT_DIR, dicom_dir)
        files = glob.glob(os.path.join(full_path, "*.dcm"))

        # Handle missing or empty directories gracefully
        if not files:
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            return img_ax, img_cor

        # Sort slices by Instance Number to reconstruct 3D volume correctly
        slices = []
        for f in files:
            try:
                d = pydicom.dcmread(f, stop_before_pixels=True)
                slices.append((int(d.InstanceNumber), f))
            except:
                continue
        slices.sort(key=lambda x: x[0])

        # Load pixel data
        vol_slices = []
        for _, f_path in slices:
            try:
                img = get_img(f_path)
                vol_slices.append(img)
            except:
                continue

        if not vol_slices:
            img_ax = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            img_cor = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            return img_ax, img_cor

        volume = np.stack(vol_slices, axis=0)  # Shape: (D, H, W)

        # Generate Tri-Slab Views
        img_ax = generate_tri_slab(volume, 0, (Config.IMG_SIZE, Config.IMG_SIZE))
        img_cor = generate_tri_slab(volume, 1, (Config.IMG_SIZE, Config.IMG_SIZE))

        # 3. Save to Cache
        np.save(cache_ax, img_ax)
        np.save(cache_cor, img_cor)

        return img_ax, img_cor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Load Images
        img_ax, img_cor = self._process_images(patient_id, row["dicom_dir"])

        # Apply Transforms
        aug_ax = self.transforms(image=img_ax)["image"]
        aug_cor = self.transforms(image=img_cor)["image"]

        # Prepare Tabular Data
        # Normalized vector for MLP embedding
        # [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Percent] -> Dim 6
        tab_norm = np.array(
            [
                row["Age_Norm"],
                row["Sex_Enc"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["Percent_Norm"],
            ],
            dtype=np.float32,
        )

        # Raw vector for Skip Connection and Parametric Scaling
        # [Baseline_FVC, Baseline_Percent, Baseline_Age]
        tab_raw = np.array(
            [row["Baseline_FVC"], row["Baseline_Percent"], row["Baseline_Age"]],
            dtype=np.float32,
        )

        # Time Delta
        time_delta = np.array([row["Time_Delta"]], dtype=np.float32)

        # Target FVC
        target = np.array([row["FVC"]], dtype=np.float32)

        return {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular_norm": torch.from_numpy(tab_norm),
            "tabular_raw": torch.from_numpy(tab_raw),
            "time_delta": torch.from_numpy(time_delta),
            "target": torch.from_numpy(target),
            "patient_week": row.get(
                "Patient_Week", f"{patient_id}_{int(row['Weeks'])}"
            ),
        }


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
):
    """
    Factory function to create Train and Validation DataLoaders.
    """
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    train_ds = OSICDataset(train_df, mode="train")
    val_ds = OSICDataset(val_df, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=Config.BATCH_SIZE):
    """
    Factory function to create Test DataLoader.
    """
    test_df = pd.read_csv(Config.TEST_CSV)
    test_ds = OSICDataset(test_df, mode="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader
