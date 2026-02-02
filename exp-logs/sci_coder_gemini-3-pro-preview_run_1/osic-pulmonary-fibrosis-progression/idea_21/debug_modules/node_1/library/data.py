import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class CTPreprocessor:
    """
    Handles loading DICOM scans, converting to HU, and generating
    Fixed Overlapping Orthogonal Tri-Slabs (Axial & Coronal).
    """

    def __init__(self, cache_dir, img_size=224):
        self.cache_dir = cache_dir
        self.img_size = img_size
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_pixels_hu(self, scans):
        """Converts raw DICOM pixel_array to Hounsfield Units."""
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0 (air is approx -1000)
        # Some scanners use -2000 for padding
        image[image == -2000] = 0

        # Convert to Hounsfield Units (HU)
        intercept = scans[0].RescaleIntercept
        slope = scans[0].RescaleSlope

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def normalize_lung_window(self, img):
        """Applies Lung Window [-1000, 400] and normalizes to [0, 1]."""
        # Standard Lung Window
        L = -600
        W = 1500
        min_hu = L - W // 2  # -1350
        max_hu = L + W // 2  # 150

        # Clip and normalize
        img = np.clip(img, min_hu, max_hu)
        img = (img - min_hu) / (max_hu - min_hu)
        return img

    def generate_tri_slab(self, volume, view="axial"):
        """
        Generates 3-channel RGB image using MIPs of 3 overlapping slabs.
        view: 'axial' (Z-axis) or 'coronal' (Y-axis).
        """
        # Determine axis for depth and slicing
        # Volume is (Z, Y, X)
        if view == "axial":
            # Depth is Z (dim 0)
            depth = volume.shape[0]
            # Slicing along Z, image plane is (Y, X)
            get_slice = lambda d_idx: volume[d_idx, :, :]
            axis_to_mip = 0
        elif view == "coronal":
            # Depth is Y (dim 1)
            depth = volume.shape[1]
            # Slicing along Y, image plane is (Z, X)
            get_slice = lambda d_idx: volume[:, d_idx, :]
            axis_to_mip = 1
        else:
            raise ValueError("View must be 'axial' or 'coronal'")

        # Define 3 overlapping slabs
        # Logic: 0-33%, 33-66%, 66-100% with overlap
        # We use a buffer to create approx 15% overlap relative to slab size
        buffer_ratio = 0.08  # Approx overlap

        s1_end = 0.33 + buffer_ratio
        s2_start = 0.33 - buffer_ratio
        s2_end = 0.66 + buffer_ratio
        s3_start = 0.66 - buffer_ratio

        boundaries = [
            (0, int(s1_end * depth)),
            (int(s2_start * depth), int(s2_end * depth)),
            (int(s3_start * depth), depth),
        ]

        channels = []
        for start, end in boundaries:
            start = max(0, start)
            end = min(depth, end)

            if start >= end:
                # Fallback for very small volumes (single slice)
                idx = min(start, depth - 1)
                if view == "axial":
                    slab = volume[idx : idx + 1, :, :]
                else:
                    slab = volume[:, idx : idx + 1, :]
            else:
                # Extract slab
                if view == "axial":
                    slab = volume[start:end, :, :]
                else:
                    slab = volume[:, start:end, :]

            # Compute MIP
            mip = np.max(slab, axis=axis_to_mip)
            channels.append(mip)

        # Stack to (H, W, 3) or (Z, X, 3)
        img = np.stack(channels, axis=-1)

        # Resize to target size
        img = cv2.resize(
            img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        return img

    def process_patient(self, patient_id, dicom_dir, load_cached_data=True):
        axial_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        coronal_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")

        # 1. Try loading from cache
        if (
            load_cached_data
            and os.path.exists(axial_path)
            and os.path.exists(coronal_path)
        ):
            try:
                img_ax = np.load(axial_path)
                img_cor = np.load(coronal_path)
                return img_ax.astype(np.float32), img_cor.astype(np.float32)
            except Exception:
                pass  # Re-process if corrupt

        # 2. Process from scratch
        # Create blank if directory missing (safety)
        if not os.path.exists(dicom_dir):
            blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            return blank, blank

        # Load DICOMs
        files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not files:
            blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            return blank, blank

        try:
            scans = [pydicom.dcmread(f) for f in files]
            # Sort by ImagePositionPatient Z (index 2)
            scans.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except Exception:
            # Fallback sort by filename or instance number if header is broken
            scans.sort(
                key=lambda x: (
                    float(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
                )
            )

        # Convert to HU
        try:
            volume = self.get_pixels_hu(scans)  # (Z, Y, X)
        except Exception:
            # Fallback for empty or bad DICOMs
            blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            return blank, blank

        # Generate Tri-Slabs
        img_ax = self.generate_tri_slab(volume, view="axial")
        img_cor = self.generate_tri_slab(volume, view="coronal")

        # Normalize
        img_ax = self.normalize_lung_window(img_ax)
        img_cor = self.normalize_lung_window(img_cor)

        # Save to cache
        np.save(axial_path, img_ax.astype(np.float32))
        np.save(coronal_path, img_cor.astype(np.float32))

        return img_ax.astype(np.float32), img_cor.astype(np.float32)


class LungDataset(Dataset):
    def __init__(self, df, preprocessor, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.transforms = transforms
        self.mode = mode

        # Feature Encoding Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # Normalization Constants (Approximate from training stats)
        self.age_mean, self.age_std = 67.0, 7.0
        self.pct_mean, self.pct_std = 77.0, 20.0
        self.week_mean, self.week_std = 31.0, 23.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Determine DICOM directory
        dicom_path = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])

        # Load Images (Cached)
        img_ax, img_cor = self.preprocessor.process_patient(
            patient_id, dicom_path, load_cached_data=True
        )

        # Apply Transforms
        if self.transforms:
            # Apply independently to maintain view integrity
            aug_ax = self.transforms(image=img_ax)["image"]
            aug_cor = self.transforms(image=img_cor)["image"]
        else:
            aug_ax = ToTensorV2()(image=img_ax)["image"]
            aug_cor = ToTensorV2()(image=img_cor)["image"]

        # Extract Tabular Features
        # Handle column name differences between train/test
        week = row["Weeks"] if "Weeks" in row else row["Predict_Week"]
        percent = row["Percent"] if "Percent" in row else row["Baseline_Percent"]
        age = row["Age"] if "Age" in row else row["Baseline_Age"]
        sex = row["Sex"] if "Sex" in row else row["Baseline_Sex"]
        smoke = (
            row["SmokingStatus"]
            if "SmokingStatus" in row
            else row["Baseline_SmokingStatus"]
        )

        # Encode & Normalize
        feat_week = (week - self.week_mean) / self.week_std
        feat_pct = (percent - self.pct_mean) / self.pct_std
        feat_age = (age - self.age_mean) / self.age_std
        feat_sex = float(self.sex_map.get(sex, 0))
        feat_smoke = float(self.smoke_map.get(smoke, 0))

        # Create Tabular Vector: [Week, Percent, Age, Sex, Smoke]
        tabular = torch.tensor(
            [feat_week, feat_pct, feat_age, feat_sex, feat_smoke], dtype=torch.float32
        )

        # Baseline FVC (Anchor)
        # If not present (e.g. raw train), default to 0.0, but get_dataloaders should inject it.
        baseline_fvc = float(row.get("Baseline_FVC", 0.0))
        baseline_week = float(row.get("Baseline_Week", 0.0))

        meta = {
            "Patient_Week": str(row.get("Patient_Week", f"{patient_id}_{week}")),
            "Weeks": float(week),
            "Patient": patient_id,
            "Baseline_FVC": baseline_fvc,
            "Baseline_Week": baseline_week,
        }

        # Target
        target = (
            torch.tensor([row["FVC"]], dtype=torch.float32)
            if "FVC" in row
            else torch.tensor([0.0])
        )

        return {
            "image_axial": aug_ax,
            "image_coronal": aug_cor,
            "tabular": tabular,
            "target": target,
            "meta": meta,
        }


def get_dataloaders(
    debug=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    # 1. Load Dataframes
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if debug:
        train_df = train_df.iloc[: Config.DEBUG_DATA_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_DATA_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_DATA_SIZE]

    # 2. Enrich Train/Val with Baseline Info
    # We define baseline as the measurement at the earliest week for that patient
    def add_baseline(df):
        # Sort by patient and week
        df = df.sort_values(["Patient", "Weeks"])
        # Group by patient and take first entry as baseline
        baseline = df.groupby("Patient").first().reset_index()
        baseline = baseline[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
        )
        # Merge back
        return pd.merge(df, baseline, on="Patient", how="left")

    train_df = add_baseline(train_df)
    val_df = add_baseline(val_df)

    # Test DF already has Baseline_FVC and Baseline_Week from metadata generation

    # 3. Initialize Preprocessor
    preprocessor = CTPreprocessor(cache_dir=Config.CACHE_DIR, img_size=Config.IMG_SIZE)

    # 4. Pre-process all patients (Ensure cache is populated)
    # This avoids race conditions in DataLoader workers
    print("Pre-processing images (checking cache)...")
    all_patients = pd.concat(
        [
            train_df[["Patient", "dicom_dir"]],
            val_df[["Patient", "dicom_dir"]],
            test_df[["Patient", "dicom_dir"]],
        ]
    ).drop_duplicates()

    for _, row in all_patients.iterrows():
        dicom_path = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])
        preprocessor.process_patient(row["Patient"], dicom_path, load_cached_data=True)

    # 5. Define Transforms
    # Spatial augmentation only for training
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                scale_limit=0.05,
                rotate_limit=10,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # 6. Create Datasets
    train_dataset = LungDataset(
        train_df, preprocessor, transforms=train_transform, mode="train"
    )
    val_dataset = LungDataset(
        val_df, preprocessor, transforms=val_transform, mode="val"
    )
    test_dataset = LungDataset(
        test_df, preprocessor, transforms=val_transform, mode="test"
    )

    # 7. Create Loaders
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
