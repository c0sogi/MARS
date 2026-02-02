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

from library.config import Config


class TriSlabGenerator:
    """
    Generates Fixed Overlapping Orthogonal Tri-Slab RGB images from CT volumes.
    Handles DICOM reading, MIP computation, and caching.
    """

    def __init__(self, cache_dir, img_size=224):
        self.cache_dir = cache_dir
        self.img_size = img_size
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_paths(self, patient_id):
        ax_path = os.path.join(self.cache_dir, f"{patient_id}_axial.npy")
        cor_path = os.path.join(self.cache_dir, f"{patient_id}_coronal.npy")
        return ax_path, cor_path

    def load_dicom_volume(self, dicom_dir):
        files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not files:
            return None

        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                # Use InstanceNumber for Z-ordering
                idx = int(dcm.InstanceNumber) if hasattr(dcm, "InstanceNumber") else 0
                slices.append((idx, dcm))
            except Exception:
                continue

        if not slices:
            return None

        # Sort by Z-position
        slices.sort(key=lambda x: x[0])

        # Extract pixel data and convert to Hounsfield Units (HU)
        images = []
        for _, dcm in slices:
            try:
                img = dcm.pixel_array.astype(np.float32)
                intercept = (
                    dcm.RescaleIntercept if hasattr(dcm, "RescaleIntercept") else -1024
                )
                slope = dcm.RescaleSlope if hasattr(dcm, "RescaleSlope") else 1
                img = img * slope + intercept
                images.append(img)
            except:
                continue

        if not images:
            return None

        volume = np.stack(images)  # Shape: (D, H, W)
        return volume

    def preprocess_volume(self, volume):
        # Standard Lung Window: L=-600, W=1500 -> Range [-1350, 150]
        L, W = -600, 1500
        lower = L - W / 2
        upper = L + W / 2

        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower)  # Normalize to [0, 1]
        return volume

    def generate_tri_slab(self, volume, axis=0):
        # axis=0 -> Axial (D, H, W), slicing along D
        # axis=1 -> Coronal (D, H, W) -> transpose to (H, D, W), slicing along H

        if axis == 1:
            # Transpose to make the slicing axis the first dimension
            volume = np.transpose(volume, (1, 0, 2))

        depth = volume.shape[0]
        if depth == 0:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        # Define 3 overlapping slabs with ~15% overlap logic
        # Slab 1: 0 - 40%
        # Slab 2: 30% - 70%
        # Slab 3: 60% - 100%
        idx1 = int(depth * 0.4)
        idx2_start = int(depth * 0.3)
        idx2_end = int(depth * 0.7)
        idx3_start = int(depth * 0.6)

        # Ensure indices are valid
        idx1 = max(1, idx1)
        idx2_end = max(idx2_start + 1, idx2_end)
        idx3_start = min(depth - 1, idx3_start)

        slab1 = volume[:idx1]
        slab2 = volume[idx2_start:idx2_end]
        slab3 = volume[idx3_start:]

        # Compute Maximum Intensity Projection (MIP)
        c1 = np.max(slab1, axis=0) if slab1.shape[0] > 0 else np.zeros_like(volume[0])
        c2 = np.max(slab2, axis=0) if slab2.shape[0] > 0 else np.zeros_like(volume[0])
        c3 = np.max(slab3, axis=0) if slab3.shape[0] > 0 else np.zeros_like(volume[0])

        # Stack to RGB
        img = np.stack([c1, c2, c3], axis=-1)

        # Resize to fixed resolution
        img = cv2.resize(img, (self.img_size, self.img_size))

        return img

    def process_patient(self, patient_id, dicom_dir, load_cached_data=True):
        ax_path, cor_path = self.get_cache_paths(patient_id)

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(ax_path) and os.path.exists(cor_path):
            try:
                img_ax = np.load(ax_path)
                img_cor = np.load(cor_path)
                return img_ax, img_cor
            except:
                pass  # Fallback to recomputing if cache is corrupt

        # 2. Compute from scratch
        volume = self.load_dicom_volume(dicom_dir)

        if volume is None:
            # Return black images if loading fails
            img_ax = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            img_cor = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        else:
            volume = self.preprocess_volume(volume)
            img_ax = self.generate_tri_slab(volume, axis=0)
            img_cor = self.generate_tri_slab(volume, axis=1)

        # 3. Save to cache
        np.save(ax_path, img_ax)
        np.save(cor_path, img_cor)

        return img_ax, img_cor


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mode = mode
        self.transform = transform
        self.processor = TriSlabGenerator(cache_dir, Config.IMG_SIZE)

        # Feature Engineering
        self.prepare_tabular_features()

    def prepare_tabular_features(self):
        # Normalize continuous variables
        # Age: approximate range 30-100
        self.df["Age_norm"] = (self.df["Age"] - 30) / 70.0
        # Percent: approximate range 0-150
        self.df["Percent_norm"] = self.df["Percent"] / 100.0

        # Encode Sex: Female=1, Male=0
        self.df["Sex_enc"] = self.df["Sex"].apply(
            lambda x: 1.0 if x == "Female" else 0.0
        )

        # Encode SmokingStatus: One-Hot
        # Categories: 'Ex-smoker', 'Never smoked', 'Currently smokes'
        self.df["Smoke_Ex"] = (self.df["SmokingStatus"] == "Ex-smoker").astype(float)
        self.df["Smoke_Never"] = (self.df["SmokingStatus"] == "Never smoked").astype(
            float
        )
        self.df["Smoke_Current"] = (
            self.df["SmokingStatus"] == "Currently smokes"
        ).astype(float)

        # Identify Baseline FVC and Week for Anchor
        # For Test set, these are already provided as 'Baseline_FVC' and 'Baseline_Week'
        # For Train/Val, we must compute them from the patient history
        if "Baseline_FVC" not in self.df.columns:
            # We assume the baseline is the measurement closest to Week 0 (or the first measurement)
            # Sort by absolute week number to find the one closest to 0
            baseline_df = self.df.copy()
            baseline_df["Week_Abs"] = baseline_df["Weeks"].abs()
            baseline_df = baseline_df.sort_values(["Patient", "Week_Abs"])

            # Extract first entry per patient
            baseline_fvc_map = baseline_df.groupby("Patient")["FVC"].first()
            baseline_week_map = baseline_df.groupby("Patient")["Weeks"].first()

            self.df["Baseline_FVC"] = self.df["Patient"].map(baseline_fvc_map)
            self.df["Baseline_Week"] = self.df["Patient"].map(baseline_week_map)

        # Fallback for safety
        if "Baseline_Week" not in self.df.columns:
            self.df["Baseline_Week"] = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        dicom_dir = os.path.join(Config.INPUT_ROOT, row["dicom_dir"])
        img_ax, img_cor = self.processor.process_patient(
            patient_id, dicom_dir, load_cached_data=True
        )

        # 2. Augmentations
        if self.transform:
            # Pass both images to albumentations
            # Note: additional_targets={'image_cor': 'image'} ensures consistent transforms if configured
            res = self.transform(image=img_ax, image_cor=img_cor)
            img_ax = res["image"]
            img_cor = res["image_cor"]
        else:
            # Manual conversion if no transform provided
            img_ax = torch.from_numpy(img_ax.transpose(2, 0, 1))
            img_cor = torch.from_numpy(img_cor.transpose(2, 0, 1))

        # 3. Tabular Vector
        # Order: [Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Percent]
        tab_vec = np.array(
            [
                row["Age_norm"],
                row["Sex_enc"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
                row["Percent_norm"],
            ],
            dtype=np.float32,
        )

        # 4. Targets and Anchors
        # Handle test set column naming differences
        fvc = row["FVC"] if "FVC" in row else 0.0
        weeks = row["Weeks"] if "Weeks" in row else row["Predict_Week"]

        base_fvc = row["Baseline_FVC"]
        base_week = row["Baseline_Week"]

        return {
            "img_ax": img_ax,
            "img_cor": img_cor,
            "tabular": torch.from_numpy(tab_vec),
            "weeks": torch.tensor(weeks, dtype=torch.float32),
            "fvc": torch.tensor(fvc, dtype=torch.float32),
            "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            "base_week": torch.tensor(base_week, dtype=torch.float32),
            "patient_id": patient_id,
        }


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms.
    Train: Spatial augmentations (Flip, Shift, Rotate). No intensity changes.
    Val/Test: Normalization only.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_cor": "image"},
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                ToTensorV2(),
            ],
            additional_targets={"image_cor": "image"},
        )


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    # Load generated metadata
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    # Initialize Datasets
    train_ds = OSICDataset(
        train_df, Config.CACHE_DIR, mode="train", transform=get_transforms("train")
    )
    val_ds = OSICDataset(
        val_df, Config.CACHE_DIR, mode="val", transform=get_transforms("val")
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Rename baseline columns to match standard feature names for the Dataset class
    rename_map = {
        "Baseline_Age": "Age",
        "Baseline_Sex": "Sex",
        "Baseline_SmokingStatus": "SmokingStatus",
        "Baseline_Percent": "Percent",
    }
    test_df = test_df.rename(columns=rename_map)

    test_ds = OSICDataset(
        test_df, Config.CACHE_DIR, mode="test", transform=get_transforms("test")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, test_df
