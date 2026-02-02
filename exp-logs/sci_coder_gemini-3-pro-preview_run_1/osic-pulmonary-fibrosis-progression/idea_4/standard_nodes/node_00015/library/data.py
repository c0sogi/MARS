import os
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Attempt to import pydicom. If missing, we cannot process images,
# but we ensure the code structure remains valid.
try:
    import pydicom
except ImportError:
    pydicom = None

from library.config import Config, seed_everything


class CTPreprocessor:
    """
    Handles loading of DICOM scans, generation of Tri-Slab images (Axial & Coronal),
    and caching of processed arrays to disk.
    """

    def __init__(self, cache_dir=Config.CACHE_DIR, img_size=Config.IMG_SIZE):
        self.cache_dir = cache_dir
        self.img_size = img_size
        os.makedirs(self.cache_dir, exist_ok=True)

    def _read_dicom_scan(self, dicom_dir):
        """
        Reads and sorts DICOM files from a directory.
        Returns a 3D numpy array (D, H, W) in Hounsfield Units.
        """
        if pydicom is None:
            # Fallback if pydicom is not installed in the environment
            return np.zeros((10, 512, 512), dtype=np.float32)

        files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
        if not files:
            return np.zeros((10, 512, 512), dtype=np.float32)

        # Read files
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(f)
                # Ensure pixel data exists
                if hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except Exception:
                continue

        if not slices:
            return np.zeros((10, 512, 512), dtype=np.float32)

        # Sort by ImagePositionPatient Z (if available) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            slices.sort(key=lambda x: int(x.InstanceNumber))

        # Extract pixel data and convert to HU
        images = []
        for s in slices:
            img = s.pixel_array.astype(np.float32)
            slope = getattr(s, "RescaleSlope", 1)
            intercept = getattr(s, "RescaleIntercept", -1024)
            img = img * slope + intercept
            images.append(img)

        # Stack to (D, H, W)
        volume = np.stack(images)
        return volume

    def _process_volume(self, volume):
        """
        Generates Axial and Coronal Tri-Slabs from the volume.
        """
        # Clip to lung window [-1000, 400] and normalize to [0, 1]
        volume = np.clip(volume, -1000, 400)
        volume = (volume + 1000) / 1400.0

        D, H, W = volume.shape

        # --- Axial Tri-Slab (Top-down view) ---
        # Split Depth (D) into 3 overlapping slabs
        if D < 3:
            # Pad if too few slices
            pad = 3 - D
            volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), "constant")
            D = 3

        chunk_size = D // 3
        # Simple splitting
        s1 = volume[:chunk_size]
        s2 = volume[chunk_size : 2 * chunk_size]
        s3 = volume[2 * chunk_size :]

        # Maximum Intensity Projection (MIP)
        mip1 = np.max(s1, axis=0) if s1.size > 0 else np.zeros((H, W))
        mip2 = np.max(s2, axis=0) if s2.size > 0 else np.zeros((H, W))
        mip3 = np.max(s3, axis=0) if s3.size > 0 else np.zeros((H, W))

        axial_img = np.stack([mip1, mip2, mip3], axis=-1)  # (H, W, 3)

        # --- Coronal Tri-Slab (Front-back view) ---
        # Volume is (D, H, W). Coronal view looks at (D, W) along H (Y-axis).
        # Split Height (H) into 3 slabs (Anterior-Posterior splits)
        h_chunk = H // 3

        c_s1 = volume[:, :h_chunk, :]
        c_s2 = volume[:, h_chunk : 2 * h_chunk, :]
        c_s3 = volume[:, 2 * h_chunk :, :]

        # MIP along axis 1 (H)
        c_mip1 = np.max(c_s1, axis=1) if c_s1.size > 0 else np.zeros((D, W))
        c_mip2 = np.max(c_s2, axis=1) if c_s2.size > 0 else np.zeros((D, W))
        c_mip3 = np.max(c_s3, axis=1) if c_s3.size > 0 else np.zeros((D, W))

        coronal_img = np.stack([c_mip1, c_mip2, c_mip3], axis=-1)  # (D, W, 3)

        # Resize both to target size
        # Note: Coronal image (D, W) might have very small D, resizing stretches it.
        # This stretching is actually beneficial as it normalizes the depth representation.
        axial_resized = cv2.resize(
            axial_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )
        coronal_resized = cv2.resize(
            coronal_img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
        )

        return axial_resized.astype(np.float32), coronal_resized.astype(np.float32)

    def get_patient_data(self, patient_id, dicom_dir, load_cached_data=True):
        """
        Returns (axial, coronal) numpy arrays.
        Uses caching to speed up subsequent epochs.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        if load_cached_data and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                return data[0], data[1]
            except Exception:
                pass  # Failed to load, recompute

        # Compute
        full_dicom_path = os.path.join(Config.INPUT_ROOT, dicom_dir)
        volume = self._read_dicom_scan(full_dicom_path)
        axial, coronal = self._process_volume(volume)

        # Save to cache
        try:
            np.save(cache_path, np.array([axial, coronal]))
        except Exception:
            pass  # Ignore save errors (e.g. disk full)

        return axial, coronal


class FibrosisDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, load_cache=True):
        """
        Args:
            df: DataFrame containing metadata.
            mode: 'train', 'val', or 'test'.
            transform: Albumentations transform.
            load_cache: Boolean to use cached images.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.load_cache = load_cache
        self.preprocessor = CTPreprocessor()

        # Mappings
        self.sex_map = Config.SEX_MAP
        self.smoke_map = Config.SMOKING_MAP

        # Pre-calculate baselines for Train/Val sets
        self.patient_baselines = {}
        if self.mode in ["train", "val"]:
            self._compute_baselines()

    def _compute_baselines(self):
        """
        Identifies the baseline visit (closest to Week 0) for each patient
        to extract Baseline_FVC and Baseline_Percent.
        """
        patients = self.df["Patient"].unique()
        for p in patients:
            p_data = self.df[self.df["Patient"] == p]
            # Find row with min absolute weeks (closest to baseline)
            # We use this row's FVC and Percent as the patient's baseline stats
            idx_min = p_data["Weeks"].abs().idxmin()
            baseline_row = p_data.loc[idx_min]

            self.patient_baselines[p] = {
                "Weeks": int(baseline_row["Weeks"]),
                "FVC": float(baseline_row["FVC"]),
                "Percent": float(baseline_row["Percent"]),
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Images
        dicom_dir = row["dicom_dir"]
        axial, coronal = self.preprocessor.get_patient_data(
            patient_id, dicom_dir, self.load_cache
        )

        # 2. Augmentation
        if self.transform:
            # Apply independent transforms to views
            res_ax = self.transform(image=axial)["image"]
            res_cor = self.transform(image=coronal)["image"]
            axial = res_ax
            coronal = res_cor
        else:
            # Convert to tensor (C, H, W)
            axial = torch.tensor(axial.transpose(2, 0, 1), dtype=torch.float32)
            coronal = torch.tensor(coronal.transpose(2, 0, 1), dtype=torch.float32)

        # 3. Tabular Features & Meta
        # Determine Baseline Percent and Week Delta
        if self.mode in ["train", "val"]:
            baseline_info = self.patient_baselines.get(patient_id, {})
            base_pct = baseline_info.get("Percent", 0.0)
            base_fvc = baseline_info.get("FVC", 2000.0)
            base_week = baseline_info.get("Weeks", 0)

            current_week = int(row["Weeks"])
            week_delta = current_week - base_week
            true_fvc = float(row["FVC"])

        else:
            # Test mode: Metadata already contains baseline info
            base_pct = float(row["Baseline_Percent"])
            base_fvc = float(row["Baseline_FVC"])
            base_week = int(row["Baseline_Week"])
            current_week = int(row["Predict_Week"])
            week_delta = current_week - base_week
            true_fvc = 0.0  # Dummy

        # Normalize Features
        age = float(row["Age"] if "Age" in row else row["Baseline_Age"]) / 100.0
        sex_raw = row["Sex"] if "Sex" in row else row["Baseline_Sex"]
        smoke_raw = (
            row["SmokingStatus"]
            if "SmokingStatus" in row
            else row["Baseline_SmokingStatus"]
        )

        sex = self.sex_map.get(sex_raw, 0)
        smoke = self.smoke_map.get(smoke_raw, 0)
        base_pct_norm = base_pct / 100.0

        # Tabular vector: [Age, Base_Percent, Sex, Smoking]
        tabular = torch.tensor(
            [age, base_pct_norm, float(sex), float(smoke)], dtype=torch.float32
        )

        # 4. Return
        if self.mode in ["train", "val"]:
            targets = {
                "fvc": torch.tensor(true_fvc, dtype=torch.float32),
                "base_pct": torch.tensor(base_pct, dtype=torch.float32),  # For Aux Loss
                "week_delta": torch.tensor(week_delta, dtype=torch.float32),
                "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
            }
            return (axial, coronal, tabular), targets
        else:
            meta = {
                "week_delta": torch.tensor(week_delta, dtype=torch.float32),
                "base_fvc": torch.tensor(base_fvc, dtype=torch.float32),
                "patient_week": row["Patient_Week"],
            }
            return (axial, coronal, tabular), meta


def get_dataloaders(train_df, val_df, test_df, batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles DEBUG mode by sampling data.
    """
    # Debug Sampling
    if Config.DEBUG:
        train_df = train_df.head(Config.DEBUG_SIZE)
        val_df = val_df.head(Config.DEBUG_SIZE)
        test_df = test_df.head(Config.DEBUG_SIZE)

    # Transforms
    # Spatial augmentation only for training
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose([ToTensorV2()])

    # Datasets
    train_ds = FibrosisDataset(train_df, mode="train", transform=train_transform)
    val_ds = FibrosisDataset(val_df, mode="val", transform=val_transform)
    test_ds = FibrosisDataset(test_df, mode="test", transform=val_transform)

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
