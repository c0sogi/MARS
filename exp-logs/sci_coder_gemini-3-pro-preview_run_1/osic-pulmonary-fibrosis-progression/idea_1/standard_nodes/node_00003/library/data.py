import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from library.config import Config

# Attempt to import pydicom for DICOM handling
# If missing, the code will fallback to generating empty image placeholders
try:
    import pydicom
except ImportError:
    pydicom = None


class DataProcessor:
    """
    Handles image preprocessing: DICOM loading, MIP generation, windowing, and caching.
    """

    def __init__(self):
        self.img_size = Config.IMG_SIZE
        self.slices_percentage = Config.SLICES_PERCENTAGE
        self.hu_min = Config.HU_MIN
        self.hu_max = Config.HU_MAX
        self.cache_dir = Config.CACHE_DIR

    def get_img_path(self, patient_id, dataset_type):
        """Constructs the path to the patient's DICOM directory."""
        if dataset_type == "train" or dataset_type == "val":
            return os.path.join(Config.TRAIN_IMG_DIR, patient_id)
        else:
            return os.path.join(Config.TEST_IMG_DIR, patient_id)

    def load_scan(self, path):
        """Loads DICOM slices from a directory."""
        if pydicom is None:
            return []

        slices = []
        if not os.path.exists(path):
            return []

        for f in os.listdir(path):
            if f.endswith(".dcm"):
                try:
                    s = pydicom.dcmread(os.path.join(path, f))
                    slices.append(s)
                except Exception:
                    continue

        # Sort slices by InstanceNumber to ensure correct 3D reconstruction
        slices.sort(
            key=lambda x: float(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
        )
        return slices

    def get_pixels_hu(self, scans):
        """Converts raw pixel data to Hounsfield Units (HU)."""
        if not scans:
            return np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        valid_slices = []
        valid_scans = []

        for s in scans:
            try:
                # Attempt to access pixel array (triggers decompression)
                arr = s.pixel_array.astype(np.float32)
                valid_slices.append(arr)
                valid_scans.append(s)
            except Exception:
                # Skip slices that cannot be decompressed (e.g. missing dependencies)
                continue

        if not valid_slices:
            return np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        # Stack slices into 3D array
        image = np.stack(valid_slices)

        # Convert to HU using RescaleSlope and RescaleIntercept from first valid scan
        intercept = (
            valid_scans[0].RescaleIntercept
            if hasattr(valid_scans[0], "RescaleIntercept")
            else -1024
        )
        slope = (
            valid_scans[0].RescaleSlope
            if hasattr(valid_scans[0], "RescaleSlope")
            else 1
        )

        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.float32)

        image += np.float32(intercept)
        return image

    def generate_mip(self, patient_id, dataset_type, load_cached_data=True):
        """
        Generates Maximum Intensity Projection (MIP) image.
        Checks cache first; if missing, computes from DICOMs and saves to cache.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # 1. Attempt to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Cache load failed, recompute

        # 2. Compute MIP from scratch
        dicom_dir = self.get_img_path(patient_id, dataset_type)
        scans = self.load_scan(dicom_dir)

        if not scans:
            # Fallback for missing data or missing pydicom: Black image
            mip = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        else:
            # Use middle percentage of slices to avoid noise from top/bottom
            n_slices = len(scans)
            if n_slices > 1:
                start = int(n_slices * (1 - self.slices_percentage) / 2)
                end = int(n_slices * (1 + self.slices_percentage) / 2)
                scans = scans[start:end]

            # Get 3D volume in HU
            image_3d = self.get_pixels_hu(scans)

            # Compute MIP: Max value along the depth axis (axis 0)
            if image_3d.shape[0] > 0:
                mip = np.max(image_3d, axis=0)
            else:
                mip = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            # Resize to target resolution
            if mip.shape[0] != self.img_size or mip.shape[1] != self.img_size:
                mip = cv2.resize(mip, (self.img_size, self.img_size))

            # Windowing (Clip to lung window) and Normalization [0, 1]
            mip = np.clip(mip, self.hu_min, self.hu_max)
            mip = (mip - self.hu_min) / (self.hu_max - self.hu_min)

        # Save processed image to cache
        np.save(cache_path, mip)

        return mip


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Function Decline Prediction.
    Combines MIP images with clinical tabular data.
    """

    def __init__(self, df, dataset_type="train", load_cached_data=True):
        self.df = df.reset_index(drop=True)
        self.dataset_type = dataset_type  # 'train', 'val', 'test'
        self.load_cached_data = load_cached_data
        self.processor = DataProcessor()

        # Mappings for categorical features
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

        # For training/validation, we need to identify the baseline FVC for each patient
        # (The FVC at the time of the CT scan, or the earliest recorded FVC)
        if dataset_type != "test":
            self._compute_baselines()

    def _compute_baselines(self):
        """
        Determines the Baseline FVC and Baseline Percent for each patient in the training set.
        Assumes the earliest visit (min Weeks) represents the baseline.
        """
        patient_baselines = {}
        patient_percent = {}

        # Group by patient to find their specific baseline
        for pid, group in self.df.groupby("Patient"):
            # Sort by Weeks to find the initial visit
            group = group.sort_values("Weeks")
            baseline_row = group.iloc[0]

            patient_baselines[pid] = baseline_row["FVC"]
            patient_percent[pid] = baseline_row["Percent"]

        # Map back to the main dataframe
        self.df["Baseline_FVC"] = self.df["Patient"].map(patient_baselines)
        if "Baseline_Percent" not in self.df.columns:
            self.df["Baseline_Percent"] = self.df["Patient"].map(patient_percent)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- 1. Image Data ---
        # Test set images are in 'test' folder, others in 'train'
        img_source_type = "test" if self.dataset_type == "test" else "train"
        mip = self.processor.generate_mip(
            patient_id, img_source_type, self.load_cached_data
        )

        # Convert to Tensor (C, H, W) where C=1
        img_tensor = torch.tensor(mip, dtype=torch.float32).unsqueeze(0)

        # --- 2. Tabular Data ---
        # Features: Age, Sex(OH), Smoking(OH), Baseline_Percent

        # Age: Normalize (approx mean 65, std 15)
        # Use Baseline_Age if available (test set structure), else Age
        age = row["Baseline_Age"] if "Baseline_Age" in row else row["Age"]
        age_norm = (age - 65.0) / 15.0

        # Sex: One-Hot Encoding
        sex = row["Baseline_Sex"] if "Baseline_Sex" in row else row["Sex"]
        sex_idx = self.sex_map.get(sex, 0)
        sex_oh = [0, 0]
        sex_oh[sex_idx] = 1

        # SmokingStatus: One-Hot Encoding
        smoke = (
            row["Baseline_SmokingStatus"]
            if "Baseline_SmokingStatus" in row
            else row["SmokingStatus"]
        )
        smoke_idx = self.smoke_map.get(smoke, 0)
        smoke_oh = [0, 0, 0]
        smoke_oh[smoke_idx] = 1

        # Percent: Normalize (0-1 range approx)
        percent = row["Baseline_Percent"]
        percent_norm = percent / 100.0

        # Combine into a single feature vector
        tab_features = [age_norm] + sex_oh + smoke_oh + [percent_norm]
        tab_tensor = torch.tensor(tab_features, dtype=torch.float32)

        # --- 3. Targets and Meta ---
        if self.dataset_type == "test":
            # For test, we predict for 'Predict_Week'
            weeks = row["Predict_Week"]
            fvc_true = 0.0  # Dummy value
        else:
            weeks = row["Weeks"]
            fvc_true = row["FVC"]

        baseline_fvc = row["Baseline_FVC"]

        return {
            "image": img_tensor,
            "tabular": tab_tensor,
            "base_fvc": torch.tensor(baseline_fvc, dtype=torch.float32),
            "weeks": torch.tensor(weeks, dtype=torch.float32),
            "fvc_true": torch.tensor(fvc_true, dtype=torch.float32),
            "patient_id": patient_id,
        }


def get_dataloaders(debug=False):
    """
    Constructs and returns DataLoaders for Train, Validation, and Test sets.
    """
    # Load Metadata CSVs
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debug mode: subset data for quick pipeline check
    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        test_df = test_df.head(Config.DEBUG_SAMPLES)

    # Initialize Datasets
    train_ds = LungDataset(
        train_df, dataset_type="train", load_cached_data=Config.LOAD_CACHED_DATA
    )
    val_ds = LungDataset(
        val_df, dataset_type="val", load_cached_data=Config.LOAD_CACHED_DATA
    )
    test_ds = LungDataset(
        test_df, dataset_type="test", load_cached_data=Config.LOAD_CACHED_DATA
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
