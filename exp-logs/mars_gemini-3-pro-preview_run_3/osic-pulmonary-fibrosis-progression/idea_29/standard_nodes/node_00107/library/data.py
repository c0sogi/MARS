import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class CTDataset(Dataset):
    """
    Dataset class for loading CT scans and clinical data.
    Implements caching, radiological windowing, and content-adaptive slice selection.
    """

    def __init__(self, df, mode="train", cache=True, stats=None):
        self.df = df.copy()
        self.mode = mode
        self.cache = cache
        self.stats = stats if stats is not None else {}

        # Extract columns for fast access
        self.patient_ids = self.df["Patient"].values
        self.weeks = self.df["Weeks"].values
        self.ages = self.df["Age"].values
        self.sexes = self.df["Sex"].values
        self.smokings = self.df["SmokingStatus"].values
        self.image_paths = self.df["image_path"].values

        # Target variable (FVC)
        if "FVC" in self.df.columns:
            self.fvcs = self.df["FVC"].values
        else:
            self.fvcs = np.zeros(len(self.df), dtype=np.float32)

        # Baseline FVC is required for the input stream
        if "Baseline_FVC" not in self.df.columns:
            # Fallback: if not provided, assume the current FVC is baseline (mostly for test set raw load)
            # However, get_dataloaders should handle this.
            self.baseline_fvcs = self.fvcs
        else:
            self.baseline_fvcs = self.df["Baseline_FVC"].values

    def __len__(self):
        return len(self.df)

    def _process_dicom(self, full_path):
        """
        Reads DICOMs, applies windowing, selects slices, and resizes.
        Returns: np.array of shape (3, H, W) normalized to [0, 1]
        """
        if not os.path.exists(full_path):
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
        if not files:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Read DICOMs
        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(os.path.join(full_path, f))
                slices.append(dcm)
            except Exception:
                continue

        if not slices:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Sort by InstanceNumber or filename
        slices.sort(
            key=lambda x: (
                int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else x.filename
            )
        )

        # Helper: Convert to HU
        def get_hu(dcm):
            slope = getattr(dcm, "RescaleSlope", 1)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            img = dcm.pixel_array.astype(np.float32)
            img = img * slope + intercept
            return img

        # Calculate Lung Areas for Slice Selection
        # Lung window for segmentation check: < -320 HU
        areas = []
        images_hu = []

        for dcm in slices:
            try:
                img_hu = get_hu(dcm)
                images_hu.append(img_hu)
                # Approximate lung area
                binary = (img_hu < -320) & (img_hu > -2000)
                areas.append(np.sum(binary))
            except:
                areas.append(0)
                images_hu.append(np.zeros((512, 512), dtype=np.float32))

        # Content-Adaptive Slice Selection
        num_available = len(images_hu)
        if num_available == 0:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        max_idx = np.argmax(areas)
        max_area = areas[max_idx]

        if max_area == 0:
            # Fallback if no lung detected
            selected_indices = [
                max(0, num_available // 2 - 1),
                num_available // 2,
                min(num_available - 1, num_available // 2 + 1),
            ]
        else:
            # Find boundaries (50% of max area)
            low_idx = max_idx
            for i in range(max_idx, -1, -1):
                if areas[i] < 0.5 * max_area:
                    break
                low_idx = i

            high_idx = max_idx
            for i in range(max_idx, num_available):
                if areas[i] < 0.5 * max_area:
                    break
                high_idx = i

            # Ensure indices are distinct if possible
            if low_idx == max_idx and max_idx > 0:
                low_idx -= 1
            if high_idx == max_idx and max_idx < num_available - 1:
                high_idx += 1

            selected_indices = [low_idx, max_idx, high_idx]

        # Process Selected Slices
        final_stack = []
        for idx in selected_indices:
            idx = max(0, min(idx, num_available - 1))
            img = images_hu[idx]

            # Resize
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

            # Radiological Windowing
            L = Config.WINDOW_LEVEL
            W = Config.WINDOW_WIDTH
            lower, upper = L - W // 2, L + W // 2
            img = np.clip(img, lower, upper)

            # Normalize [0, 1]
            img = (img - lower) / (upper - lower)
            final_stack.append(img)

        return np.stack(final_stack, axis=0).astype(np.float32)

    def _load_image(self, patient_id, rel_path):
        """Handles caching logic."""
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # 1. Try to load from cache
        if self.cache and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, recompute

        # 2. Compute
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img_tensor = self._process_dicom(full_path)

        # 3. Save to cache
        if self.cache:
            try:
                np.save(cache_path, img_tensor)
            except Exception:
                pass

        return img_tensor

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        # 1. Image Data
        img = self._load_image(patient_id, self.image_paths[idx])

        # 2. Tabular Data
        # Retrieve raw values
        week = self.weeks[idx]
        age = self.ages[idx]
        base_fvc = self.baseline_fvcs[idx]
        sex_raw = self.sexes[idx]
        smoke_raw = self.smokings[idx]

        # Feature Engineering & Scaling

        # Relative Time: scaled
        t_rel = week * Config.TIME_SCALE

        # Age: Standardized
        age_mean = self.stats.get("age_mean", 65.0)
        age_std = self.stats.get("age_std", 10.0)
        age_scaled = (age - age_mean) / (age_std + 1e-6)

        # Baseline FVC: Standardized
        base_fvc_mean = self.stats.get("base_fvc_mean", 2500.0)
        base_fvc_std = self.stats.get("base_fvc_std", 500.0)
        base_fvc_scaled = (base_fvc - base_fvc_mean) / (base_fvc_std + 1e-6)

        # Sex: Binary (Male=0, Female=1)
        sex = 1.0 if sex_raw == "Female" else 0.0

        # Smoking: Ordinal (Never=0, Ex=1, Current=2)
        if smoke_raw == "Never smoked":
            smoke = 0.0
        elif smoke_raw == "Ex-smoker":
            smoke = 1.0
        else:
            smoke = 2.0

        # Construct Tabular Input Vector
        # [Baseline FVC, Time, Age, Sex, Smoking]
        tabular = np.array(
            [base_fvc_scaled, t_rel, age_scaled, sex, smoke], dtype=np.float32
        )

        # 3. Target
        # Z-score standardization
        fvc_raw = self.fvcs[idx]
        target = (fvc_raw - Config.TARGET_MEAN) / Config.TARGET_STD

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": f"{patient_id}_{week}",
        }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Prepares DataLoaders for Train, Val, and Test sets.
    Computes normalization stats from the training set.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # --- Baseline FVC Extraction ---
    # For training/val data, we have histories. We define Baseline FVC as the FVC
    # measured at the time closest to Week 0 (the CT scan time).

    def get_baseline_dict(df):
        baseline_map = {}
        for pid in df["Patient"].unique():
            p_data = df[df["Patient"] == pid]
            # Find index where 'Weeks' is closest to 0
            idx = p_data["Weeks"].abs().idxmin()
            baseline_map[pid] = p_data.loc[idx, "FVC"]
        return baseline_map

    # Compute baselines from the respective dataframes
    train_baselines = get_baseline_dict(train_df)
    val_baselines = get_baseline_dict(val_df)

    # Map back to dataframes
    train_df["Baseline_FVC"] = train_df["Patient"].map(train_baselines)
    val_df["Baseline_FVC"] = val_df["Patient"].map(val_baselines)

    # For Test set: The provided row IS the baseline (Week 0 approx)
    test_df["Baseline_FVC"] = test_df["FVC"]

    # --- Compute Normalization Stats ---
    # We compute stats on Train set only to avoid leakage
    stats = {
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
        "base_fvc_mean": train_df["Baseline_FVC"].mean(),
        "base_fvc_std": train_df["Baseline_FVC"].std(),
    }

    # --- Create Datasets ---
    train_ds = CTDataset(train_df, mode="train", cache=True, stats=stats)
    val_ds = CTDataset(val_df, mode="val", cache=True, stats=stats)
    test_ds = CTDataset(test_df, mode="test", cache=True, stats=stats)

    # --- Create Loaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Important for Batch Norm stability
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

    return train_loader, val_loader, test_loader, stats
