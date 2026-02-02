import os
import glob
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


class LungDataset(Dataset):
    """
    Dataset class for loading Lung CT scans and clinical data.
    Implements caching, radiological windowing, and content-adaptive slice selection.
    """

    def __init__(self, df, mode="train", cache_dir=Config.CACHE_DIR, age_stats=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cache_dir = cache_dir

        # Use provided age stats or compute from current df (fallback)
        if age_stats:
            self.age_mean, self.age_std = age_stats
        else:
            self.age_mean = self.df["Age"].mean()
            self.age_std = self.df["Age"].std()

        # Pre-process tabular data (Baseline extraction and normalization)
        self.processed_df = self._process_tabular_data()

    def _process_tabular_data(self):
        """
        Groups data by patient to identify baseline FVC and Week.
        Computes relative time and standardizes features.
        """
        # Create a list to store processed records
        processed_records = []

        # Group by patient to find baselines
        # In train/val, baseline is the earliest visit.
        # In test, the provided row is the baseline.
        for patient_id, group in self.df.groupby("Patient"):
            # Sort by weeks to find the earliest visit (Baseline)
            group = group.sort_values("Weeks")
            baseline_row = group.iloc[0]

            base_fvc = baseline_row["FVC"]
            base_week = baseline_row["Weeks"]

            # Apply to all visits for this patient
            for _, row in group.iterrows():
                record = row.to_dict()

                # 1. Relative Time Calculation
                # t_rel = Weeks - Baseline_Week
                rel_weeks = record["Weeks"] - base_week
                record["Rel_Weeks"] = rel_weeks
                record["Scaled_Rel_Weeks"] = rel_weeks * Config.TIME_SCALE

                # 2. Baseline FVC Standardization
                # Z-score using global stats
                record["Scaled_Base_FVC"] = (
                    base_fvc - Config.TARGET_MEAN
                ) / Config.TARGET_STD
                record["Base_FVC"] = base_fvc  # Keep raw for reference

                # 3. Age Standardization
                record["Scaled_Age"] = (record["Age"] - self.age_mean) / (
                    self.age_std + 1e-6
                )

                # 4. Categorical Encoding (One-Hot)
                # Sex
                record["Sex_Male"] = 1.0 if record["Sex"] == "Male" else 0.0
                record["Sex_Female"] = 1.0 if record["Sex"] == "Female" else 0.0

                # SmokingStatus
                record["Smoke_Ex"] = (
                    1.0 if record["SmokingStatus"] == "Ex-smoker" else 0.0
                )
                record["Smoke_Never"] = (
                    1.0 if record["SmokingStatus"] == "Never smoked" else 0.0
                )
                record["Smoke_Current"] = (
                    1.0 if record["SmokingStatus"] == "Currently smokes" else 0.0
                )

                processed_records.append(record)

        return pd.DataFrame(processed_records)

    def __len__(self):
        return len(self.processed_df)

    def __getitem__(self, idx):
        row = self.processed_df.iloc[idx]
        patient_id = row["Patient"]

        # --- Image Loading (with Caching) ---
        img_tensor = self._load_patient_image(patient_id, row["image_path"])

        # --- Tabular Data Construction ---
        # Vector: [Base_FVC, Rel_Weeks, Age, Sex_M, Sex_F, Smoke_Ex, Smoke_N, Smoke_C]
        tabular = np.array(
            [
                row["Scaled_Base_FVC"],
                row["Scaled_Rel_Weeks"],
                row["Scaled_Age"],
                row["Sex_Male"],
                row["Sex_Female"],
                row["Smoke_Ex"],
                row["Smoke_Never"],
                row["Smoke_Current"],
            ],
            dtype=np.float32,
        )

        # --- Target Construction ---
        if self.mode != "test":
            raw_target = row["FVC"]
            # Z-score standardization for training target
            target_val = (raw_target - Config.TARGET_MEAN) / Config.TARGET_STD
            target = np.array([target_val], dtype=np.float32)
        else:
            # Dummy target for test set
            target = np.array([0.0], dtype=np.float32)

        return {
            "image": torch.tensor(img_tensor, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_id": patient_id,
            "weeks": row["Weeks"],
            "rel_weeks": row["Rel_Weeks"],
            "base_fvc": row["Base_FVC"],
        }

    def _load_patient_image(self, patient_id, rel_path, load_cached_data=True):
        """
        Loads patient image from cache if available, otherwise processes from DICOMs.
        """
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")

        # Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Fallback to processing if load fails

        # Process from scratch
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = self._process_dicom_dir(full_path)

        # Save to cache
        try:
            np.save(cache_path, img)
        except Exception:
            pass  # Ignore save errors (e.g. disk full)

        return img

    def _process_dicom_dir(self, dir_path):
        """
        Reads DICOMs, applies windowing, and selects anchor/boundary slices.
        """
        files = glob.glob(os.path.join(dir_path, "*.dcm"))

        # Handle missing files gracefully
        if not files:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Load slices
        slices = []
        for f in files:
            try:
                dcm = pydicom.dcmread(f)
                # Ensure pixel data exists
                if hasattr(dcm, "pixel_array"):
                    slices.append(dcm)
            except Exception:
                continue

        if not slices:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # Sort slices by Z-position (ImagePositionPatient[2]) or InstanceNumber
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except AttributeError:
                pass  # Keep file order if no tags found

        processed_slices = []
        lung_areas = []

        # Process each slice
        for s in slices:
            # 1. Convert to HU
            intercept = getattr(s, "RescaleIntercept", -1024)
            slope = getattr(s, "RescaleSlope", 1)
            img = s.pixel_array.astype(np.float32) * slope + intercept

            # 2. Calculate Lung Area (Approximate via thresholding)
            # Lung tissue is roughly between -1000 and -200 HU
            mask = (img > -1000) & (img < -200)
            area = np.sum(mask)
            lung_areas.append(area)

            # 3. Radiological Windowing (Lung Window)
            lower = Config.WINDOW_LEVEL - Config.WINDOW_WIDTH / 2
            upper = Config.WINDOW_LEVEL + Config.WINDOW_WIDTH / 2
            img = np.clip(img, lower, upper)

            # 4. Normalize to [0, 1]
            img = (img - lower) / (upper - lower)

            # 5. Resize
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
            processed_slices.append(img)

        processed_slices = np.array(processed_slices)
        lung_areas = np.array(lung_areas)

        if len(processed_slices) == 0:
            return np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # --- Content-Adaptive Slice Selection ---
        max_area = np.max(lung_areas)

        # Identify "valid" lung slices (Area > 50% of max)
        valid_indices = np.where(lung_areas > 0.5 * max_area)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(len(processed_slices))

        # Anchor: Slice with maximum lung area
        anchor_idx = np.argmax(lung_areas)

        # Boundaries: Top and Bottom of the valid lung region
        # Since slices are sorted by Z, these are min and max indices of valid set
        top_idx = valid_indices[0]
        bottom_idx = valid_indices[-1]

        # Select indices: [Top, Anchor, Bottom]
        selected_indices = [top_idx, anchor_idx, bottom_idx]

        # Stack slices (Channel, Height, Width)
        final_img = processed_slices[selected_indices]

        return final_img.astype(np.float32)


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # Debug Mode: Subset data
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        # Keep test small but valid
        test_df = test_df.iloc[: min(len(test_df), Config.DEBUG_SAMPLE_SIZE)]

    # Compute Age Statistics from Training Data to prevent leakage
    age_mean = train_df["Age"].mean()
    age_std = train_df["Age"].std()
    age_stats = (age_mean, age_std)

    # Initialize Datasets
    train_dataset = LungDataset(train_df, mode="train", age_stats=age_stats)
    val_dataset = LungDataset(val_df, mode="val", age_stats=age_stats)
    test_dataset = LungDataset(test_df, mode="test", age_stats=age_stats)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Stability for Batch Norm
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
