import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from library.config import Config
from library.utils import seed_everything


class OSICDataset(Dataset):
    """
    Dataset class for the OSIC Pulmonary Fibrosis Progression competition.
    Handles loading of DICOM images (with caching), tabular features, and target normalization.
    """

    def __init__(self, df, mode="train", transform=None, scalers=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.scalers = scalers or {}

        # Categorical Encodings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # --- 1. Image Loading & Processing ---
        # Determine path to DICOM directory
        if "image_path" in row and pd.notna(row["image_path"]):
            rel_path = row["image_path"]
            full_dir_path = os.path.join(Config.INPUT_DIR, rel_path)
        else:
            # Fallback logic for test set if image_path not explicitly in row
            test_path = os.path.join(Config.INPUT_DIR, "test", patient_id)
            if os.path.exists(test_path):
                full_dir_path = test_path
            else:
                full_dir_path = os.path.join(Config.INPUT_DIR, "train", patient_id)

        # Load processed 3-channel image (Cached or Computed)
        image_np = self.process_patient_image(
            patient_id, full_dir_path, load_cached_data=True
        )

        # Convert to Tensor (C, H, W)
        image = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)

        # --- 2. Tabular Feature Construction ---
        # Features: [Base_FVC_norm, Age_norm, Sex_code, Smoke_code, Relative_Time]

        base_fvc = row["Base_FVC"]
        base_week = row["Base_Week"]
        current_week = row["Weeks"]
        age = row["Age"]

        # Normalize Input Scalars using training set statistics
        fvc_norm = (base_fvc - self.scalers.get("fvc_mean", 0)) / self.scalers.get(
            "fvc_std", 1
        )
        age_norm = (age - self.scalers.get("age_mean", 0)) / self.scalers.get(
            "age_std", 1
        )

        sex_code = self.sex_map.get(row["Sex"], 0)
        smoke_code = self.smoke_map.get(row["SmokingStatus"], 0)

        # Relative Time (Scaled)
        t_rel = (current_week - base_week) * Config.TIME_SCALE

        tabular = np.array(
            [fvc_norm, age_norm, sex_code, smoke_code, t_rel], dtype=np.float32
        )

        tabular = torch.tensor(tabular, dtype=torch.float32)

        # --- 3. Target Preparation ---
        if self.mode != "test":
            raw_target = row["FVC"]
            # Z-score normalization for the target variable
            target = (raw_target - Config.TARGET_MEAN) / Config.TARGET_STD
            target = torch.tensor(target, dtype=torch.float32)
        else:
            # Dummy target for inference
            target = torch.tensor(0.0, dtype=torch.float32)

        return image, tabular, target

    def process_patient_image(self, patient_id, patient_dir, load_cached_data=True):
        """
        Selects 'Anchor' slice and 2 boundaries, windows them, and stacks to 3 channels.
        Implements caching mechanism.
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

        # 1. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception:
                pass  # Corrupt file, recompute

        # 2. Compute from Scratch
        dcm_files = []
        file_paths = glob.glob(os.path.join(patient_dir, "*.dcm"))

        if not file_paths:
            # Return black image if no files found
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Read DICOMs
        for f in file_paths:
            try:
                dcm = pydicom.dcmread(f)
                dcm_files.append(dcm)
            except Exception:
                continue

        if not dcm_files:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        # Sort by Z-position (InstanceNumber or ImagePositionPatient)
        def get_z_pos(d):
            if hasattr(d, "ImagePositionPatient"):
                return float(d.ImagePositionPatient[2])
            if hasattr(d, "InstanceNumber"):
                return float(d.InstanceNumber)
            return 0.0

        dcm_files.sort(key=get_z_pos)

        # Slice Selection Logic
        # Calculate approximate Lung Area using raw HU values
        areas = []
        valid_indices = []

        for i, dcm in enumerate(dcm_files):
            try:
                slope = getattr(dcm, "RescaleSlope", 1)
                intercept = getattr(dcm, "RescaleIntercept", 0)
                arr = dcm.pixel_array.astype(np.float32) * slope + intercept

                # Lung area approximation: pixels between -1000 and -400 HU
                area = np.sum((arr > -1000) & (arr < -400))
                areas.append(area)
                valid_indices.append(i)
            except Exception:
                continue

        if not areas:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

        areas = np.array(areas)
        max_area_idx_local = np.argmax(areas)
        max_area = areas[max_area_idx_local]
        max_area_idx_global = valid_indices[max_area_idx_local]

        # Identify boundary slices (50% of max area)
        threshold = 0.5 * max_area
        candidate_indices_local = np.where(areas > threshold)[0]

        if len(candidate_indices_local) < 3:
            # Fallback: replicate max slice
            selected_indices = [max_area_idx_global] * 3
        else:
            # Select Top, Anchor (Max), and Bottom
            min_cand = np.min(candidate_indices_local)
            max_cand = np.max(candidate_indices_local)

            idx_top = valid_indices[min_cand]
            idx_anchor = max_area_idx_global
            idx_bottom = valid_indices[max_cand]

            # Ensure unique and sorted spatially
            selected_set = {idx_top, idx_anchor, idx_bottom}
            selected_indices = sorted(list(selected_set))

            # Pad if duplicates collapsed the set
            while len(selected_indices) < 3:
                selected_indices.append(selected_indices[-1])
            selected_indices = selected_indices[:3]

        # Process the 3 selected slices
        processed_channels = []
        for idx in selected_indices:
            dcm = dcm_files[idx]
            slope = getattr(dcm, "RescaleSlope", 1)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            img = dcm.pixel_array.astype(np.float32) * slope + intercept

            # Apply Lung Window
            level = Config.LUNG_WINDOW_LEVEL
            width = Config.LUNG_WINDOW_WIDTH
            lower = level - width / 2
            upper = level + width / 2

            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)  # Normalize 0..1

            # Resize
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
            processed_channels.append(img)

        # Stack to (H, W, 3)
        final_image = np.stack(processed_channels, axis=-1)

        # 3. Save to Cache
        if load_cached_data:
            try:
                np.save(cache_path, final_image)
            except Exception:
                pass

        return final_image


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE
):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Computes normalization statistics from the training set.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_meta_df = pd.read_csv(Config.TEST_METADATA)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    if Config.DEBUG:
        train_df = train_df.iloc[:64]
        val_df = val_df.iloc[:32]

    # --- Helper to attach Baseline Info ---
    def add_baseline_info(df, ref_df=None):
        if ref_df is None:
            ref_df = df
        # Find the row with minimum 'Weeks' for each patient to define baseline
        # Sort by Weeks and take first
        baselines = ref_df.sort_values("Weeks").groupby("Patient").first().reset_index()
        baselines = baselines[["Patient", "FVC", "Weeks"]].rename(
            columns={"FVC": "Base_FVC", "Weeks": "Base_Week"}
        )
        # Merge baseline info back to the main dataframe
        df = df.merge(baselines, on="Patient", how="left")
        return df

    # For Train and Val, we derive baseline from their full history
    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # --- Compute Input Scalers (from Train only) ---
    scalers = {
        "fvc_mean": train_df["Base_FVC"].mean(),
        "fvc_std": train_df["Base_FVC"].std(),
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
    }

    # --- Prepare Test Data ---
    # 1. Parse Patient and Week from sample_submission
    test_df = sample_sub.copy()
    test_df["Patient"] = test_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    test_df["Weeks"] = test_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # 2. Prepare Baseline info from test.csv
    # test.csv contains the single baseline visit for test patients
    test_base = test_meta_df.rename(columns={"FVC": "Base_FVC", "Weeks": "Base_Week"})
    cols_needed = [
        "Patient",
        "Base_FVC",
        "Base_Week",
        "Age",
        "Sex",
        "SmokingStatus",
        "image_path",
    ]
    test_base = test_base[cols_needed]

    # 3. Merge baseline info into submission rows
    test_df = test_df.merge(test_base, on="Patient", how="left")

    # --- Instantiate Datasets ---
    train_dataset = OSICDataset(train_df, mode="train", scalers=scalers)
    val_dataset = OSICDataset(val_df, mode="val", scalers=scalers)
    test_dataset = OSICDataset(test_df, mode="test", scalers=scalers)

    # --- Instantiate Loaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
