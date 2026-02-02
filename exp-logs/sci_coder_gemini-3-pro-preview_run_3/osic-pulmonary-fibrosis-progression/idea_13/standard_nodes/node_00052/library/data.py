import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import TargetScaler

# --------------------------------------------------------------------------
# Image Processing Utilities
# --------------------------------------------------------------------------


def load_scans(dcm_dir):
    """Loads and sorts DICOM files from a directory."""
    files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not files:
        return []

    scans = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dcm_dir, f))
            scans.append(ds)
        except Exception:
            continue

    # Sort by ImagePositionPatient[2] (Z-axis) if available, else InstanceNumber
    try:
        scans.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        scans.sort(key=lambda x: int(x.InstanceNumber))

    return scans


def get_pixels_hu(scans):
    """Converts raw DICOM pixel data to Hounsfield Units."""
    valid_pixels = []
    valid_scans = []

    for s in scans:
        try:
            arr = s.pixel_array
            valid_pixels.append(arr)
            valid_scans.append(s)
        except Exception:
            continue

    if not valid_pixels:
        return np.zeros((0, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.int16)

    image = np.stack(valid_pixels)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    intercept = valid_scans[0].RescaleIntercept
    slope = valid_scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return image


def select_slices_adaptive(image_vol):
    """
    Selects 3 slices (Apical, Middle, Basal) using a content-adaptive heuristic
    based on lung area.
    """
    if len(image_vol) < 3:
        # Padding if not enough slices
        if len(image_vol) == 0:
            return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE))

        # Repeat middle slice
        mid = image_vol[len(image_vol) // 2]
        return np.stack([mid, mid, mid])

    # 1. Threshold to find lung area (approx -1000 to -400 HU)
    # We use a simple threshold < -400 for air/lung
    bw = image_vol < -400
    # Sum over H, W to get area per slice
    slice_areas = bw.sum(axis=(1, 2))

    # 2. Find Anchor (Max Lung Area)
    idx_max = np.argmax(slice_areas)
    max_area = slice_areas[idx_max]

    # 3. Find Boundaries (50% of max area)
    # If max_area is too small (noise), fallback to geometric spacing
    if max_area < 100:
        indices = np.linspace(0, len(image_vol) - 1, 3).astype(int)
    else:
        # Search Upwards (Apical)
        idx_top = 0
        for i in range(idx_max, -1, -1):
            if slice_areas[i] < 0.5 * max_area:
                idx_top = i
                break

        # Search Downwards (Basal)
        idx_bottom = len(image_vol) - 1
        for i in range(idx_max, len(image_vol)):
            if slice_areas[i] < 0.5 * max_area:
                idx_bottom = i
                break

        # Ensure distinct indices if possible
        if idx_top == idx_max:
            idx_top = max(0, idx_max - 1)
        if idx_bottom == idx_max:
            idx_bottom = min(len(image_vol) - 1, idx_max + 1)

        # If still collapsed (e.g. only 1 valid slice), fallback
        if idx_top == idx_bottom:
            indices = np.linspace(0, len(image_vol) - 1, 3).astype(int)
        else:
            indices = [idx_top, idx_max, idx_bottom]

    selected_slices = image_vol[indices]
    return selected_slices


def resize_and_normalize(slices):
    """Resizes slices to Config.IMG_SIZE and normalizes to [0, 1]."""
    processed = []
    for slc in slices:
        # Resize
        img = cv2.resize(slc.astype(float), (Config.IMG_SIZE, Config.IMG_SIZE))

        # Windowing (Lung Window: W=1500, L=-600)
        # Min = -600 - 1500/2 = -1350
        # Max = -600 + 1500/2 = 150
        min_hu = -1350
        max_hu = 150

        img = np.clip(img, min_hu, max_hu)

        # Normalize to [0, 1]
        img = (img - min_hu) / (max_hu - min_hu)
        processed.append(img)

    # Stack to (3, H, W)
    return np.stack(processed, axis=0).astype(np.float32)


def process_patient_image(patient_id, rel_path, load_cached_data=True):
    """
    Handles the caching logic for image processing.
    Returns a numpy array of shape (3, H, W).
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    full_path = os.path.join(Config.INPUT_DIR, rel_path)
    if not os.path.exists(full_path):
        # Return zeros if directory missing (should not happen per metadata check)
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    scans = load_scans(full_path)
    if not scans:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    vol = get_pixels_hu(scans)
    selected = select_slices_adaptive(vol)
    processed = resize_and_normalize(selected)

    # 3. Save to cache
    try:
        np.save(cache_path, processed)
    except Exception:
        pass

    return processed


# --------------------------------------------------------------------------
# Tabular Processing Utilities
# --------------------------------------------------------------------------


class TabularPreprocessor:
    """
    Handles encoding and normalization of tabular features.
    """

    def __init__(self):
        self.age_mean = 0
        self.age_std = 1
        self.bfvc_mean = 0
        self.bfvc_std = 1
        self.fitted = False

    def fit(self, df):
        """Calculates statistics from the training dataframe."""
        self.age_mean = df["Age"].mean()
        self.age_std = df["Age"].std()

        # For Baseline FVC, we need to extract it first.
        # Assuming df has 'Baseline_FVC' column already populated.
        self.bfvc_mean = df["Baseline_FVC"].mean()
        self.bfvc_std = df["Baseline_FVC"].std()
        self.fitted = True

    def transform(self, df):
        """Transforms the dataframe into a normalized feature matrix."""
        if not self.fitted:
            raise RuntimeError("TabularPreprocessor not fitted.")

        data = df.copy()

        # Encode Categoricals (0-indexed integers)
        # Sex: Male=0, Female=1
        data["Sex_Code"] = data["Sex"].map({"Male": 0, "Female": 1}).fillna(0)

        # Smoking: Never smoked=0, Ex-smoker=1, Currently smokes=2
        smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
        data["Smoking_Code"] = data["SmokingStatus"].map(smoke_map).fillna(0)

        # Normalize Numerical
        data["Age_Norm"] = (data["Age"] - self.age_mean) / self.age_std
        data["Baseline_FVC_Norm"] = (
            data["Baseline_FVC"] - self.bfvc_mean
        ) / self.bfvc_std

        # Select features in order: Age, Sex, Smoking, Baseline_FVC
        # Matches Config.TABULAR_FEATURES logic implicitly, but we return vector
        # Vector: [Age_Norm, Sex_Code, Smoking_Code, Baseline_FVC_Norm]

        features = data[
            ["Age_Norm", "Sex_Code", "Smoking_Code", "Baseline_FVC_Norm"]
        ].values.astype(np.float32)
        return features


def add_baseline_info(df):
    """
    Augments the dataframe with 'Baseline_FVC' and 'Baseline_Week'.
    For train/val, baseline is the measurement at the earliest week.
    """
    df = df.copy()
    # Sort by Patient and Weeks to ensure first row is baseline
    df = df.sort_values(["Patient", "Weeks"])

    # Extract baseline rows
    baseline_df = df.groupby("Patient").first().reset_index()
    baseline_df = baseline_df[["Patient", "FVC", "Weeks"]]
    baseline_df.columns = ["Patient", "Baseline_FVC", "Baseline_Week"]

    # Merge back
    df = df.merge(baseline_df, on="Patient", how="left")
    return df


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(
        self,
        df,
        tabular_preprocessor,
        target_scaler=None,
        mode="train",
        load_cached_data=True,
    ):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient info and visits.
            tabular_preprocessor (TabularPreprocessor): Fitted preprocessor.
            target_scaler (TargetScaler, optional): Fitted scaler for FVC.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use cached images.
        """
        self.df = df.reset_index(drop=True)
        self.tab_preprocessor = tabular_preprocessor
        self.target_scaler = target_scaler
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Pre-compute tabular features to save time during iteration
        self.tabular_features = self.tab_preprocessor.transform(self.df)

        # Pre-compute relative time
        # t_rel = (Weeks - Baseline_Week) * Scale
        weeks = self.df["Weeks"].values
        base_weeks = self.df["Baseline_Week"].values
        self.t_rel = (weeks - base_weeks) * Config.WEEKS_SCALE
        self.t_rel = self.t_rel.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        image = process_patient_image(
            patient_id, row["image_path"], self.load_cached_data
        )

        # 2. Tabular Features
        tab_vec = self.tabular_features[idx]

        # 3. Relative Time
        t = self.t_rel[idx]

        data = {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "t_rel": torch.tensor([t], dtype=torch.float32),  # Shape (1,)
            "patient_week": f"{patient_id}_{row['Weeks']}",
        }

        # 4. Target (if available)
        if self.mode != "test":
            fvc = row["FVC"]
            if self.target_scaler:
                fvc_scaled = self.target_scaler.transform(fvc)
                data["target"] = torch.tensor([fvc_scaled], dtype=torch.float32)
            else:
                data["target"] = torch.tensor([fvc], dtype=torch.float32)

        return data


# --------------------------------------------------------------------------
# Data Loaders
# --------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for training and validation.
    Also returns the fitted TargetScaler for inverse transformation.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Debugging option
    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
        val_df = val_df.iloc[: Config.MAX_TRAIN_SAMPLES]

    # 2. Add Baseline Info (FVC at week 0/base)
    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)

    # 3. Fit Scalers on Training Data
    target_scaler = TargetScaler()
    target_scaler.fit(train_df["FVC"].values)

    tab_preprocessor = TabularPreprocessor()
    tab_preprocessor.fit(train_df)

    # 4. Create Datasets
    train_dataset = OSICDataset(
        train_df,
        tab_preprocessor,
        target_scaler,
        mode="train",
        load_cached_data=load_cached_data,
    )

    val_dataset = OSICDataset(
        val_df,
        tab_preprocessor,
        target_scaler,
        mode="val",
        load_cached_data=load_cached_data,
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, target_scaler, tab_preprocessor


def get_submission_loader(tab_preprocessor, load_cached_data=True):
    """
    Prepares a DataLoader for the submission file.
    Expands the sample submission into a dataframe compatible with the model.
    """
    # 1. Load Data
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
    test_meta = pd.read_csv(Config.TEST_CSV)

    # 2. Parse Patient and Weeks from Patient_Week column
    # Format: ID..._WeekNum
    # We need to handle negative weeks carefully if split by '_'
    # A safer way is to split by last occurrence of '_'

    parsed_data = []
    for pw in sample_sub["Patient_Week"]:
        split_idx = pw.rfind("_")
        patient = pw[:split_idx]
        week = int(pw[split_idx + 1 :])
        parsed_data.append({"Patient_Week": pw, "Patient": patient, "Weeks": week})

    sub_df = pd.DataFrame(parsed_data)

    # 3. Merge with Test Metadata to get Static Features (Age, Sex, etc.)
    # Test metadata contains the Baseline info for these patients.
    # Note: In test.csv, 'FVC' is the baseline FVC.

    # Rename FVC to Baseline_FVC and Weeks to Baseline_Week in test_meta
    test_base = test_meta.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge
    sub_df = sub_df.merge(test_base, on="Patient", how="left")

    # Ensure image_path is present (it was in test_meta)
    # If test_meta didn't have image_path, we'd reconstruct it, but metadata/test.csv has it.

    # 4. Create Dataset
    # Note: We do not pass a target scaler because we don't have targets
    dataset = OSICDataset(
        sub_df,
        tab_preprocessor,
        target_scaler=None,
        mode="test",
        load_cached_data=load_cached_data,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader
