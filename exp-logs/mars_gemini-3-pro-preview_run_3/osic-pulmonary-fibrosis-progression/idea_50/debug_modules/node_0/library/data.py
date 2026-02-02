import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config

# -------------------------------------------------------------------------
# Image Processing & Caching
# -------------------------------------------------------------------------


def load_dicom_stack(dicom_dir, plane="axial"):
    """
    Loads all DICOM files from a directory, sorts them by instance number,
    and converts them to Hounsfield Units (HU).
    """
    files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not files:
        return None

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except Exception:
            continue

    if not slices:
        return None

    # Sort by InstanceNumber (z-position)
    # Some files might not have InstanceNumber, fallback to filename or SliceLocation
    try:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    except:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except:
            slices.sort(key=lambda x: f.filename)

    # Convert to HU
    images = []
    for s in slices:
        img = s.pixel_array.astype(np.float32)
        slope = getattr(s, "RescaleSlope", 1.0)
        intercept = getattr(s, "RescaleIntercept", 0.0)
        img = img * slope + intercept
        images.append(img)

    return np.array(images)


def select_slices(volume):
    """
    Selects 3 slices: Anchor (Max Lung Area) + 2 Boundaries (50% Area Threshold).
    """
    if volume is None or len(volume) == 0:
        # Return zeros if failed
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    num_slices = len(volume)
    if num_slices < 3:
        # Pad if fewer than 3 slices
        indices = [0] * 3
        for i in range(num_slices):
            indices[i] = i
    else:
        # Estimate lung area using a simple threshold (e.g., < -320 HU)
        # Lung tissue is typically -900 to -400 HU. Air is -1000.
        lung_mask = volume < -320
        slice_areas = np.sum(lung_mask, axis=(1, 2))

        # Anchor: Slice with maximum lung area
        anchor_idx = np.argmax(slice_areas)
        max_area = slice_areas[anchor_idx]

        # Boundaries: Find slices with ~50% of max area
        # Search downwards
        lower_idx = 0
        for i in range(anchor_idx, -1, -1):
            if slice_areas[i] < 0.5 * max_area:
                lower_idx = i
                break

        # Search upwards
        upper_idx = num_slices - 1
        for i in range(anchor_idx, num_slices):
            if slice_areas[i] < 0.5 * max_area:
                upper_idx = i
                break

        indices = sorted([lower_idx, anchor_idx, upper_idx])

    selected = volume[indices]
    return selected


def process_patient_image(patient_id, dicom_dir, cache_dir, load_cached=True):
    """
    Processes DICOM images for a patient: Windowing, Slice Selection, Resizing.
    Handles caching.
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing

    # 2. Process from scratch
    volume = load_dicom_stack(dicom_dir)

    # Select Slices (returns 3 slices in HU)
    slices = select_slices(volume)

    # Apply Radiological Windowing
    # Level: -600, Width: 1500 => Range: [-1350, 150]
    lower = Config.WINDOW_LEVEL - (Config.WINDOW_WIDTH / 2)
    upper = Config.WINDOW_LEVEL + (Config.WINDOW_WIDTH / 2)

    processed_slices = []
    for i in range(len(slices)):
        img = slices[i]
        # Clip and Normalize to [0, 1]
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)

        # Resize
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        processed_slices.append(img)

    # Stack to (3, H, W)
    tensor_np = np.stack(processed_slices, axis=0).astype(np.float32)

    # 3. Save to cache
    np.save(cache_path, tensor_np)

    return tensor_np


def prepare_image_cache(patient_ids, data_dir, cache_dir, load_cached=True):
    """
    Iterates over a list of patients and ensures their images are processed and cached.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # We don't need to return anything, just ensure cache exists
    # This avoids multiprocessing issues if done inside DataLoader
    for pid in patient_ids:
        # Determine path (could be in train or test)
        # We check both or rely on input structure
        path_train = os.path.join(Config.TRAIN_DIR, pid)
        path_test = os.path.join(Config.TEST_DIR, pid)

        if os.path.exists(path_train):
            d_dir = path_train
        elif os.path.exists(path_test):
            d_dir = path_test
        else:
            continue  # Should not happen based on metadata

        process_patient_image(pid, d_dir, cache_dir, load_cached)


# -------------------------------------------------------------------------
# Tabular Processing
# -------------------------------------------------------------------------


class TabularPreprocessor:
    def __init__(self):
        self.scaler_fvc = StandardScaler()
        self.scaler_age = StandardScaler()
        self.scaler_percent = StandardScaler()
        # Smoking: 0=Never, 1=Ex, 2=Current (Ordinal)
        self.smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
        self.sex_map = {"Male": 0, "Female": 1}
        self.is_fitted = False

    def fit(self, df):
        # Fit scalers on training data
        # We need to extract the "Baseline" values for fitting to avoid bias from multiple rows
        # However, standard practice is to fit on the feature distribution provided to the model.
        # Here we fit on the unique baseline values to represent the population statistics.

        # Group by patient to get baseline stats
        base_df = df.sort_values("Weeks").groupby("Patient").first().reset_index()

        self.scaler_fvc.fit(base_df[["FVC"]])
        self.scaler_age.fit(base_df[["Age"]])
        self.scaler_percent.fit(base_df[["Percent"]])
        self.is_fitted = True

    def transform(self, df, is_test=False):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        out_df = df.copy()

        # 1. Identify Baseline for each row
        # For training/val, we have history. We need to find the baseline (Week 0 or min week) for each patient.
        # The provided metadata might not explicitly link rows to a baseline.
        # Strategy: For each patient, find the row with min(Weeks). Use that FVC/Percent as baseline.

        # We can do this efficiently using merge
        if "Baseline_FVC" not in out_df.columns:
            # Get baseline info per patient
            # Note: For test set, the provided row IS the baseline.
            if is_test:
                base_info = out_df[["Patient", "FVC", "Percent", "Weeks"]].copy()
            else:
                base_info = (
                    out_df.sort_values("Weeks")
                    .groupby("Patient")
                    .first()[["FVC", "Percent", "Weeks"]]
                    .reset_index()
                )

            base_info = base_info.rename(
                columns={
                    "FVC": "Baseline_FVC",
                    "Percent": "Baseline_Percent",
                    "Weeks": "Baseline_Week",
                }
            )

            out_df = pd.merge(out_df, base_info, on="Patient", how="left")

        # 2. Calculate Relative Time
        # t_rel = (Current_Week - Baseline_Week) * 0.01
        # For test set submission, 'Weeks' in input df is the target week.
        out_df["Relative_Time"] = (out_df["Weeks"] - out_df["Baseline_Week"]) * 0.01

        # 3. Encode Categoricals
        out_df["Sex_Code"] = out_df["Sex"].map(self.sex_map).fillna(0)
        out_df["Smoking_Code"] = out_df["SmokingStatus"].map(self.smoking_map).fillna(0)

        # 4. Scale Numerical Features using the fitted scalers
        # Note: We scale the BASELINE values, as those are the inputs to the model
        out_df["Baseline_FVC_Scaled"] = self.scaler_fvc.transform(
            out_df[["Baseline_FVC"]]
        )
        out_df["Baseline_Percent_Scaled"] = self.scaler_percent.transform(
            out_df[["Baseline_Percent"]]
        )
        out_df["Age_Scaled"] = self.scaler_age.transform(out_df[["Age"]])

        # 5. Normalize Target (if present)
        if "FVC" in out_df.columns and not is_test:
            # Global Target Normalization
            out_df["FVC_Target_Scaled"] = (
                out_df["FVC"] - Config.TARGET_MEAN
            ) / Config.TARGET_STD

        return out_df


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train"):
        """
        Args:
            df: DataFrame with processed features.
            cache_dir: Directory where images are cached.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df
        self.cache_dir = cache_dir
        self.mode = mode

        # Feature columns in order:
        # 1. Baseline FVC (std)
        # 2. Relative Time (scaled)
        # 3. Age (std)
        # 4. Sex (0/1)
        # 5. Smoking (0/1/2)
        # 6. Percent (std)
        self.feature_cols = [
            "Baseline_FVC_Scaled",
            "Relative_Time",
            "Age_Scaled",
            "Sex_Code",
            "Smoking_Code",
            "Baseline_Percent_Scaled",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # We assume prepare_cache has been run, so we just load.
        # If missing (should not happen), we process on the fly.
        try:
            image = process_patient_image(
                patient_id, "", self.cache_dir, load_cached=True
            )
        except Exception:
            # Fallback zero tensor
            image = np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # 2. Tabular Features
        tabular = row[self.feature_cols].values.astype(np.float32)

        # 3. Target / ID
        if self.mode in ["train", "val"]:
            target = row["FVC_Target_Scaled"]
            return (
                torch.tensor(image),
                torch.tensor(tabular),
                torch.tensor(target, dtype=torch.float32),
            )
        else:
            # Test mode: Return ID for submission
            # Construct Patient_Week ID
            pat_week = f"{patient_id}_{row['Weeks']}"
            return torch.tensor(image), torch.tensor(tabular), pat_week


# -------------------------------------------------------------------------
# Main Interface
# -------------------------------------------------------------------------


def get_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, load_cached_data=True
):
    """
    Prepares datasets and dataloaders for Train, Val, and Test.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_meta_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # 2. Prepare Image Cache
    # Collect all unique patients
    all_patients = pd.concat(
        [train_df["Patient"], val_df["Patient"], test_meta_df["Patient"]]
    ).unique()
    prepare_image_cache(
        all_patients, Config.INPUT_DIR, Config.CACHE_DIR, load_cached=load_cached_data
    )

    # 3. Tabular Preprocessing
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    train_df = preprocessor.transform(train_df, is_test=False)
    val_df = preprocessor.transform(val_df, is_test=False)

    # 4. Prepare Test Data
    # The test set requires predicting every week in sample_submission for the test patients.
    # sample_submission contains 'Patient_Week' like 'ID..._12'.
    # We need to parse this to get Patient and Week, then merge with test_meta_df for baseline info.

    # Parse Sample Submission
    sub_df = sample_sub.copy()
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge with baseline metadata
    # test_meta_df contains the baseline info (Age, Sex, FVC, etc.) for each test patient
    test_df = pd.merge(
        sub_df,
        test_meta_df.drop(columns=["Weeks", "FVC", "Percent"]),
        on="Patient",
        how="left",
    )

    # We also need the Baseline FVC/Percent/Weeks from test_meta_df.
    # In test_meta_df, the provided FVC/Weeks IS the baseline.
    base_info = test_meta_df[["Patient", "FVC", "Percent", "Weeks"]].rename(
        columns={
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Weeks": "Baseline_Week",
        }
    )
    test_df = pd.merge(test_df, base_info, on="Patient", how="left")

    # Transform Test Data
    # Note: We manually constructed the baseline columns, so we skip the internal logic in transform that does groupby
    # But our transform method checks for 'Baseline_FVC'. Since we added it, it will use it.
    test_df = preprocessor.transform(test_df, is_test=True)

    # 5. Create Datasets
    train_dataset = OSICDataset(train_df, Config.CACHE_DIR, mode="train")
    val_dataset = OSICDataset(val_df, Config.CACHE_DIR, mode="val")
    test_dataset = OSICDataset(test_df, Config.CACHE_DIR, mode="test")

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
