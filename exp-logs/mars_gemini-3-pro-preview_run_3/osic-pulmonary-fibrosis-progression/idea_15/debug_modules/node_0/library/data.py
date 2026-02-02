import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from library.config import Config
from library.utils import TargetScaler

# -------------------------------------------------------------------------
# Image Processing & Caching
# -------------------------------------------------------------------------


def load_scan(path):
    """Loads all DICOM files from a directory and sorts them by InstanceNumber."""
    if not os.path.exists(path):
        return []
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return slices


def get_pixels_hu(slices):
    """Converts DICOM slices to Hounsfield Units (HU)."""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def process_patient_scan(patient_dir):
    """
    Reads DICOMs, selects 3 slices (Anchor + 2 Boundaries), resizes, and normalizes.
    Returns a (3, H, W) numpy array.
    """
    slices = load_scan(patient_dir)
    if not slices:
        # Fallback for missing data: return zero tensor
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # Convert to HU
    try:
        image = get_pixels_hu(slices)
    except Exception:
        # Fallback if pixel conversion fails
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # Calculate Lung Area per slice (Thresholding)
    # Lung tissue is typically between -1000 and -400 HU.
    # We use a simple threshold < -300 to identify air/lung.
    lung_areas = []
    for i in range(image.shape[0]):
        # Count pixels that look like lung/air
        mask = (image[i] < -300) & (image[i] > -1200)
        lung_areas.append(np.sum(mask))

    lung_areas = np.array(lung_areas)
    max_area = np.max(lung_areas)

    if max_area == 0:
        # No lung detected
        indices = [0, len(slices) // 2, len(slices) - 1]
    else:
        # Identify slices with significant lung area
        valid_indices = np.where(lung_areas > (max_area * Config.SLICE_AREA_THRESHOLD))[
            0
        ]

        if len(valid_indices) == 0:
            indices = [0, len(slices) // 2, len(slices) - 1]
        else:
            # Anchor: Max area slice
            anchor_idx = np.argmax(lung_areas)

            # Boundaries: First and Last valid slice
            top_idx = valid_indices[0]
            bottom_idx = valid_indices[-1]

            # Ensure we have 3 distinct slices if possible, else duplicate
            indices = [top_idx, anchor_idx, bottom_idx]

    # Extract, Resize, Normalize
    final_slices = []
    for idx in indices:
        img = image[idx]

        # Normalize to [0, 1] using lung window [-1000, 400]
        # v < -1000 -> 0, v > 400 -> 1
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.0

        # Resize
        img = cv2.resize(
            img, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        final_slices.append(img)

    # Stack -> (3, H, W)
    tensor = np.stack(final_slices, axis=0).astype(np.float32)
    return tensor


def prepare_image_cache(df, cache_dir, load_cached_data=True):
    """
    Iterates over unique patients in the dataframe.
    If cached file exists and load_cached_data is True, skip.
    Else, process and save.
    """
    unique_patients = df["Patient"].unique()

    # We don't print progress bars, but we ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)

    for patient in unique_patients:
        save_path = os.path.join(cache_dir, f"{patient}.npy")

        if load_cached_data and os.path.exists(save_path):
            continue

        # Construct path to DICOMs
        # The dataframe has 'image_path' relative to input dir
        rel_path = df[df["Patient"] == patient].iloc[0]["image_path"]
        full_path = os.path.join(Config.INPUT_DIR, rel_path)

        # Process
        img_tensor = process_patient_scan(full_path)

        # Save
        np.save(save_path, img_tensor)


# -------------------------------------------------------------------------
# Tabular Preprocessing
# -------------------------------------------------------------------------


class TabularPreprocessor:
    """
    Handles scaling of numerical features and encoding of categorical features.
    """

    def __init__(self):
        self.scaler_fvc = StandardScaler()
        self.scaler_age = StandardScaler()
        self.le_sex = LabelEncoder()
        self.le_smoke = LabelEncoder()
        self.is_fitted = False

    def fit(self, df):
        # Fit on unique patients to avoid bias from patients with many visits
        # However, for simplicity and robustness, we fit on the baseline data of unique patients

        # Extract baseline rows for fitting
        # We assume the passed df is the training set
        patients = df.groupby("Patient").first().reset_index()

        self.scaler_fvc.fit(
            patients[["FVC"]]
        )  # We use FVC as proxy for Baseline FVC distribution
        self.scaler_age.fit(patients[["Age"]])
        self.le_sex.fit(patients["Sex"])
        self.le_smoke.fit(patients["SmokingStatus"])
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor not fitted.")

        out = df.copy()
        # Note: We are scaling the *Baseline* FVC, not the target FVC.
        # The DF passed here is expected to have 'Baseline_FVC' column created beforehand.
        out["Baseline_FVC"] = self.scaler_fvc.transform(out[["Baseline_FVC"]])
        out["Age"] = self.scaler_age.transform(out[["Age"]])
        out["Sex"] = self.le_sex.transform(out["Sex"])
        out["SmokingStatus"] = self.le_smoke.transform(out["SmokingStatus"])
        return out


def add_baseline_features(df, train_df_ref=None):
    """
    Adds 'Baseline_FVC' and 'Baseline_Weeks' to the dataframe.
    For training data, we find the visit closest to week 0.
    For test data, the provided row is the baseline.
    """
    df = df.copy()

    # If this is the test set (indicated by limited columns or context),
    # the provided FVC/Weeks are the baseline.
    # However, to be robust, we calculate baseline per patient.

    # Helper to find baseline for a group
    def get_baseline(group):
        # For training data, baseline is usually Week 0 or min(abs(Weeks))
        # For test data, there's only one row usually, so it picks that.
        # We prioritize the visit closest to Week 0.
        idx = np.argmin(np.abs(group["Weeks"].values))
        baseline_row = group.iloc[idx]
        return pd.Series(
            {
                "Baseline_Weeks": baseline_row["Weeks"],
                "Baseline_FVC": baseline_row["FVC"],
            }
        )

    # If we are processing a subset (e.g. validation), we should ideally look up
    # baselines from the full history.
    # If train_df_ref is provided, we use it to determine baselines for patients.

    if train_df_ref is not None:
        # Use reference DF to find baselines
        baselines = train_df_ref.groupby("Patient").apply(get_baseline)
    else:
        # Use self
        baselines = df.groupby("Patient").apply(get_baseline)

    # Merge baseline info back
    df = df.merge(baselines, on="Patient", how="left")

    # Calculate Relative Weeks
    df["Relative_Weeks"] = df["Weeks"] - df["Baseline_Weeks"]

    return df


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(
        self, df, cache_dir, tabular_preprocessor, target_scaler=None, is_train=True
    ):
        """
        Args:
            df: DataFrame containing patient visits.
            cache_dir: Directory where processed images are stored.
            tabular_preprocessor: Fitted TabularPreprocessor.
            target_scaler: Fitted TargetScaler (for Z-scoring the target).
            is_train: Boolean.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.tab_preprocessor = tabular_preprocessor
        self.target_scaler = target_scaler
        self.is_train = is_train

        # Pre-process tabular features
        # We assume add_baseline_features has been called
        self.processed_df = self.tab_preprocessor.transform(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.processed_df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        img_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        if os.path.exists(img_path):
            image = np.load(img_path)
        else:
            # Fallback (should not happen if cache is prepared)
            image = np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        # 2. Tabular Features
        # [Baseline_FVC, Age, Sex, SmokingStatus]
        # Note: Baseline_FVC and Age are already scaled by TabularPreprocessor
        tab_vec = np.array(
            [row["Baseline_FVC"], row["Age"], row["Sex"], row["SmokingStatus"]],
            dtype=np.float32,
        )

        # 3. Relative Time
        # Scaled by Config.TIME_SCALE
        t_rel = np.array([row["Relative_Weeks"] * Config.TIME_SCALE], dtype=np.float32)

        # 4. Target
        target = np.array([0.0], dtype=np.float32)
        if self.is_train and self.target_scaler:
            raw_fvc = row["FVC"]
            scaled_fvc = self.target_scaler.transform(raw_fvc)
            target = np.array([scaled_fvc], dtype=np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tab_vec, dtype=torch.float32),
            "time": torch.tensor(t_rel, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_week": f"{patient_id}_{row['Weeks']}",
            "raw_fvc": float(row["FVC"]) if "FVC" in row else 0.0,
        }


# -------------------------------------------------------------------------
# Main Data Loading Function
# -------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True):
    """
    Prepares datasets and dataloaders for Train, Val, and Test.
    Handles caching and preprocessing.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Image Cache
    # We combine all unique patients to process efficiently
    all_patients_df = pd.concat([train_df, val_df, test_df], axis=0).drop_duplicates(
        subset=["Patient"]
    )
    prepare_image_cache(
        all_patients_df, Config.CACHE_DIR, load_cached_data=load_cached_data
    )

    # 3. Feature Engineering (Baseline Extraction)
    # We use train_df as reference for baseline calculation for train/val
    train_df = add_baseline_features(train_df)
    val_df = add_baseline_features(val_df, train_df_ref=train_df)

    # For test, the provided row is the baseline
    test_df = add_baseline_features(test_df)

    # 4. Fit Preprocessors (on Train only)
    tab_preprocessor = TabularPreprocessor()
    tab_preprocessor.fit(train_df)

    target_scaler = TargetScaler()
    target_scaler.fit(train_df["FVC"].values)

    # 5. Create Datasets
    train_dataset = LungDataset(
        train_df, Config.CACHE_DIR, tab_preprocessor, target_scaler, is_train=True
    )

    val_dataset = LungDataset(
        val_df,
        Config.CACHE_DIR,
        tab_preprocessor,
        target_scaler,
        is_train=True,  # Validation still has targets
    )

    # For Test, we need to generate rows for all requested weeks?
    # The prompt says: "For each Patient_Week, you must predict..."
    # The sample_submission.csv contains the Patient_Weeks we need to predict.
    # We should construct the test dataset based on sample_submission.
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Parse sample submission to get Patient and Week
    # Format: ID..._Week
    # We split on the last underscore
    sample_sub["Patient"] = sample_sub["Patient_Week"].apply(
        lambda x: x.rsplit("_", 1)[0]
    )
    sample_sub["Weeks"] = sample_sub["Patient_Week"].apply(
        lambda x: int(x.rsplit("_", 1)[1])
    )

    # Merge with test metadata to get image paths and static features
    test_expanded = sample_sub.merge(
        test_df.drop(columns=["Weeks", "FVC", "Relative_Weeks"]),
        on="Patient",
        how="left",
    )

    # Recalculate Relative Weeks for these new rows
    # Note: test_df had 'Baseline_Weeks' computed in step 3
    # We need to preserve that.
    test_baselines = test_df[
        ["Patient", "Baseline_Weeks", "Baseline_FVC"]
    ].drop_duplicates()
    test_expanded = test_expanded.merge(test_baselines, on="Patient", how="left")
    test_expanded["Relative_Weeks"] = (
        test_expanded["Weeks"] - test_expanded["Baseline_Weeks"]
    )

    # We fill missing FVC with 0 (not used in inference)
    test_expanded["FVC"] = 0

    test_dataset = LungDataset(
        test_expanded, Config.CACHE_DIR, tab_preprocessor, target_scaler, is_train=False
    )

    # 6. Create DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, target_scaler
