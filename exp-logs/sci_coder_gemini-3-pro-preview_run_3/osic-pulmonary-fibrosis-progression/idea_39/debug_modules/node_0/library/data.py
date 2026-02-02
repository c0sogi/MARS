import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from sklearn.preprocessing import StandardScaler
from library.config import Config
from library.utils import seed_everything

# Attempt to import pydicom, but define fallback if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_dicom_raw(path):
    """
    Reads a DICOM file as raw bytes and extracts pixel data.
    Fallback method when pydicom is not available.
    Assumes standard CT format: 512x512 int16 (signed).
    """
    try:
        with open(path, "rb") as f:
            b = f.read()

        # Standard CT image data size: 512 * 512 * 2 bytes = 524,288 bytes
        # We look for the last 524,288 bytes of the file
        expected_bytes = 512 * 512 * 2

        if len(b) >= expected_bytes:
            pixel_data = b[-expected_bytes:]
            # Load as int16
            img = np.frombuffer(pixel_data, dtype=np.int16).copy()
            img = img.reshape((512, 512))

            # Apply typical CT Rescale Slope/Intercept (Slope=1, Intercept=-1024)
            # This is a heuristic since we cannot parse the header without pydicom
            img = img.astype(np.float32)
            img = img * 1.0 - 1024.0
            return img
        else:
            # File too small or unexpected format, return empty slice
            return np.zeros((512, 512), dtype=np.float32)
    except Exception:
        return np.zeros((512, 512), dtype=np.float32)


def load_slice(path):
    """
    Loads a single DICOM slice. Uses pydicom if available, else raw read.
    """
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            # Apply rescale if available
            slope = getattr(dcm, "RescaleSlope", 1)
            intercept = getattr(dcm, "RescaleIntercept", -1024)
            img = img * slope + intercept
            return img
        except:
            return read_dicom_raw(path)
    else:
        return read_dicom_raw(path)


def process_patient_images(patient_id, base_dir):
    """
    Loads, selects, and preprocesses CT slices for a patient.
    Implements Caching to disk to speed up subsequent epochs.

    Args:
        patient_id (str): The patient ID.
        base_dir (str): The directory containing the patient folder (e.g. input/train).

    Returns:
        np.array: Tensor of shape (3, 260, 260)
    """
    # 1. Check Cache
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except:
            pass  # Reload if corrupt

    # 2. Load Dicom Files
    patient_dir = os.path.join(base_dir, patient_id)
    files = glob.glob(os.path.join(patient_dir, "*.dcm"))

    if not files:
        # Return zeros if no files found
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort by instance number (filename)
    try:
        files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        files.sort()  # Fallback lexical sort

    # Load all slices
    imgs = []
    for f in files:
        img = load_slice(f)
        imgs.append(img)

    imgs = np.array(imgs)  # Shape: (D, 512, 512)

    # 3. Select Slices (Anchor + Boundaries)
    # Heuristic: Find slice with maximum lung area.
    # Lung Window range for selection: [-1000, -200] HU
    if len(imgs) > 0:
        lung_pixels = np.logical_and(imgs >= -1000, imgs <= -200)
        lung_areas = np.sum(lung_pixels, axis=(1, 2))
        idx_anchor = np.argmax(lung_areas)
    else:
        idx_anchor = 0

    # Select 3 slices with spacing
    num_slices = len(imgs)
    step = max(1, num_slices // 10)  # 10% spacing to capture volume

    indices = [idx_anchor - step, idx_anchor, idx_anchor + step]
    # Clamp indices to valid range
    indices = [max(0, min(i, num_slices - 1)) for i in indices]

    selected_imgs = imgs[indices]  # (3, 512, 512)

    # 4. Preprocess (Windowing + Resizing)
    processed = []
    lower = Config.WINDOW_LEVEL - Config.WINDOW_WIDTH / 2
    upper = Config.WINDOW_LEVEL + Config.WINDOW_WIDTH / 2

    for i in range(3):
        img = selected_imgs[i]

        # Apply Lung Window
        img = np.clip(img, lower, upper)
        # Normalize to [0, 1]
        img = (img - lower) / (upper - lower)

        # Resize to model input size
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        processed.append(img)

    final_tensor = np.stack(processed, axis=0).astype(np.float32)  # (3, 260, 260)

    # 5. Save to Cache
    np.save(cache_path, final_tensor)

    return final_tensor


class DataProcessor:
    """
    Handles tabular feature engineering, scaler fitting, and baseline extraction.
    """

    def __init__(self):
        self.scalers = {}
        self.patient_baselines = {}

    def fit(self, train_df, all_df):
        """
        Fits scalers using training data and computes baselines using full history.
        """
        # 1. Compute Baselines (Week, FVC) for all patients
        # Baseline is defined as the measurement at the earliest week
        for patient, group in all_df.groupby("Patient"):
            group = group.sort_values("Weeks")
            base_row = group.iloc[0]
            self.patient_baselines[patient] = {
                "BaseWeek": base_row["Weeks"],
                "BaseFVC": base_row["FVC"],
            }

        # 2. Fit Scalers on Training Data
        # Augment train_df with baseline info first
        train_aug = self._augment_df(train_df.copy())

        # Age Scaler
        self.scalers["age"] = StandardScaler()
        self.scalers["age"].fit(train_aug[["Age"]])

        # Baseline FVC Scaler
        self.scalers["base_fvc"] = StandardScaler()
        self.scalers["base_fvc"].fit(train_aug[["BaseFVC"]])

        # Target FVC Scaler (for loss calculation and inverse transform)
        self.scalers["target_fvc"] = StandardScaler()
        self.scalers["target_fvc"].fit(train_aug[["FVC"]])

    def _augment_df(self, df):
        """Adds BaseWeek, BaseFVC, RelWeeks columns to dataframe."""
        base_weeks = []
        base_fvcs = []

        for pid in df["Patient"]:
            if pid in self.patient_baselines:
                base_weeks.append(self.patient_baselines[pid]["BaseWeek"])
                base_fvcs.append(self.patient_baselines[pid]["BaseFVC"])
            else:
                # Fallback (should not happen if all_df is complete)
                base_weeks.append(0)
                base_fvcs.append(2000)

        df["BaseWeek"] = base_weeks
        df["BaseFVC"] = base_fvcs
        df["RelWeeks"] = df["Weeks"] - df["BaseWeek"]
        return df

    def transform(self, df):
        """Applies transformations to a dataframe."""
        df = df.copy()
        df = self._augment_df(df)

        # Numerical Features (Standardized)
        df["Age_s"] = self.scalers["age"].transform(df[["Age"]]).flatten()
        df["BaseFVC_s"] = self.scalers["base_fvc"].transform(df[["BaseFVC"]]).flatten()

        # Target (Z-scored)
        if "FVC" in df.columns:
            df["FVC_s"] = self.scalers["target_fvc"].transform(df[["FVC"]]).flatten()

        # Categorical Features
        # Sex: Male=0, Female=1
        df["Sex_c"] = df["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(int)

        # SmokingStatus: Ordinal (Never=0, Ex=1, Current=2)
        smoking_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
        df["Smoke_c"] = df["SmokingStatus"].map(smoking_map).fillna(0).astype(int)

        # Relative Time (Scaled by 0.01, no Z-score)
        df["RelWeeks_s"] = df["RelWeeks"] * 0.01

        return df


class LungDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # Determine correct input directory based on metadata path
        if "train" in row["image_path"]:
            base_dir = Config.TRAIN_DIR
        else:
            base_dir = Config.TEST_DIR

        img = process_patient_images(patient_id, base_dir)

        # 2. Tabular Features
        # Stream A Input: Age, Sex, Smoke, Time, BaseFVC
        age_s = float(row["Age_s"])
        sex_c = float(row["Sex_c"])
        smoke_c = float(row["Smoke_c"])
        rel_weeks_s = float(row["RelWeeks_s"])
        base_fvc_s = float(row["BaseFVC_s"])

        meta_a = np.array(
            [age_s, sex_c, smoke_c, rel_weeks_s, base_fvc_s], dtype=np.float32
        )

        # Stream B Context: Time, BaseFVC (Standardized)
        meta_b = np.array([rel_weeks_s, base_fvc_s], dtype=np.float32)

        # 3. Target
        if self.mode != "test":
            target = float(row["FVC_s"])
            return img, meta_a, meta_b, np.array([target], dtype=np.float32)
        else:
            return img, meta_a, meta_b, np.zeros((1,), dtype=np.float32)


def get_data(debug=False):
    """
    Factory function to load metadata, fit processors, and return datasets.
    """
    processor = DataProcessor()

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)
    test_meta = pd.read_csv(Config.TEST_CSV)

    # Load raw data to establish global baselines
    raw_train = pd.read_csv(os.path.join(Config.INPUT_DIR, "train.csv"))
    raw_test = pd.read_csv(os.path.join(Config.INPUT_DIR, "test.csv"))
    all_df = pd.concat([raw_train, raw_test], ignore_index=True)

    # Fit processor
    processor.fit(train_meta, all_df)

    # Transform DataFrames
    train_df = processor.transform(train_meta)
    val_df = processor.transform(val_meta)
    test_df = processor.transform(test_meta)

    if debug:
        train_df = train_df.iloc[:32]
        val_df = val_df.iloc[:32]
        test_df = test_df.iloc[:5]

    # Initialize Datasets
    train_ds = LungDataset(train_df, mode="train")
    val_ds = LungDataset(val_df, mode="val")
    test_ds = LungDataset(test_df, mode="test")

    return train_ds, val_ds, test_ds, processor
