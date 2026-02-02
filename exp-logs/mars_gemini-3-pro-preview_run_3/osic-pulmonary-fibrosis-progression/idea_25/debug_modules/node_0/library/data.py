import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config

# Attempt to import pydicom
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom not found. Image features will be zeroed out.")

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------


def get_img_path(patient_id, input_dir):
    """Constructs path to DICOM directory."""
    path_train = os.path.join(input_dir, "train", patient_id)
    path_test = os.path.join(input_dir, "test", patient_id)
    if os.path.exists(path_train):
        return path_train
    elif os.path.exists(path_test):
        return path_test
    return None


def load_scan(path):
    """Loads all DICOM files from a directory, sorted by instance number."""
    if not HAS_PYDICOM:
        return []
    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return []
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in files]
    slices.sort(key=lambda x: float(x.InstanceNumber))
    return slices


def get_pixels_hu(scans):
    """Converts raw DICOM pixel data to Hounsfield Units."""
    if not scans:
        return np.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.int16)

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0 (approx air)
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def process_patient_images(
    patient_id, input_dir=Config.INPUT_DIR, cache_dir=Config.CACHE_DIR
):
    """
    Loads, windows, selects slices, resizes, and caches images for a patient.
    Returns: numpy array of shape (3, 260, 260) normalized to [0, 1].
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # Load from cache if exists
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass

    # If pydicom is missing or path invalid, return zeros
    path = get_img_path(patient_id, input_dir)
    if not HAS_PYDICOM or path is None:
        tensor = np.zeros(
            (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )
        np.save(cache_path, tensor)
        return tensor

    try:
        scans = load_scan(path)
        if not scans:
            raise ValueError("No slices found")
        image_hu = get_pixels_hu(scans)
    except Exception as e:
        # Fallback to zeros on error
        tensor = np.zeros(
            (Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE),
            dtype=np.float32,
        )
        np.save(cache_path, tensor)
        return tensor

    # Windowing
    min_hu = Config.DICOM_WINDOW_CENTER - Config.DICOM_WINDOW_WIDTH / 2
    max_hu = Config.DICOM_WINDOW_CENTER + Config.DICOM_WINDOW_WIDTH / 2

    image_windowed = np.clip(image_hu, min_hu, max_hu)
    image_norm = (image_windowed - min_hu) / (max_hu - min_hu)

    # Slice Selection
    # Heuristic: Count pixels in lung range (-1000 to -400 HU)
    lung_mask = (image_hu > -1000) & (image_hu < -400)
    lung_area = np.sum(lung_mask, axis=(1, 2))

    if np.max(lung_area) == 0:
        # Fallback: middle slices
        mid = len(scans) // 2
        selected_indices = [max(0, mid - 1), mid, min(len(scans) - 1, mid + 1)]
    else:
        anchor_idx = np.argmax(lung_area)
        max_a = lung_area[anchor_idx]
        valid_indices = np.where(lung_area > 0.5 * max_a)[0]

        if len(valid_indices) < 3:
            selected_indices = [anchor_idx] * 3
        else:
            # Pick start, anchor, end of valid range
            selected_indices = [valid_indices[0], anchor_idx, valid_indices[-1]]
            selected_indices.sort()

    # Resize and Stack
    final_slices = []
    for idx in selected_indices:
        slc = image_norm[idx]
        slc_resized = cv2.resize(
            slc, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        final_slices.append(slc_resized)

    tensor = np.stack(final_slices, axis=0).astype(np.float32)

    # Cache
    np.save(cache_path, tensor)

    return tensor


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class OSICDataset(Dataset):
    def __init__(self, df, mode="train", cache_dir=Config.CACHE_DIR):
        self.df = df.copy()
        self.mode = mode
        self.cache_dir = cache_dir

        # 1. Baseline Extraction
        if "Baseline_Week" not in self.df.columns:
            # For train/val, baseline is the earliest visit
            self.df["Baseline_Week"] = self.df.groupby("Patient")["Weeks"].transform(
                "min"
            )

            # Extract baseline FVC and Percent
            base_df = self.df.loc[
                self.df["Weeks"] == self.df["Baseline_Week"],
                ["Patient", "FVC", "Percent"],
            ]
            base_df = base_df.rename(
                columns={"FVC": "Baseline_FVC", "Percent": "Baseline_Percent"}
            )

            # Drop duplicates if any (rare case of multiple entries for same week)
            base_df = base_df.drop_duplicates(subset=["Patient"])

            self.df = self.df.merge(base_df, on="Patient", how="left")

        # 2. Feature Engineering
        self.df["Relative_Weeks"] = self.df["Weeks"] - self.df["Baseline_Week"]

        # Categorical Encoding
        self.df["Sex_Code"] = (
            self.df["Sex"].map({"Male": 0, "Female": 1}).fillna(0).astype(float)
        )
        self.df["Smoking_Code"] = (
            self.df["SmokingStatus"]
            .map({"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2})
            .fillna(0)
            .astype(float)
        )

        # 3. Normalization
        self.df["Age_Scaled"] = (self.df["Age"] - Config.AGE_MEAN) / Config.AGE_STD
        self.df["Baseline_FVC_Scaled"] = (
            self.df["Baseline_FVC"] - Config.TARGET_MEAN
        ) / Config.TARGET_STD
        self.df["Baseline_Percent_Scaled"] = (
            self.df["Baseline_Percent"] - Config.PERCENT_MEAN
        ) / Config.PERCENT_STD
        self.df["Relative_Weeks_Scaled"] = self.df["Relative_Weeks"] * Config.TIME_SCALE

        if self.mode != "test":
            self.df["FVC_Scaled"] = (
                self.df["FVC"] - Config.TARGET_MEAN
            ) / Config.TARGET_STD

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # Image
        image = process_patient_images(patient_id, cache_dir=self.cache_dir)

        # Tabular Vector
        tabular = np.array(
            [
                row["Baseline_FVC_Scaled"],
                row["Baseline_Percent_Scaled"],
                row["Age_Scaled"],
                row["Sex_Code"],
                row["Smoking_Code"],
                row["Relative_Weeks_Scaled"],
            ],
            dtype=np.float32,
        )

        if self.mode != "test":
            target = np.array([row["FVC_Scaled"]], dtype=np.float32)
            return image, tabular, target
        else:
            return image, tabular, patient_id


# -------------------------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------------------------


def get_train_val_datasets():
    train_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))

    return OSICDataset(train_df, mode="train"), OSICDataset(val_df, mode="val")


def get_test_dataset(submission_file=None):
    if submission_file is None:
        submission_file = os.path.join(Config.INPUT_DIR, "sample_submission.csv")

    test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    sub_df = pd.read_csv(submission_file)

    # Parse Patient_Week
    split = sub_df["Patient_Week"].str.rsplit("_", n=1, expand=True)
    sub_df["Patient"] = split[0]
    sub_df["Weeks"] = split[1].astype(int)

    # Prepare metadata for merge
    # test.csv contains the baseline info. We rename it to be explicit.
    test_meta_renamed = test_meta.rename(
        columns={
            "FVC": "Baseline_FVC",
            "Percent": "Baseline_Percent",
            "Weeks": "Baseline_Week",
        }
    )

    # Merge
    cols = [
        "Patient",
        "Baseline_FVC",
        "Baseline_Percent",
        "Baseline_Week",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
    test_expanded = sub_df.merge(test_meta_renamed[cols], on="Patient", how="left")

    return OSICDataset(test_expanded, mode="test"), sub_df
