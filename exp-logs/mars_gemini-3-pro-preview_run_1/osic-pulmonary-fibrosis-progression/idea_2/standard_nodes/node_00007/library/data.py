import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config

# Attempt to import pydicom for DICOM handling
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("Warning: pydicom module not found. Image data will be replaced with zeros.")


def process_dicom_trislab(patient_id, dicom_dir, cache_dir, load_cached_data=True):
    """
    Reads a patient's DICOM directory, generates a Tri-Slab MIP (RGB) image,
    and caches the result as a .npy file.

    Args:
        patient_id (str): Unique patient identifier.
        dicom_dir (str): Path to the directory containing .dcm files.
        cache_dir (str): Directory to store cached .npy files.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: A (H, W, 3) uint8 image representing the Tri-Slab MIP.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")
    img_size = Config.IMG_SIZE

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from Scratch
    # Initialize default blank image (fallback)
    dummy_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if not HAS_PYDICOM or not os.path.exists(dicom_dir):
        np.save(cache_path, dummy_img)
        return dummy_img

    files = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")]
    if not files:
        np.save(cache_path, dummy_img)
        return dummy_img

    # Read DICOMs and extract metadata for sorting
    slices = []
    for f in files:
        fpath = os.path.join(dicom_dir, f)
        try:
            dcm = pydicom.dcmread(fpath)
            # Extract Z-position or Instance Number for sorting
            z_pos = (
                float(dcm.ImagePositionPatient[2])
                if hasattr(dcm, "ImagePositionPatient")
                else 0
            )
            inst_num = int(dcm.InstanceNumber) if hasattr(dcm, "InstanceNumber") else 0
            slices.append({"dcm": dcm, "z": z_pos, "inst": inst_num})
        except Exception:
            continue

    if not slices:
        np.save(cache_path, dummy_img)
        return dummy_img

    # Sort slices by depth (Z-axis)
    # Primary sort by Z position, secondary by Instance Number
    slices.sort(key=lambda x: (x["z"], x["inst"]))

    # Process slices: HU conversion -> Windowing -> Normalization -> Resize
    processed_slices = []
    for s in slices:
        dcm = s["dcm"]
        try:
            pixel_array = dcm.pixel_array.astype(np.float32)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            slope = getattr(dcm, "RescaleSlope", 1)
            img = pixel_array * slope + intercept

            # Lung Window: [-1000, 400]
            img = np.clip(img, -1000, 400)

            # Normalize to [0, 1]
            img = (img - (-1000)) / (400 - (-1000))

            # Resize
            img = cv2.resize(img, (img_size, img_size))
            processed_slices.append(img)
        except Exception:
            continue

    if not processed_slices:
        np.save(cache_path, dummy_img)
        return dummy_img

    processed_slices = np.array(processed_slices)  # Shape: (Depth, H, W)

    # Split into 3 Slabs (Top, Middle, Bottom)
    depth = len(processed_slices)
    if depth < 3:
        # Pad with zeros if fewer than 3 slices
        padding = np.zeros((3 - depth, img_size, img_size), dtype=np.float32)
        processed_slices = np.concatenate([processed_slices, padding], axis=0)
        depth = 3

    chunk_size = depth // 3
    remainder = depth % 3

    # Calculate split indices
    idx1 = chunk_size + (1 if remainder > 0 else 0)
    idx2 = idx1 + chunk_size + (1 if remainder > 1 else 0)

    slab1 = processed_slices[:idx1]
    slab2 = processed_slices[idx1:idx2]
    slab3 = processed_slices[idx2:]

    # Compute Maximum Intensity Projection (MIP) for each slab
    # Handle empty slabs gracefully (though padding prevents this)
    m1 = np.max(slab1, axis=0) if len(slab1) > 0 else np.zeros((img_size, img_size))
    m2 = np.max(slab2, axis=0) if len(slab2) > 0 else np.zeros((img_size, img_size))
    m3 = np.max(slab3, axis=0) if len(slab3) > 0 else np.zeros((img_size, img_size))

    # Stack into RGB channels
    merged = np.stack([m1, m2, m3], axis=-1)

    # Convert to uint8 [0, 255]
    merged = (merged * 255).astype(np.uint8)

    # Save to cache
    np.save(cache_path, merged)

    return merged


def prepare_training_dataframe(df):
    """
    Enriches the training dataframe by identifying the baseline visit for each patient
    and propagating baseline features (FVC, Percent, Age, etc.) to all rows.
    """
    # Sort by Patient and Weeks to find the earliest visit
    df_sorted = df.sort_values(["Patient", "Weeks"])

    # Extract baseline info (first row per patient)
    baseline_df = df_sorted.groupby("Patient").first().reset_index()

    # Select and rename columns to merge
    cols_to_keep = ["Patient", "FVC", "Percent", "Age", "Sex", "SmokingStatus", "Weeks"]
    baseline_df = baseline_df[cols_to_keep]
    baseline_df.columns = [
        "Patient",
        "Baseline_FVC",
        "Baseline_Percent",
        "Baseline_Age",
        "Baseline_Sex",
        "Baseline_SmokingStatus",
        "Baseline_Week",
    ]

    # Merge baseline info back to the original dataframe
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    # Calculate relative time delta
    merged_df["Time_Delta"] = merged_df["Weeks"] - merged_df["Baseline_Week"]

    return merged_df


class PulmonaryDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing patient metadata.
            mode (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentations to apply.
        """
        self.mode = mode
        self.transform = transform
        self.cache_dir = Config.CACHE_DIR

        # Preprocess DataFrame based on mode
        if mode in ["train", "val"]:
            # For training, we need to derive baseline info from history
            self.df = prepare_training_dataframe(df)
        else:
            # For test, df is expected to have 'Predict_Week', 'Baseline_FVC', etc.
            self.df = df.copy()
            # Ensure Time_Delta exists
            if "Time_Delta" not in self.df.columns:
                self.df["Time_Delta"] = (
                    self.df["Predict_Week"] - self.df["Baseline_Week"]
                )

        # Extract columns to arrays for faster access
        self.patient_ids = self.df["Patient"].values
        self.dicom_dirs = self.df["dicom_dir"].values

        self.ages = self.df["Baseline_Age"].values
        self.percents = self.df["Baseline_Percent"].values
        self.sexes = self.df["Baseline_Sex"].values
        self.smokings = self.df["Baseline_SmokingStatus"].values
        self.base_fvcs = self.df["Baseline_FVC"].values
        self.time_deltas = self.df["Time_Delta"].values

        if mode != "test":
            self.targets = self.df["FVC"].values
        else:
            self.targets = np.zeros(len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        # Construct full path to DICOM directory
        # dicom_dir in metadata is relative (e.g., "train/ID...")
        d_dir = os.path.join(Config.INPUT_DIR, self.dicom_dirs[idx])

        # 1. Load Image (Tri-Slab MIP)
        # We always try to use cached data to speed up epochs
        img = process_dicom_trislab(pid, d_dir, self.cache_dir, load_cached_data=True)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            # Fallback: Convert to tensor and normalize to [0, 1]
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        # 2. Process Tabular Features
        # Normalize Age
        age = (self.ages[idx] - Config.AGE_MEAN) / Config.AGE_STD

        # Normalize Percent
        percent = (self.percents[idx] - Config.PERCENT_MEAN) / Config.PERCENT_STD

        # Encode Sex (Male=0, Female=1)
        sex = 0.0 if self.sexes[idx] == "Male" else 1.0

        # Encode Smoking Status
        # Mapping: Ex-smoker=0, Never smoked=1, Currently smokes=2
        s_status = self.smokings[idx]
        if s_status == "Ex-smoker":
            smoke = 0.0
        elif s_status == "Never smoked":
            smoke = 1.0
        else:
            smoke = 2.0

        tabular = torch.tensor([age, percent, sex, smoke], dtype=torch.float32)

        # 3. Scalar Values
        base_fvc = torch.tensor(self.base_fvcs[idx], dtype=torch.float32)
        time_delta = torch.tensor(self.time_deltas[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        return img, tabular, base_fvc, time_delta, target
