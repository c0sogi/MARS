import os
import numpy as np
import pandas as pd
import cv2
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


def get_scalers(train_df):
    """
    Computes statistics for standardization from the training dataframe.
    """
    # 1. Target FVC Statistics
    fvc_mean = train_df["FVC"].mean()
    fvc_std = train_df["FVC"].std()

    # 2. Baseline FVC and Age Statistics
    # We need to determine baseline FVC for each patient in training
    # Baseline is defined as the measurement at the earliest week
    temp_df = train_df.copy()
    temp_df = temp_df.sort_values(["Patient", "Weeks"])
    baseline_df = temp_df.groupby("Patient").first().reset_index()

    base_fvc_mean = baseline_df["FVC"].mean()
    base_fvc_std = baseline_df["FVC"].std()

    age_mean = baseline_df["Age"].mean()
    age_std = baseline_df["Age"].std()

    # 3. Weeks Statistics (Relative to baseline)
    # Calculate relative weeks for the whole training set
    # First, merge baseline week back
    base_weeks = baseline_df[["Patient", "Weeks"]].rename(
        columns={"Weeks": "Base_Week"}
    )
    temp_df = temp_df.merge(base_weeks, on="Patient", how="left")
    temp_df["Rel_Weeks"] = temp_df["Weeks"] - temp_df["Base_Week"]

    rel_weeks_mean = temp_df["Rel_Weeks"].mean()
    rel_weeks_std = temp_df["Rel_Weeks"].std()

    scalers = {
        "fvc_mean": fvc_mean,
        "fvc_std": fvc_std,
        "base_fvc_mean": base_fvc_mean,
        "base_fvc_std": base_fvc_std,
        "age_mean": age_mean,
        "age_std": age_std,
        "rel_weeks_mean": rel_weeks_mean,
        "rel_weeks_std": rel_weeks_std,
    }
    return scalers


def load_scan(patient_id, split_type):
    """
    Loads, sorts, selects slices, windows, and resizes CT scan.
    Returns: numpy array of shape (Img_Size, Img_Size, 3) normalized to [0, 1]
    """
    if split_type == "train" or split_type == "val":
        base_dir = Config.TRAIN_DICOM_DIR
    else:
        base_dir = Config.TEST_DICOM_DIR

    path = os.path.join(base_dir, patient_id)
    if not os.path.exists(path):
        # Fallback for missing directory: return black image
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    files = [f for f in os.listdir(path) if f.endswith(".dcm")]
    if not files:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Sort files
    # If pydicom is available, sort by InstanceNumber
    # Else, sort by filename number (assuming 1.dcm, 2.dcm...)
    sorted_files = []
    if HAS_PYDICOM:
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f), stop_before_pixels=True)
                slices.append((int(ds.InstanceNumber), f))
            except:
                pass
        slices.sort(key=lambda x: x[0])
        sorted_files = [x[1] for x in slices]

    if not sorted_files:  # Fallback sorting
        try:
            sorted_files = sorted(files, key=lambda x: int(x.split(".")[0]))
        except:
            sorted_files = sorted(files)

    # Select Slices
    num_slices = len(sorted_files)
    indices = [int(num_slices * pos) for pos in Config.SLICE_POSITIONS]
    # Clamp indices
    indices = [min(max(idx, 0), num_slices - 1) for idx in indices]

    selected_files = [sorted_files[i] for i in indices]

    img_channels = []

    for f in selected_files:
        full_path = os.path.join(path, f)

        # Read and Window
        # HU = pixel * slope + intercept
        # Window: (HU - (level - width/2)) / width

        if HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(full_path)
                img = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", 1)
                intercept = getattr(ds, "RescaleIntercept", -1024)
                img = img * slope + intercept
            except:
                img = np.full(
                    (Config.IMG_SIZE, Config.IMG_SIZE), -1024.0, dtype=np.float32
                )
        else:
            # Fallback using cv2 if pydicom missing
            # Assume image is raw, try to approximate HU
            try:
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    img = np.full(
                        (Config.IMG_SIZE, Config.IMG_SIZE), -1024.0, dtype=np.float32
                    )
                else:
                    img = img.astype(np.float32)
                    # Heuristic: if values are large > 3000, might be unsigned.
                    # Usually CT is ~ -1000 to 2000.
                    # Without headers, this is a guess. Assume offset -1024.
                    img = img - 1024
            except:
                img = np.full(
                    (Config.IMG_SIZE, Config.IMG_SIZE), -1024.0, dtype=np.float32
                )

        # Apply Windowing
        level = Config.WINDOW_LEVEL
        width = Config.WINDOW_WIDTH
        lower = level - width / 2
        upper = level + width / 2

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)  # [0, 1]

        # Resize
        img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        img_channels.append(img_resized)

    # Stack: (H, W, 3)
    img_stack = np.dstack(img_channels)
    return img_stack


def process_patient_image(patient_id, split_type, load_cached_data=True):
    """
    Handles caching and processing of patient images.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except:
            pass  # Failed to load, recompute

    # Compute
    img_stack = load_scan(patient_id, split_type)

    # Save
    try:
        np.save(cache_path, img_stack)
    except:
        pass  # Failed to save, ignore

    return img_stack


class OSICDataset(Dataset):
    def __init__(self, df, split_type="train", scalers=None, load_cached_data=True):
        self.df = df.copy()
        self.split_type = split_type
        self.scalers = scalers
        self.load_cached_data = load_cached_data

        # Pre-process Tabular Data
        # 1. Identify Baseline for each patient in this dataset
        # Sort by Patient and Weeks to find first entry
        sorted_df = self.df.sort_values(["Patient", "Weeks"])
        baseline_info = sorted_df.groupby("Patient").first()[
            ["FVC", "Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
        ]
        baseline_info = baseline_info.rename(
            columns={"FVC": "Base_FVC", "Weeks": "Base_Week"}
        )

        # Merge back
        # We drop original static columns to avoid collision if we merge
        cols_to_drop = ["Age", "Sex", "SmokingStatus", "Percent"]
        self.df = self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns]
        )

        self.df = self.df.merge(baseline_info, on="Patient", how="left")

        # Transforms
        self.transform = A.Compose(
            [A.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD), ToTensorV2()]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image
        img_numpy = process_patient_image(
            patient_id, self.split_type, self.load_cached_data
        )
        # img_numpy is (256, 256, 3) in [0, 1]

        # Apply Albumentations (Normalize + ToTensor)
        aug = self.transform(image=img_numpy)
        img_tensor = aug["image"]  # (3, 256, 256)

        # 2. Tabular Features
        # Features: [Age_sc, Base_FVC_sc, Rel_Week_sc, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Cur]

        # Scalers
        s = self.scalers

        # Continuous
        age = (row["Age"] - s["age_mean"]) / s["age_std"]
        base_fvc = (row["Base_FVC"] - s["base_fvc_mean"]) / s["base_fvc_std"]
        rel_week = (row["Weeks"] - row["Base_Week"] - s["rel_weeks_mean"]) / s[
            "rel_weeks_std"
        ]

        # Categorical (One-Hot)
        # Sex: Male, Female
        sex_m = 1.0 if row["Sex"] == "Male" else 0.0
        sex_f = 1.0 if row["Sex"] == "Female" else 0.0

        # Smoking: Ex-smoker, Never smoked, Currently smokes
        smoke_ex = 1.0 if row["SmokingStatus"] == "Ex-smoker" else 0.0
        smoke_never = 1.0 if row["SmokingStatus"] == "Never smoked" else 0.0
        smoke_cur = 1.0 if row["SmokingStatus"] == "Currently smokes" else 0.0

        tab_vector = np.array(
            [age, base_fvc, rel_week, sex_m, sex_f, smoke_ex, smoke_never, smoke_cur],
            dtype=np.float32,
        )

        tab_tensor = torch.from_numpy(tab_vector)

        # 3. Target
        # If FVC is present (Train/Val), return it scaled
        # If not (Test), return dummy

        if "FVC" in row and pd.notna(row["FVC"]):
            raw_fvc = row["FVC"]
            # Z-score scaling
            target_val = (raw_fvc - s["fvc_mean"]) / s["fvc_std"]
            target_tensor = torch.tensor([target_val], dtype=np.float32)
        else:
            target_tensor = torch.tensor([0.0], dtype=np.float32)

        return {
            "image": img_tensor,
            "tabular": tab_tensor,
            "target": target_tensor,
            "patient_week": f"{patient_id}_{row['Weeks']}",
        }
