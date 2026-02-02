import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from library.config import Config
from library.utils import seed_everything

# Set global seed
seed_everything(Config.SEED)


def get_img_path(patient_id):
    """
    Resolves the directory path for a patient's DICOM files.
    """
    train_path = os.path.join(Config.TRAIN_DIR, patient_id)
    test_path = os.path.join(Config.TEST_DIR, patient_id)

    if os.path.exists(train_path):
        return train_path
    elif os.path.exists(test_path):
        return test_path
    else:
        # Fallback: check if the patient folder is directly in input (rare case)
        return None


def load_scan(path):
    """
    Loads all DICOM files from a directory and sorts them by spatial position.
    """
    slices = [
        pydicom.dcmread(os.path.join(path, s))
        for s in os.listdir(path)
        if s.endswith(".dcm")
    ]

    # Sort by ImagePositionPatient Z coordinate if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(scans):
    """
    Converts raw DICOM pixel_array to Hounsfield Units (HU).
    """
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0 (air)
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def apply_window(image, level, width):
    """
    Applies radiological windowing and normalizes to [0, 1].
    """
    lower = level - width // 2
    upper = level + width // 2

    img = np.clip(image, lower, upper)
    img = (img - lower) / (upper - lower)
    return img


def select_slices_and_resize(image_hu):
    """
    Selects Anchor slice (max lung area) and 2 boundary slices, then resizes.
    Returns: (H, W, 3) float32 array.
    """
    # 1. Calculate Lung Area per slice
    # Heuristic: Lung tissue is approx -1000 to -400 HU.
    lung_mask = (image_hu > -1000) & (image_hu < -400)
    area = lung_mask.sum(axis=(1, 2))

    # 2. Find Anchor (Max Area)
    if len(area) == 0:
        # Fallback for empty/corrupt scans
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    anchor_idx = np.argmax(area)
    max_area = area[anchor_idx]

    # 3. Find Boundary Slices (Approx 50% area)
    n_slices = len(image_hu)
    target_area = 0.5 * max_area

    area_diff = np.abs(area - target_area)
    # Exclude anchor from being picked again
    area_diff[anchor_idx] = np.inf

    above_indices = np.where(np.arange(n_slices) < anchor_idx)[0]
    below_indices = np.where(np.arange(n_slices) > anchor_idx)[0]

    if len(above_indices) > 0:
        idx_above = above_indices[np.argmin(area_diff[above_indices])]
    else:
        idx_above = max(0, anchor_idx - 1)

    if len(below_indices) > 0:
        idx_below = below_indices[np.argmin(area_diff[below_indices])]
    else:
        idx_below = min(n_slices - 1, anchor_idx + 1)

    selected_indices = [idx_above, anchor_idx, idx_below]

    # 4. Extract and Window
    selected_slices = image_hu[selected_indices]
    windowed = apply_window(selected_slices, Config.WINDOW_LEVEL, Config.WINDOW_WIDTH)

    # 5. Resize
    resized_channels = []
    for i in range(3):
        slc = windowed[i]
        slc = cv2.resize(
            slc, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_AREA
        )
        resized_channels.append(slc)

    # Stack to (H, W, 3)
    img_processed = np.stack(resized_channels, axis=-1)

    return img_processed.astype(np.float32)


def process_patient(patient_id, load_cached_data=True):
    """
    Orchestrates loading, processing, and caching of patient images.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to re-processing if corrupt

    # 2. Process from Scratch
    img_dir = get_img_path(patient_id)
    if img_dir is None:
        # Return black image if path not found (should not happen with valid metadata)
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    scans = load_scan(img_dir)
    image_hu = get_pixels_hu(scans)
    img_processed = select_slices_and_resize(image_hu)

    # 3. Save Cache
    np.save(cache_path, img_processed)

    return img_processed


class LungDataset(Dataset):
    def __init__(self, df, mode="train", transform=None, load_cached_data=True):
        self.df = df.copy()
        self.mode = mode
        self.transform = transform
        self.load_cached_data = load_cached_data

        # Mappings
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}

        # Baseline FVC Lookup
        # If 'BaselineFVC' column exists (prepared for submission), use it.
        # Otherwise, derive from history (find row with min |Weeks|).
        self.baseline_lookup = {}

        if "BaselineFVC" in self.df.columns:
            for idx, row in self.df.iterrows():
                self.baseline_lookup[row["Patient"]] = row["BaselineFVC"]
        else:
            unique_patients = self.df["Patient"].unique()
            for pid in unique_patients:
                patient_rows = self.df[self.df["Patient"] == pid]
                # Find row closest to Week 0
                idx_min = patient_rows["Weeks"].abs().idxmin()
                self.baseline_lookup[pid] = patient_rows.loc[idx_min, "FVC"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Image
        img = process_patient(pid, self.load_cached_data)
        # Transpose to (C, H, W) for PyTorch
        img = np.transpose(img, (2, 0, 1))

        # 2. Tabular Features
        base_fvc = self.baseline_lookup.get(pid, 2000)  # Default if missing
        current_weeks = row["Weeks"]

        # Relative Time (scaled)
        t_rel = current_weeks * 0.01

        # Normalize Scalars
        base_fvc_scaled = (base_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
        age_scaled = (row["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        sex_enc = self.sex_map.get(row["Sex"], 0)
        smoke_enc = self.smoke_map.get(row["SmokingStatus"], 0)

        tabular = np.array(
            [base_fvc_scaled, t_rel, age_scaled, sex_enc, smoke_enc], dtype=np.float32
        )

        # 3. Target / Return
        if self.mode in ["train", "val"]:
            target_fvc = row["FVC"]
            target_scaled = (target_fvc - Config.TARGET_MEAN) / Config.TARGET_STD
            return (
                torch.tensor(img),
                torch.tensor(tabular),
                torch.tensor(target_scaled, dtype=torch.float32),
            )

        elif self.mode == "submission":
            pw_id = row["Patient_Week"]
            return torch.tensor(img), torch.tensor(tabular), pw_id

        else:
            return torch.tensor(img), torch.tensor(tabular)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and submission.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # --- Prepare Submission Data ---
    # Merge sample_submission (Patient_Week) with test_df (Static Features)
    sub_df = sample_sub.copy()

    # Extract Patient and Week from ID (e.g., "ID001_-12")
    sub_df[["Patient", "Weeks_str"]] = sub_df["Patient_Week"].str.rsplit(
        "_", n=1, expand=True
    )
    sub_df["Weeks"] = sub_df["Weeks_str"].astype(int)

    # Prepare baseline info from test_df to merge
    # test_df contains the baseline measurement for test patients
    test_static = test_df.drop(columns=["Weeks", "FVC", "Percent"])
    test_baseline = test_df[["Patient", "FVC"]].rename(columns={"FVC": "BaselineFVC"})

    # Merge static features and baseline FVC
    sub_merged = sub_df.merge(test_static, on="Patient", how="left")
    sub_merged = sub_merged.merge(test_baseline, on="Patient", how="left")

    # Ensure FVC column exists (dummy) for Dataset compatibility
    sub_merged["FVC"] = 0

    # --- Create Datasets ---
    train_dataset = LungDataset(
        train_df, mode="train", load_cached_data=load_cached_data
    )
    val_dataset = LungDataset(val_df, mode="val", load_cached_data=load_cached_data)
    submission_dataset = LungDataset(
        sub_merged, mode="submission", load_cached_data=load_cached_data
    )

    # --- Create Loaders ---
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

    submission_loader = DataLoader(
        submission_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, submission_loader
