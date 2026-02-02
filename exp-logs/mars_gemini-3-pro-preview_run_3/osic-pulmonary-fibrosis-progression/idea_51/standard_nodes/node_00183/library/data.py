import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from concurrent.futures import ProcessPoolExecutor
from library.config import Config
from library.utils import get_global_stats

# -------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------


def load_scans(dcm_dir):
    """Loads DICOM files from a directory and sorts them by slice position."""
    files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not files:
        return []

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dcm_dir, f))
            slices.append(ds)
        except Exception:
            continue

    # Sort by ImagePositionPatient Z if available, else InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: int(x.InstanceNumber))

    return slices


def get_pixels_hu(scans):
    """Converts raw DICOM pixel values to Hounsfield Units (HU)."""
    image = np.stack([s.pixel_array.astype(np.float32) for s in scans])

    for i, s in enumerate(scans):
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.float32)

        image[i] += np.float32(intercept)

    return image


def apply_window(image, center, width):
    """Applies radiological windowing and normalizes to [0, 1]."""
    lower = center - width // 2
    upper = center + width // 2

    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image


def select_slices_and_resize(image, target_size=256, num_slices=3):
    """
    Selects 'num_slices' based on maximum lung area content and resizes them.
    Strategy: Find anchor slice (max area) and neighbors with ~50% of that area.
    Cite solution_lesson_node_00124: Content-Adaptive Sampling.
    """
    depth = image.shape[0]

    # Pad if fewer slices than required
    if depth < num_slices:
        padding = [image[0]] * (num_slices - depth)
        image = np.concatenate([np.array(padding), image], axis=0)
        depth = image.shape[0]

    # Calculate approximate lung area per slice
    # Lung window normalized [0, 1] corresponds to [-1350, 150] HU
    # Air (-1000 HU) is approx 0.23. Tissue (0 HU) is approx 0.9.
    areas = []
    for i in range(depth):
        slc = image[i]
        mask = (slc > 0.1) & (slc < 0.6)
        areas.append(np.sum(mask))

    # Anchor slice is the one with maximum lung area
    anchor_idx = np.argmax(areas)
    max_area = areas[anchor_idx]

    if max_area == 0:
        indices = [0, 0, 0]
    else:
        target_area = max_area * 0.5

        # Search below anchor for slice closest to 50% area
        lower_idx = anchor_idx
        min_diff = float("inf")
        # Scan outwards
        for i in range(anchor_idx - 1, -1, -1):
            diff = abs(areas[i] - target_area)
            if diff < min_diff:
                min_diff = diff
                lower_idx = i
            else:
                pass

        # Search above anchor for slice closest to 50% area
        upper_idx = anchor_idx
        min_diff = float("inf")
        for i in range(anchor_idx + 1, depth):
            diff = abs(areas[i] - target_area)
            if diff < min_diff:
                min_diff = diff
                upper_idx = i
            else:
                pass

        indices = [lower_idx, anchor_idx, upper_idx]

    # Sort and ensure valid
    indices.sort()

    selected = image[indices]  # Shape: (3, H, W)

    # Resize each slice
    resized = []
    for i in range(num_slices):
        res = cv2.resize(
            selected[i], (target_size, target_size), interpolation=cv2.INTER_AREA
        )
        resized.append(res)

    return np.stack(resized, axis=0)


def process_single_patient(args):
    """Worker function to process and cache one patient's images."""
    patient_id, dcm_path, cache_path, img_size, num_slices, wl, ww = args

    # Skip if already cached
    if os.path.exists(cache_path):
        return

    try:
        scans = load_scans(dcm_path)
        if not scans:
            # Create zero volume if loading fails
            processed = np.zeros((num_slices, img_size, img_size), dtype=np.float32)
        else:
            vol_hu = get_pixels_hu(scans)
            vol_windowed = apply_window(vol_hu, wl, ww)
            processed = select_slices_and_resize(
                vol_windowed, target_size=img_size, num_slices=num_slices
            )

        np.save(cache_path, processed.astype(np.float32))

    except Exception:
        # Fallback to zeros on error
        processed = np.zeros((num_slices, img_size, img_size), dtype=np.float32)
        np.save(cache_path, processed)


# -------------------------------------------------------------------------
# Dataset Class
# -------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", scalers=None, global_stats=None):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mode = mode
        self.scalers = scalers
        self.global_mean_fvc, self.global_std_fvc = (
            global_stats if global_stats else (0, 1)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image from Cache
        cache_path = os.path.join(self.cache_dir, f"{patient_id}.npy")
        try:
            image = np.load(cache_path)  # Shape: (3, H, W)
        except:
            image = np.zeros(
                (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
            )

        image_tensor = torch.tensor(image, dtype=torch.float32)

        # 2. Prepare Tabular Data
        base_fvc = row["Base_FVC"]
        weeks = row["Weeks"]
        base_week = row["Base_Week"]
        age = row["Age"]
        sex = row["Sex"]
        smoking = row["SmokingStatus"]

        # Feature Engineering
        rel_time = (weeks - base_week) * Config.TIME_SCALE

        # Scaling/Encoding
        # Note: Scalers expect 2D arrays
        s_age = self.scalers["age"].transform([[age]])[0][0]
        s_base_fvc = self.scalers["fvc"].transform([[base_fvc]])[0][0]
        sex_code = self.scalers["sex"].transform([[sex]])[0][0]
        smoke_code = self.scalers["smoke"].transform([[smoking]])[0][0]

        # Stream A: Restricted Inputs [Baseline FVC (scaled), Relative Time]
        stream_a = torch.tensor([s_base_fvc, rel_time], dtype=torch.float32)

        # Stream B: Context Inputs [Baseline FVC, Rel Time, Age, Sex, Smoking]
        stream_b = torch.tensor(
            [s_base_fvc, rel_time, s_age, sex_code, smoke_code], dtype=torch.float32
        )

        data = {
            "image": image_tensor,
            "stream_a": stream_a,
            "stream_b": stream_b,
            "patient_week": f"{patient_id}_{weeks}",
        }

        # 3. Target (Training/Validation only)
        if self.mode != "test":
            fvc = row["FVC"]
            # Global Normalization
            fvc_norm = (fvc - self.global_mean_fvc) / self.global_std_fvc
            data["target"] = torch.tensor(fvc_norm, dtype=torch.float32)
            data["fvc_raw"] = torch.tensor(fvc, dtype=torch.float32)

        return data


# -------------------------------------------------------------------------
# Data Preparation Pipeline
# -------------------------------------------------------------------------


def prepare_data(load_cached_data=True):
    """
    Orchestrates data loading, metadata augmentation, image caching, and dataset creation.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Augment Metadata with Baseline Info
    def add_baseline_info(df, is_test=False):
        if is_test:
            # For test set, the row itself is the baseline
            df["Base_Week"] = df["Weeks"]
            df["Base_FVC"] = df["FVC"]
            return df

        # For train/val, identify baseline (first visit) and merge back
        baseline_df = (
            df.sort_values(["Patient", "Weeks"])
            .groupby("Patient")
            .first()
            .reset_index()
        )
        baseline_df = baseline_df[["Patient", "Weeks", "FVC"]].rename(
            columns={"Weeks": "Base_Week", "FVC": "Base_FVC"}
        )
        df = df.merge(baseline_df, on="Patient", how="left")
        return df

    train_df = add_baseline_info(train_df)
    val_df = add_baseline_info(val_df)
    test_meta_df = add_baseline_info(test_df, is_test=True)

    # 3. Expand Test Data for Submission
    # The submission requires predictions for specific weeks defined in sample_submission.csv
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)
    split_data = sub_df["Patient_Week"].str.rsplit("_", n=1, expand=True)
    sub_df["Patient"] = split_data[0]
    sub_df["Weeks"] = split_data[1].astype(int)

    # Merge clinical info from test metadata
    test_expanded_df = sub_df.merge(
        test_meta_df[
            [
                "Patient",
                "Age",
                "Sex",
                "SmokingStatus",
                "Base_Week",
                "Base_FVC",
                "image_path",
            ]
        ],
        on="Patient",
        how="left",
    )
    test_expanded_df = test_expanded_df.dropna(subset=["Base_FVC"])

    # 4. Image Caching
    def get_caching_tasks(df):
        tasks = []
        unique_df = df[["Patient", "image_path"]].drop_duplicates()
        for _, row in unique_df.iterrows():
            full_path = os.path.join(Config.INPUT_DIR, row["image_path"])
            cache_path = os.path.join(Config.CACHE_DIR, f"{row['Patient']}.npy")
            tasks.append(
                (
                    row["Patient"],
                    full_path,
                    cache_path,
                    Config.IMG_SIZE,
                    Config.NUM_SLICES,
                    Config.WINDOW_LEVEL,
                    Config.WINDOW_WIDTH,
                )
            )
        return tasks

    all_tasks = get_caching_tasks(pd.concat([train_df, val_df])) + get_caching_tasks(
        test_meta_df
    )

    # Filter tasks that need running
    tasks_to_run = [
        t for t in all_tasks if (not load_cached_data) or (not os.path.exists(t[2]))
    ]

    if tasks_to_run:
        print(f"Caching {len(tasks_to_run)} patient images...")
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(executor.map(process_single_patient, tasks_to_run))
    else:
        print("All images found in cache.")

    # 5. Fit Scalers (Train set only)
    scaler_age = StandardScaler()
    scaler_fvc = StandardScaler()
    encoder_sex = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder_smoke = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    scaler_age.fit(train_df[["Age"]])
    scaler_fvc.fit(train_df[["Base_FVC"]])
    encoder_sex.fit(train_df[["Sex"]])
    encoder_smoke.fit(train_df[["SmokingStatus"]])

    scalers = {
        "age": scaler_age,
        "fvc": scaler_fvc,
        "sex": encoder_sex,
        "smoke": encoder_smoke,
    }

    # 6. Global Stats
    global_mean, global_std = get_global_stats(Config.TRAIN_CSV)

    # 7. Debug
    if Config.DEBUG:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLES]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLES]

    # 8. Create Datasets
    train_ds = LungDataset(
        train_df, Config.CACHE_DIR, "train", scalers, (global_mean, global_std)
    )
    val_ds = LungDataset(
        val_df, Config.CACHE_DIR, "val", scalers, (global_mean, global_std)
    )
    test_ds = LungDataset(
        test_expanded_df, Config.CACHE_DIR, "test", scalers, (global_mean, global_std)
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(load_cached_data=True):
    """Returns DataLoaders for train, val, and test sets."""
    train_ds, val_ds, test_ds = prepare_data(load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
