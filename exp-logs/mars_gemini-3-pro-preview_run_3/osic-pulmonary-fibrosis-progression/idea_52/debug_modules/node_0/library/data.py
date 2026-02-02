import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# --- Helper Functions ---


def get_img(path):
    """
    Loads a DICOM file, applies lung windowing, and normalizes to [0, 1].
    Window: Level -600, Width 1500.
    """
    try:
        d = pydicom.dcmread(path)
        img = d.pixel_array.astype(np.float32)

        # Rescale Intercept and Slope
        if hasattr(d, "RescaleIntercept") and hasattr(d, "RescaleSlope"):
            slope = float(d.RescaleSlope)
            intercept = float(d.RescaleIntercept)
            img = img * slope + intercept

        # Apply Lung Window
        level = Config.WINDOW_LEVEL
        width = Config.WINDOW_WIDTH
        lower = level - width / 2
        upper = level + width / 2

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)

        return img
    except Exception as e:
        # Return a blank image in case of error
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def process_patient_images(patient_id, image_dir):
    """
    Loads all DICOMs for a patient, selects 3 slices (Anchor + 2 Boundary),
    resizes them, and stacks them into a (3, H, W) tensor.
    """
    if not os.path.exists(image_dir):
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    files = [f for f in os.listdir(image_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort by InstanceNumber to ensure Z-ordering
    dicoms = []
    for f in files:
        try:
            d = pydicom.dcmread(os.path.join(image_dir, f), stop_before_pixels=True)
            # Use InstanceNumber if available, else filename integer
            inst_num = float(d.InstanceNumber) if hasattr(d, "InstanceNumber") else 0
            dicoms.append((f, inst_num))
        except:
            continue

    # Fallback sort by filename if InstanceNumber failed or wasn't unique enough
    if not dicoms:
        dicoms = [(f, 0) for f in files]

    dicoms.sort(key=lambda x: x[1])
    sorted_files = [x[0] for x in dicoms]

    if not sorted_files:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Load all images to compute lung area
    imgs = []
    areas = []

    for f in sorted_files:
        path = os.path.join(image_dir, f)
        img = get_img(path)

        # Calculate Lung Area (heuristic: pixels < 0.6 in normalized window)
        # 0.0 -> -1350 HU, 1.0 -> 150 HU.
        # Lung tissue ~ -900 to -600 HU, which is ~0.3 to ~0.5.
        area = np.sum(img < 0.6)

        # Resize now to save memory
        img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        imgs.append(img_resized)
        areas.append(area)

    areas = np.array(areas)

    # Select Slices
    # 1. Anchor: Max Area
    idx_max = np.argmax(areas)
    max_area = areas[idx_max]

    # 2. Boundaries: Range where area > 50% of max
    if max_area == 0:
        indices = [0, len(imgs) // 2, len(imgs) - 1]
    else:
        valid_indices = np.where(areas > 0.5 * max_area)[0]
        if len(valid_indices) > 0:
            idx_lower = valid_indices[0]
            idx_upper = valid_indices[-1]
        else:
            idx_lower = idx_max
            idx_upper = idx_max

        # Ensure we have 3 distinct slices if possible: Lower, Anchor, Upper
        indices = [idx_lower, idx_max, idx_upper]

        # Handle collapse (duplicate indices) by spreading out if possible
        if indices[0] == indices[1]:
            indices[0] = max(0, indices[1] - 1)
        if indices[2] == indices[1]:
            indices[2] = min(len(imgs) - 1, indices[1] + 1)

    # Stack
    selected_imgs = [imgs[i] for i in indices]

    # If we still don't have 3 (e.g. only 1 file), pad
    while len(selected_imgs) < 3:
        selected_imgs.append(selected_imgs[-1])

    tensor = np.stack(selected_imgs[:3], axis=0)  # (3, H, W)
    return tensor.astype(np.float32)


# --- Dataset Class ---


class OSICDataset(Dataset):
    def __init__(
        self, df, mode="train", cache_dir=Config.CACHE_DIR, load_cached_data=True
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing patient metadata.
            mode (str): 'train', 'val', or 'test'.
            cache_dir (str): Directory to store/load cached numpy arrays.
            load_cached_data (bool): Whether to use disk caching.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cache_dir = cache_dir
        self.load_cached_data = load_cached_data

        # Pre-compute unique patients to process images
        self.patient_ids = self.df["Patient"].unique()

        # Ensure cache directory exists and populate it
        if self.load_cached_data:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._cache_images()

    def _cache_images(self):
        """
        Iterates through patients and caches their processed 3-slice tensors.
        """
        for pid in self.patient_ids:
            save_path = os.path.join(self.cache_dir, f"{pid}.npy")

            if os.path.exists(save_path):
                continue

            # Find image path from dataframe
            patient_rows = self.df[self.df["Patient"] == pid]
            if len(patient_rows) == 0:
                continue

            rel_path = patient_rows.iloc[0]["image_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)

            # Process
            img_tensor = process_patient_images(pid, full_path)

            # Save
            np.save(save_path, img_tensor)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Image
        if self.load_cached_data:
            cache_path = os.path.join(self.cache_dir, f"{pid}.npy")
            if os.path.exists(cache_path):
                image = np.load(cache_path)
            else:
                image = np.zeros(
                    (3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
                )
        else:
            rel_path = row["image_path"]
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            image = process_patient_images(pid, full_path)

        # 2. Extract Features

        # Target (Standardized)
        if self.mode == "test":
            fvc_raw = 0.0
            target = 0.0
        else:
            fvc_raw = float(row["FVC"])
            target = (fvc_raw - Config.TARGET_MEAN) / Config.TARGET_STD

        # Restricted Stream Inputs: [Baseline_FVC_scaled, Relative_Time]
        base_fvc = float(row["Baseline_FVC"])
        base_fvc_scaled = (base_fvc - Config.TARGET_MEAN) / Config.TARGET_STD

        # Relative Time (scaled by 0.01)
        current_week = int(row["Weeks"])
        base_week = int(row["Baseline_Week"])
        rel_time = (current_week - base_week) * 0.01

        restricted_inputs = np.array([base_fvc_scaled, rel_time], dtype=np.float32)

        # Context Stream Inputs: [Baseline_FVC_scaled, Relative_Time, Age_scaled, Sex_code, Smoking_code]
        age = float(row["Age"])
        age_scaled = (age - 65.0) / 15.0  # Approx standardization

        sex = 0.0 if row["Sex"] == "Male" else 1.0

        # Smoking: Ex-smoker=0, Never smoked=1, Currently smokes=2
        status = row["SmokingStatus"]
        if status == "Ex-smoker":
            smoke = 0.0
        elif status == "Never smoked":
            smoke = 1.0
        else:
            smoke = 2.0

        context_inputs = np.array(
            [base_fvc_scaled, rel_time, age_scaled, sex, smoke], dtype=np.float32
        )

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(restricted_inputs, dtype=torch.float32),
            torch.tensor(context_inputs, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


# --- Data Preparation & Loading ---


def prepare_train_dataframe(df):
    """
    Augments the training/val dataframe with Baseline information.
    Baseline is defined as the visit with the minimum 'Weeks' value for each patient.
    """
    df = df.sort_values(["Patient", "Weeks"])

    # Get baseline rows (first row per patient after sort)
    baseline_df = df.groupby("Patient").first().reset_index()

    # Select only needed columns for merge
    baseline_df = baseline_df[["Patient", "FVC", "Weeks"]]
    baseline_df = baseline_df.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge back
    merged_df = pd.merge(df, baseline_df, on="Patient", how="left")

    return merged_df


def prepare_test_dataframe(metadata_df, submission_df):
    """
    Prepares the test dataframe by merging metadata (baseline info) with
    submission requirements (target weeks).
    """
    # Rename metadata columns to Baseline
    meta = metadata_df.rename(columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"})

    # Parse submission_df to get Patient and Target Weeks
    sub = submission_df.copy()
    sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge
    merged = pd.merge(sub, meta, on="Patient", how="left")

    return merged


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Generates DataLoaders for Train, Val, and Test sets.
    """
    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_CSV)
    val_meta = pd.read_csv(Config.VAL_CSV)
    test_meta = pd.read_csv(Config.TEST_CSV)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # Prepare DataFrames
    train_df = prepare_train_dataframe(train_meta)
    val_df = prepare_train_dataframe(val_meta)
    test_df = prepare_test_dataframe(test_meta, sample_sub)

    # Instantiate Datasets
    # Note: Caching happens here.
    train_dataset = OSICDataset(train_df, mode="train")
    val_dataset = OSICDataset(val_df, mode="val")
    test_dataset = OSICDataset(test_df, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
