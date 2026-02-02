import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


def get_img(path):
    """
    Reads a DICOM file and converts it to HU values.
    """
    try:
        d = pydicom.dcmread(path)
        return d.pixel_array.astype(np.float32) * d.RescaleSlope + d.RescaleIntercept
    except Exception as e:
        # Fallback for corrupt or missing files (should not happen with clean metadata)
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_lung_window(img, level=Config.WINDOW_LEVEL, width=Config.WINDOW_WIDTH):
    """
    Applies lung windowing to the image.
    """
    lower = level - width / 2
    upper = level + width / 2
    img = np.clip(img, lower, upper)
    # Normalize to [0, 1]
    img = (img - lower) / (upper - lower)
    return img


def process_patient_scans(patient_id, image_dir):
    """
    Loads all DICOMs for a patient, selects 3 slices (Anchor + Boundaries),
    and returns a stacked tensor of shape (3, H, W).
    """
    if not os.path.exists(image_dir):
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # List all dicom files
    files = [f for f in os.listdir(image_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    # Sort files by instance number (if available) or filename
    # We try to sort numerically by filename assuming they are like 1.dcm, 2.dcm
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()

    # Load all slices to find the anchor
    slices = []
    for f in files:
        img = get_img(os.path.join(image_dir, f))
        # Simple resize for area calculation speedup
        img_small = cv2.resize(img, (128, 128))
        # Apply window for area calculation
        img_win = get_lung_window(img_small)
        # Calculate lung area (simple threshold heuristic: pixels < -200 HU in windowed space?)
        # Actually, in windowed space [0,1], air is 0. Tissue is 1.
        # Lung parenchyma is roughly middle gray.
        # A simple heuristic for "lung area" in a lung window is non-zero and non-saturated pixels,
        # or simply thresholding. Let's use a binary threshold on the windowed image.
        # Lung tissue is dark in CT but bright in "Lung Window" usually?
        # Standard Lung Window: -600, 1500. Air (-1000) -> 0. Bone (400) -> 1.
        # Lungs (-600) are around 0.2-0.3.
        # We look for pixels in range [0.05, 0.8] roughly.
        area = np.sum((img_win > 0.05) & (img_win < 0.9))
        slices.append((f, area))

    # 1. Find Anchor (Max Area)
    slices.sort(key=lambda x: x[1], reverse=True)
    if not slices:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32
        )

    anchor_file = slices[0][0]
    max_area = slices[0][1]

    # 2. Find Boundaries (slices with ~50% of max area)
    # We want one 'above' and one 'below' the anchor if possible, but we only have filenames/indices.
    # We will pick from the pool of slices that have sufficient area.
    valid_slices = [s[0] for s in slices if s[1] > 0.3 * max_area]

    # If we don't have enough valid slices, pad with the anchor
    selected_files = [anchor_file]

    # Try to pick distinct slices
    if len(valid_slices) > 1:
        # Pick one from the first third and one from the last third of the valid list (sorted by area)
        # This is a heuristic. Ideally we sort by Z-position.
        # Let's re-sort valid_slices by index (filename) to get spatial distribution
        try:
            valid_slices.sort(key=lambda x: int(os.path.splitext(x)[0]))
        except:
            valid_slices.sort()

        anchor_idx = -1
        for i, f in enumerate(valid_slices):
            if f == anchor_file:
                anchor_idx = i
                break

        # Select one before and one after anchor
        prev_idx = max(0, anchor_idx - 2)  # Skip immediate neighbor for variance
        next_idx = min(len(valid_slices) - 1, anchor_idx + 2)

        if prev_idx != anchor_idx:
            selected_files.append(valid_slices[prev_idx])
        if next_idx != anchor_idx and next_idx != prev_idx:
            selected_files.append(valid_slices[next_idx])

    # Fill remaining slots with anchor if needed
    while len(selected_files) < Config.NUM_SLICES:
        selected_files.append(anchor_file)

    # Ensure exactly 3
    selected_files = selected_files[: Config.NUM_SLICES]

    # Load and Process Selected Slices
    processed_imgs = []
    for f in selected_files:
        img = get_img(os.path.join(image_dir, f))
        img = get_lung_window(img)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        processed_imgs.append(img)

    # Stack: (C, H, W)
    return np.array(processed_imgs, dtype=np.float32)


def cache_patient_data(patient_id, image_dir, cache_dir, load_cached_data):
    """
    Handles caching logic. Returns the processed image tensor.
    """
    save_path = os.path.join(cache_dir, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(save_path):
        try:
            data = np.load(save_path)
            # Validate shape to ensure consistency with current config (Cite debug_lesson_7)
            if data.shape == (Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE):
                return data.astype(
                    np.float32
                )  # Ensure correct type (Cite debug_lesson_2)
        except:
            pass  # Fallback to re-process if load fails

    # Process
    img_data = process_patient_scans(patient_id, image_dir)

    # Save
    np.save(save_path, img_data)

    return img_data


class LungDataset(Dataset):
    def __init__(self, df, stats, mode="train", load_cached_data=True):
        self.df = df.copy()
        self.stats = stats
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Pre-calculate Baseline FVC for each patient
        # We assume the dataframe contains history.
        # For each patient, baseline is the FVC at min(Weeks).
        self.patient_baselines = {}

        # Group by patient to find baseline
        # Note: In test set, we only have one row per patient usually, which is the baseline.
        # In train set, we have multiple.
        for pid, group in self.df.groupby("Patient"):
            # Sort by Weeks
            group = group.sort_values("Weeks")
            # Baseline is the first measurement
            base_fvc = group.iloc[0]["FVC"]
            base_week = group.iloc[0]["Weeks"]
            self.patient_baselines[pid] = (base_fvc, base_week)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Image (Cached)
        # Construct full path to image dir
        # metadata contains 'image_path' which is relative to input dir
        image_dir = os.path.join(Config.INPUT_DIR, row["image_path"])

        # Retrieve image data (handles caching internally)
        imgs = cache_patient_data(
            pid, image_dir, Config.CACHE_DIR, self.load_cached_data
        )

        # 2. Tabular Features
        # Retrieve baseline info
        base_fvc, base_week = self.patient_baselines.get(
            pid, (row["FVC"], row["Weeks"])
        )

        # Calculate Relative Time
        weeks = row["Weeks"]
        t_rel = (weeks - base_week) * Config.TIME_SCALE

        # Normalize Features
        # Baseline FVC (Z-score)
        base_fvc_norm = (base_fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]

        # Age (Z-score)
        age_norm = (row["Age"] - self.stats["age_mean"]) / self.stats["age_std"]

        # Sex (0=Male, 1=Female)
        sex = 0.0 if row["Sex"] == "Male" else 1.0

        # SmokingStatus (0=Never, 1=Ex, 2=Current)
        smk_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
        smoke = smk_map.get(row["SmokingStatus"], 0.0)

        # Construct Feature Vector
        # [Baseline_FVC, t_rel, Age, Sex, Smoking]
        tabular = np.array(
            [base_fvc_norm, t_rel, age_norm, sex, smoke], dtype=np.float32
        )

        # 3. Target
        # If test mode, we might not have a valid target (or it's dummy), but we return it anyway
        target_fvc = row["FVC"]
        # Normalize Target
        target_norm = (target_fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]

        return (
            torch.tensor(imgs),
            torch.tensor(tabular),
            torch.tensor(target_norm, dtype=torch.float32),
        )


def get_dataloaders(batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates DataLoaders for training and validation.
    Also returns the normalization statistics.
    """
    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Compute Statistics on Training Data
    stats = {
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
    }

    # Initialize Datasets
    train_ds = LungDataset(
        train_df, stats, mode="train", load_cached_data=load_cached_data
    )
    val_ds = LungDataset(val_df, stats, mode="val", load_cached_data=load_cached_data)

    # Initialize Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, stats


def get_test_dataloader(stats, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Creates a DataLoader for the test set (submission).
    Uses sample_submission.csv to define the rows, merged with metadata/test.csv.
    """
    # Load sample submission and test metadata
    sub_df = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))
    test_meta = pd.read_csv(Config.TEST_CSV)

    # Parse Patient and Weeks from Patient_Week column in submission
    # Format: ID..._Week
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # Merge with metadata to get Age, Sex, Smoking, Image Path, Baseline FVC
    # Note: test_meta contains the baseline row for each patient
    # We drop 'Weeks' and 'FVC' from meta to avoid conflict, as we want the target Weeks from sub_df
    # But we need the Baseline FVC from meta.

    # Rename meta columns to indicate they are baseline info
    test_meta_renamed = test_meta.rename(
        columns={"FVC": "Baseline_FVC", "Weeks": "Baseline_Week"}
    )

    # Merge
    merged_df = pd.merge(sub_df, test_meta_renamed, on="Patient", how="left")

    # The LungDataset expects 'FVC' in the column to serve as target (dummy here)
    # and 'Weeks' for the current prediction time.
    # It also expects 'FVC' to calculate baseline if not provided, but we can hack this.
    # We will construct a dataframe that looks like train_df but with the submission rows.

    # LungDataset logic for baseline:
    # It groups by Patient and takes min(Weeks).
    # In merged_df, we have many weeks. The baseline is actually stored in 'Baseline_FVC' column now.
    # To make it compatible with LungDataset's internal baseline calculation,
    # we can just ensure the dataset logic finds the correct baseline.
    # However, LungDataset logic is: "For each patient, baseline is the FVC at min(Weeks)".
    # In the submission file, min(Weeks) might be negative (pre-baseline).
    # The true baseline is what's in test.csv.

    # Strategy: Create a custom Dataset or modify LungDataset to accept pre-computed baselines?
    # Or simply append the baseline row (from test.csv) to the dataframe for each patient,
    # let LungDataset find it, and then filter out the dummy rows?
    # Cleaner: Pass the baseline info directly.
    # But LungDataset calculates it internally.

    # Let's use the standard LungDataset but populate the 'FVC' column with the Baseline_FVC
    # for the row where Weeks == Baseline_Week, and dummy otherwise?
    # No, that's messy.

    # We will instantiate LungDataset with the merged dataframe.
    # We must ensure that for every patient, there is a row that represents the baseline
    # so that LungDataset.__init__ finds it.
    # The merged_df has 'Baseline_FVC' and 'Baseline_Week' columns from our merge.
    # We can just rename 'Baseline_FVC' -> 'FVC' temporarily for the rows that match baseline week?
    # Actually, the simplest way is to trust the 'test.csv' metadata.
    # LungDataset uses `self.df` to find baselines.
    # We can pass `test_meta` (original test.csv) to a helper in LungDataset?
    # No, LungDataset takes `df`.

    # Let's override the `patient_baselines` in the dataset after initialization.

    dataset_df = merged_df.copy()
    # Fill missing FVC with 0 (target is unknown)
    dataset_df["FVC"] = 0

    ds = LungDataset(dataset_df, stats, mode="test", load_cached_data=load_cached_data)

    # Manually inject the correct baselines derived from test_meta
    # test_meta has the true baseline FVC and Week for each patient
    for _, row in test_meta.iterrows():
        pid = row["Patient"]
        ds.patient_baselines[pid] = (row["FVC"], row["Weeks"])

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, sub_df
