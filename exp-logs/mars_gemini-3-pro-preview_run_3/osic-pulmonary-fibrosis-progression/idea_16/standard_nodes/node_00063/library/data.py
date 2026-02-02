import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from library.config import Config

# Attempt to import pydicom for DICOM handling
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False
    print("WARNING: pydicom not found. Image features will be zeroed out.")


class LungDataset(Dataset):
    """
    PyTorch Dataset for Lung Fibrosis Progression.
    Combines 3-channel CT slices with clinical tabular data.
    """

    def __init__(self, df, mode="train", stats=None):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'.
            stats (dict): Dictionary containing normalization statistics (mean/std).
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.stats = stats or {}

        # Pre-compute tabular features to save time during __getitem__
        self.tabular_feats = self._process_tabular()

        # For test mode, store identifiers
        if self.mode == "test":
            self.patient_weeks = self.df["Patient_Week"].values

    def _process_tabular(self):
        """
        Processes tabular features:
        - Relative Time (scaled)
        - Age (Z-scored)
        - Baseline FVC (Z-scored)
        - Sex (Binary)
        - SmokingStatus (One-Hot)
        """
        feats = []

        # 1. Baseline FVC (Standardized)
        base_fvc = self.df["Base_FVC"].values.astype(np.float32)
        if "base_fvc_mean" in self.stats:
            base_fvc = (base_fvc - self.stats["base_fvc_mean"]) / self.stats[
                "base_fvc_std"
            ]

        # 2. Relative Time (Scaled)
        # t_rel = (CurrentWeek - BaselineWeek) * Scale
        weeks = self.df["Weeks"].values.astype(np.float32)
        base_weeks = self.df["Base_Week"].values.astype(np.float32)
        t_rel = (weeks - base_weeks) * Config.TIME_SCALE

        # 3. Age (Standardized)
        age = self.df["Age"].values.astype(np.float32)
        if "age_mean" in self.stats:
            age = (age - self.stats["age_mean"]) / self.stats["age_std"]

        # 4. Sex (Binary: Male=0, Female=1)
        sex = (self.df["Sex"] == "Female").astype(np.float32).values

        # 5. SmokingStatus (One-Hot: Ex, Never, Current)
        # Order: Ex-smoker, Never smoked, Currently smokes
        smoke_ex = (self.df["SmokingStatus"] == "Ex-smoker").astype(np.float32).values
        smoke_never = (
            (self.df["SmokingStatus"] == "Never smoked").astype(np.float32).values
        )
        smoke_current = (
            (self.df["SmokingStatus"] == "Currently smokes").astype(np.float32).values
        )

        # Stack features: [BaseFVC, Time, Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current]
        # Dimensions: (N, 7)
        features = np.stack(
            [base_fvc, t_rel, age, sex, smoke_ex, smoke_never, smoke_current], axis=1
        )

        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Image
        patient_id = self.df.iloc[idx]["Patient"]
        image = self._load_image(patient_id)

        # 2. Get Tabular Features
        tabular = self.tabular_feats[idx]

        # 3. Get Target
        if self.mode != "test":
            raw_fvc = self.df.iloc[idx]["FVC"]
            # Z-score standardization for target
            target = (raw_fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]
            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(tabular, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32),
            )
        else:
            # Test mode: Return ID for submission mapping
            patient_week = self.patient_weeks[idx]
            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(tabular, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),  # Dummy target
                patient_week,
            )

    def _load_image(self, patient_id):
        """Loads processed image from cache."""
        cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")
        if os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except:
                pass
        # Fallback if cache missing or corrupt
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)


def get_pixels_hu(slices):
    """Converts DICOM slices to Hounsfield Units (HU)."""
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to HU
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def process_single_patient(patient_id, dicom_dir):
    """
    Reads DICOMs, selects adaptive slices, resizes, and normalizes.
    Returns: (3, H, W) numpy array normalized to [0, 1].
    """
    if not HAS_PYDICOM:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    patient_dir = os.path.join(dicom_dir, patient_id)
    if not os.path.exists(patient_dir):
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Load slices
    files = glob.glob(os.path.join(patient_dir, "*.dcm"))
    if not files:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    slices = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f)
            # Ensure necessary attributes exist
            if hasattr(dcm, "ImagePositionPatient") and hasattr(dcm, "pixel_array"):
                slices.append(dcm)
        except:
            continue

    if not slices:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Sort by Z position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Convert to HU
    try:
        images_hu = get_pixels_hu(slices)
    except:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    # Content-Adaptive Slice Selection
    # 1. Calculate Lung Area (approximate using threshold)
    # Lung window: -1000 to -320 HU (roughly)
    lung_areas = []
    for i in range(len(images_hu)):
        mask = (images_hu[i] > -1000) & (images_hu[i] < -320)
        lung_areas.append(mask.sum())

    lung_areas = np.array(lung_areas)
    if lung_areas.max() == 0:
        return np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

    max_idx = np.argmax(lung_areas)
    max_area = lung_areas[max_idx]

    # Identify candidate slices (> 50% of max area)
    candidates = np.where(lung_areas > 0.5 * max_area)[0]

    # Select 3 slices: [Lower, Anchor, Upper]
    # Anchor
    idx_anchor = max_idx

    # Lower (below anchor in stack)
    lowers = candidates[candidates < max_idx]
    idx_lower = lowers[len(lowers) // 2] if len(lowers) > 0 else max_idx

    # Upper (above anchor in stack)
    uppers = candidates[candidates > max_idx]
    idx_upper = uppers[len(uppers) // 2] if len(uppers) > 0 else max_idx

    selected_indices = sorted([idx_lower, idx_anchor, idx_upper])

    # Resize and Normalize
    processed_slices = []
    for idx in selected_indices:
        img = images_hu[idx].astype(np.float32)

        # Clip to Lung Window for visualization/CNN
        # Window: [-1000, 400]
        img = np.clip(img, -1000, 400)

        # Normalize to [0, 1]
        img = (img + 1000) / 1400.0

        # Resize
        img_resized = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        processed_slices.append(img_resized)

    # Stack to (3, H, W)
    final_img = np.stack(processed_slices, axis=0)
    return final_img


def process_and_cache_images(patient_ids, dicom_dir, load_cached_data=True):
    """
    Iterates over patient IDs, processes their images, and saves to cache.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    for pid in patient_ids:
        cache_path = os.path.join(Config.CACHE_DIR, f"{pid}.npy")

        if load_cached_data and os.path.exists(cache_path):
            continue

        # Process
        img_data = process_single_patient(pid, dicom_dir)

        # Save
        np.save(cache_path, img_data)


def prepare_metadata(df, is_train=True):
    """
    Enriches dataframe with Baseline FVC and Baseline Week info.
    """
    # Group by Patient to find baseline
    # We assume the visit with min absolute weeks is the baseline visit
    # or simply the first visit if sorted.
    # However, test.csv only has one row per patient (the baseline).

    if "FVC" not in df.columns and is_train:
        # Should not happen for train/val
        raise ValueError("Training data missing FVC column")

    # For train/val, we have full history.
    # We need to identify the baseline FVC for each patient.
    # Strategy: The row with minimal |Weeks| is the baseline.

    # Create a copy to avoid SettingWithCopy
    df = df.copy()

    # Identify baseline rows
    df["Abs_Weeks"] = df["Weeks"].abs()

    # Find baseline info per patient
    # We sort by Patient and Abs_Weeks, then take the first
    baseline_df = df.sort_values(["Patient", "Abs_Weeks"]).drop_duplicates(
        "Patient", keep="first"
    )

    baseline_map = baseline_df.set_index("Patient")[["FVC", "Weeks"]].to_dict("index")

    # Map back to original df
    def get_base_fvc(pid):
        return baseline_map[pid]["FVC"]

    def get_base_week(pid):
        return baseline_map[pid]["Weeks"]

    df["Base_FVC"] = df["Patient"].map(get_base_fvc)
    df["Base_Week"] = df["Patient"].map(get_base_week)

    return df


def prepare_test_metadata():
    """
    Prepares the test dataframe by expanding sample_submission.csv
    and merging with test.csv metadata.
    """
    # 1. Load Metadata (Baseline info)
    test_meta = pd.read_csv(Config.TEST_CSV)

    # 2. Load Submission Template
    sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION)

    # 3. Parse Patient and Week from "Patient_Week"
    # Format: ID00419637202311204720264_18
    sub_df["Patient"] = sub_df["Patient_Week"].apply(lambda x: x.split("_")[0])
    sub_df["Weeks"] = sub_df["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

    # 4. Merge Baseline Info
    # test_meta has columns: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus
    # Rename FVC/Weeks in meta to Base_FVC/Base_Week to avoid collision
    test_meta = test_meta.rename(columns={"FVC": "Base_FVC", "Weeks": "Base_Week"})

    # Merge
    merged_df = sub_df.merge(test_meta, on="Patient", how="left")

    # Ensure columns exist (SmokingStatus, Age, Sex)
    return merged_df


def get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=False,
    load_cached_data=True,
):
    """
    Main function to prepare DataLoaders.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        train_df = train_df.iloc[:50]
        val_df = val_df.iloc[:20]

    # 2. Prepare Metadata (Identify Baselines)
    train_df = prepare_metadata(train_df, is_train=True)
    val_df = prepare_metadata(val_df, is_train=True)

    # 3. Compute Normalization Statistics (from Train only)
    stats = {
        "fvc_mean": train_df["FVC"].mean(),
        "fvc_std": train_df["FVC"].std(),
        "base_fvc_mean": train_df["Base_FVC"].mean(),
        "base_fvc_std": train_df["Base_FVC"].std(),
        "age_mean": train_df["Age"].mean(),
        "age_std": train_df["Age"].std(),
    }

    print("Normalization Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 4. Process Images (Cache)
    # Collect all unique patients from train and val
    all_patients = pd.concat([train_df["Patient"], val_df["Patient"]]).unique()
    print(f"Processing images for {len(all_patients)} training/val patients...")
    process_and_cache_images(
        all_patients, Config.TRAIN_DICOM_DIR, load_cached_data=load_cached_data
    )

    # 5. Prepare Test Data
    test_df = prepare_test_metadata()
    test_patients = test_df["Patient"].unique()
    print(f"Processing images for {len(test_patients)} test patients...")
    process_and_cache_images(
        test_patients, Config.TEST_DICOM_DIR, load_cached_data=load_cached_data
    )

    # 6. Create Datasets
    train_dataset = LungDataset(train_df, mode="train", stats=stats)
    val_dataset = LungDataset(val_df, mode="val", stats=stats)
    test_dataset = LungDataset(test_df, mode="test", stats=stats)

    # 7. Create Loaders
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

    return train_loader, val_loader, test_loader, stats
