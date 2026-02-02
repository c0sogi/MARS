import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything


def get_img_seq(patient_id, image_dir):
    """
    Loads DICOM images, selects 3 adaptive slices (Anchor + Boundaries),
    resizes, normalizes, and returns a 3D numpy array (3, H, W).
    """
    # 1. Load all DICOM files
    dcm_files = glob.glob(os.path.join(image_dir, "*.dcm"))
    if not dcm_files:
        # Fallback for missing directories (should not happen in valid data)
        return np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    slices = []
    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(f)
            slices.append(dcm)
        except Exception:
            continue

    # Sort by InstanceNumber (Z-position)
    slices.sort(key=lambda x: int(x.InstanceNumber))

    if not slices:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    # 2. Convert to HU and Calculate Lung Areas
    # Standard Lung Window for area calculation: [-1000, -400] roughly
    # We use a simplified approach to find the "Anchor"

    img_shape = (512, 512)  # Standard CT size, used for area calc
    slice_areas = []
    processed_images = []

    for s in slices:
        # Convert to HU
        intercept = getattr(s, "RescaleIntercept", -1024)
        slope = getattr(s, "RescaleSlope", 1)
        img = s.pixel_array.astype(np.float32) * slope + intercept

        # Resize to common shape for area comparison if needed,
        # but usually pixel_array is 512x512.
        if img.shape != img_shape:
            img = cv2.resize(img, img_shape)

        # Threshold for lung tissue (approx)
        # Lung is typically -900 to -400 HU.
        # We count pixels in a generous range to identify lung presence.
        lung_pixels = np.sum((img > -1000) & (img < -400))
        slice_areas.append(lung_pixels)

        # Store processed HU image for final selection
        processed_images.append(img)

    slice_areas = np.array(slice_areas)

    # 3. Adaptive Slice Selection
    if len(slice_areas) == 0:
        return np.zeros(
            (Config.NUM_SLICES, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32
        )

    max_area_idx = np.argmax(slice_areas)
    max_area = slice_areas[max_area_idx]

    # Define ROI: slices with > 50% of max lung area
    roi_indices = np.where(slice_areas > 0.5 * max_area)[0]

    if len(roi_indices) < 3:
        # Fallback: take center and adjacent
        selected_indices = [
            max(0, max_area_idx - 1),
            max_area_idx,
            min(len(slices) - 1, max_area_idx + 1),
        ]
        # Ensure unique if len(slices) is small
        selected_indices = sorted(list(set(selected_indices)))
        # Pad if still < 3
        while len(selected_indices) < 3:
            selected_indices.append(selected_indices[-1])
    else:
        # Select: Top (first in ROI), Anchor (Max), Bottom (last in ROI)
        # Note: DICOM InstanceNumber usually goes Top->Bottom or Bottom->Top.
        # We just want spatial spread.
        idx_start = roi_indices[0]
        idx_end = roi_indices[-1]
        selected_indices = [idx_start, max_area_idx, idx_end]
        selected_indices.sort()

    # 4. Prepare Final Tensor
    final_slices = []
    for idx in selected_indices:
        img = processed_images[idx]

        # Normalize: Window [-1000, 400] -> [0, 1]
        # This covers lung (-900) to soft tissue (+40) and bone (+400)
        lower, upper = -1000, 400
        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)

        # Resize to Model Input Size
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        final_slices.append(img)

    # Stack -> (3, H, W)
    img_tensor = np.stack(final_slices, axis=0).astype(np.float32)
    return img_tensor


def load_or_process_image(patient_id, rel_image_path, load_cached_data=True):
    """
    Handles caching logic.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{patient_id}.npy")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to re-processing

    # Process
    full_image_dir = os.path.join(Config.INPUT_DIR, rel_image_path)
    img_data = get_img_seq(patient_id, full_image_dir)

    # Save
    np.save(cache_path, img_data)

    return img_data


class LungDataset(Dataset):
    def __init__(self, df, mode="train", cache_images=True):
        """
        Args:
            df: DataFrame containing metadata.
            mode: 'train', 'val', or 'test'.
            cache_images: Whether to use disk caching for images.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.cache_images = cache_images

        # Pre-encode categorical maps
        self.sex_map = {"Male": 0, "Female": 1}
        self.smoke_map = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Image Loading
        # Use image_path from metadata
        img_tensor = load_or_process_image(
            patient_id, row["image_path"], load_cached_data=self.cache_images
        )

        # 2. Tabular Feature Engineering
        # Age Standardization
        age = (row["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        # Categorical Encoding
        sex = self.sex_map.get(row["Sex"], 0)
        smoke = self.smoke_map.get(row["SmokingStatus"], 0)

        # Baseline FVC Standardization
        # Note: 'Baseline_FVC' must be present in df
        base_fvc = (row["Baseline_FVC"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Relative Time
        # In train/val, 'Weeks' is relative. In test, we construct it.
        weeks = row["Weeks"]
        rel_time = weeks * Config.TIME_SCALE
        rel_time_abs = abs(rel_time)

        # Assemble Tabular Vector: [BaseFVC, RelTime, Age, Sex, Smoke]
        # Note: We pass raw values; the model handles embedding/projection if needed.
        # But here we return a float vector for the MLP.
        # Categoricals are usually one-hot encoded or embedded.
        # For this MLP architecture, we'll pass them as floats or handle in model.
        # The prompt description says: "Input: Baseline FVC, Relative Time, Age, Sex, SmokingStatus"
        # We will return them as a tensor.
        tab_vector = np.array(
            [base_fvc, rel_time, age, float(sex), float(smoke)], dtype=np.float32
        )

        # 3. Target Preparation
        target_fvc = 0.0
        target_sigma = 0.0  # Dummy

        if self.mode in ["train", "val"]:
            raw_fvc = row["FVC"]
            # Z-score standardization for target
            target_fvc = (raw_fvc - Config.TARGET_MEAN) / Config.TARGET_STD

        # Return dictionary or tuple.
        # Model expects: image, tabular_features, relative_time_abs (for shortcut)
        return {
            "image": torch.tensor(img_tensor, dtype=torch.float32),
            "tabular": torch.tensor(tab_vector, dtype=torch.float32),
            "time_abs": torch.tensor([rel_time_abs], dtype=torch.float32),
            "target": torch.tensor([target_fvc], dtype=torch.float32),
            "patient_week": f"{patient_id}_{weeks}",  # For tracking in test
        }


def add_baseline_fvc(df):
    """
    Enriches the dataframe with a 'Baseline_FVC' column.
    For each patient, Baseline_FVC is the FVC value at the visit closest to Weeks=0.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # Identify baseline rows: min(abs(Weeks)) per patient
    # We create a temporary column for absolute weeks
    df["AbsWeeks"] = df["Weeks"].abs()

    # Find the index of the minimum AbsWeeks for each patient
    baseline_indices = df.groupby("Patient")["AbsWeeks"].idxmin()

    # Extract Patient and FVC from these baseline rows
    baseline_df = df.loc[baseline_indices, ["Patient", "FVC"]].rename(
        columns={"FVC": "Baseline_FVC"}
    )

    # Merge back to original dataframe
    df = df.merge(baseline_df, on="Patient", how="left")

    # Cleanup
    df = df.drop(columns=["AbsWeeks"])
    return df


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for Train, Val, and Test sets.

    Args:
        debug (bool): If True, subsamples data for rapid testing.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_meta_df = pd.read_csv(Config.TEST_META_PATH)

    # 2. Preprocess Metadata (Add Baseline FVC)
    # Train/Val contain history, so we calculate baseline from history
    train_df = add_baseline_fvc(train_df)
    val_df = add_baseline_fvc(val_df)

    # Test metadata contains ONLY the baseline visit.
    # So FVC in test_meta_df IS the Baseline_FVC.
    test_meta_df["Baseline_FVC"] = test_meta_df["FVC"]

    # 3. Construct Test Set (Explode for Submission Weeks)
    # We need to predict for every Patient_Week in sample_submission.csv
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))

    # Parse Patient and Weeks from 'Patient_Week' column
    # Format: ID000..._12
    # We split on the *last* underscore to be safe
    split_data = sample_sub["Patient_Week"].str.rsplit("_", n=1, expand=True)
    sample_sub["Patient"] = split_data[0]
    sample_sub["Weeks"] = split_data[1].astype(int)

    # Merge with test metadata to get static features (Age, Sex, Image Path, Baseline FVC)
    # Note: test_meta_df has one row per patient.
    test_df = sample_sub.merge(
        test_meta_df.drop(columns=["Weeks", "FVC"]), on="Patient", how="left"
    )

    # 4. Debug Subsampling
    if debug:
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 5. Create Datasets
    train_dataset = LungDataset(train_df, mode="train", cache_images=True)
    val_dataset = LungDataset(val_df, mode="val", cache_images=True)
    test_dataset = LungDataset(test_df, mode="test", cache_images=True)

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

    return train_loader, val_loader, test_loader
