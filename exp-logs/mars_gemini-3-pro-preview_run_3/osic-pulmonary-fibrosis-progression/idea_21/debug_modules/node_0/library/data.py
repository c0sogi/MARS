import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# -------------------------------------------------------------------------
# Image Processing Utilities
# -------------------------------------------------------------------------


def load_scans(dcm_dir):
    """Loads DICOM files from a directory and sorts them by InstanceNumber."""
    files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]
    if not files:
        return []

    scans = []
    for f in files:
        try:
            ds = pydicom.dcmread(os.path.join(dcm_dir, f))
            scans.append(ds)
        except Exception:
            continue

    # Sort by Instance Number (Z-position)
    scans.sort(key=lambda x: int(x.InstanceNumber))
    return scans


def get_pixels_hu(scans):
    """Converts raw DICOM pixel data to Hounsfield Units."""
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def select_slices_and_resize(image_hu, target_size=260):
    """
    Selects 3 slices (Top-Boundary, Anchor, Bottom-Boundary) based on lung area.
    Resizes them to target_size x target_size.
    """
    # Threshold to identify lung air (approx -1000 to -500 HU)
    # We use < -500 as a rough proxy for lung area
    lung_mask = image_hu < -500
    slice_areas = lung_mask.sum(axis=(1, 2))

    max_area = slice_areas.max()
    if max_area == 0:
        # Fallback if no lung detected
        indices = [0, len(image_hu) // 2, len(image_hu) - 1]
    else:
        # Find slices with at least 50% of max lung area
        valid_indices = np.where(slice_areas > 0.5 * max_area)[0]

        if len(valid_indices) == 0:
            indices = [0, len(image_hu) // 2, len(image_hu) - 1]
        else:
            anchor_idx = valid_indices[np.argmax(slice_areas[valid_indices])]
            top_idx = valid_indices[0]
            bottom_idx = valid_indices[-1]

            # Ensure distinct indices if possible, though duplicates are fine for thin ranges
            indices = [top_idx, anchor_idx, bottom_idx]
            # Sort spatially
            indices.sort()

    selected_slices = []
    for idx in indices:
        sl = image_hu[idx]

        # Clip to lung window [-1000, 400] roughly
        sl = np.clip(sl, -1000, 400)

        # Normalize to [0, 1]
        sl = (sl - (-1000)) / (400 - (-1000))

        # Resize
        sl = cv2.resize(sl, (target_size, target_size))
        selected_slices.append(sl)

    # Stack to (3, H, W)
    img_tensor = np.stack(selected_slices, axis=0)
    return img_tensor.astype(np.float32)


def process_and_cache_images(patient_ids, input_dir, cache_dir, load_cached_data=True):
    """
    Iterates over patients, processes their CT scans, and caches the result.
    """
    os.makedirs(cache_dir, exist_ok=True)

    for pid in patient_ids:
        cache_path = os.path.join(cache_dir, f"{pid}.npy")

        if load_cached_data and os.path.exists(cache_path):
            continue

        # Determine path - try train then test
        path_train = os.path.join(input_dir, "train", pid)
        path_test = os.path.join(input_dir, "test", pid)

        if os.path.exists(path_train):
            dcm_dir = path_train
        elif os.path.exists(path_test):
            dcm_dir = path_test
        else:
            # Create dummy if missing (robustness)
            dummy = np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            np.save(cache_path, dummy)
            continue

        try:
            scans = load_scans(dcm_dir)
            if not scans:
                raise ValueError("No .dcm files")
            hu = get_pixels_hu(scans)
            processed = select_slices_and_resize(hu, Config.IMG_SIZE)
            np.save(cache_path, processed)
        except Exception as e:
            # Save zeros to avoid crash during training
            dummy = np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            np.save(cache_path, dummy)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------


class LungDataset(Dataset):
    def __init__(self, df, cache_dir, transform=None, mode="train"):
        """
        Args:
            df: DataFrame containing patient data.
            cache_dir: Directory where processed images are stored.
            transform: Optional transforms.
            mode: 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.transform = transform
        self.mode = mode

        # Pre-compute categorical encodings
        self.sex_map = {"Male": 0, "Female": 1}
        # Smoking: Ex-smoker, Never smoked, Currently smokes
        self.smoking_map = {
            "Ex-smoker": [1, 0, 0],
            "Never smoked": [0, 1, 0],
            "Currently smokes": [0, 0, 1],
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["Patient"]

        # 1. Load Image
        cache_path = os.path.join(self.cache_dir, f"{pid}.npy")
        if os.path.exists(cache_path):
            img = np.load(cache_path)
        else:
            img = np.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)

        # 2. Clinical Features
        # Baseline FVC (Standardized)
        base_fvc = (row["Baseline_FVC"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Age (Standardized)
        age = (row["Age"] - Config.AGE_MEAN) / Config.AGE_STD

        # Relative Time (Scaled)
        if "Relative_Weeks" in row:
            raw_time = row["Relative_Weeks"]
        else:
            raw_time = row["Weeks"]

        t_rel = raw_time * Config.TIME_SCALE

        # Sex
        sex = self.sex_map.get(row["Sex"], 0)

        # Smoking
        smoking = self.smoking_map.get(row["SmokingStatus"], [0, 0, 0])

        # Construct Clinical Vector
        # [BaseFVC, Time, Age, Sex, Smoke1, Smoke2, Smoke3]
        clinical = np.array([base_fvc, t_rel, age, sex] + smoking, dtype=np.float32)

        # 3. Target
        if self.mode != "test":
            target_raw = row["FVC"]
            target = (target_raw - Config.TARGET_MEAN) / Config.TARGET_STD
        else:
            target = 0.0  # Dummy

        # Convert to tensors
        img_tensor = torch.tensor(img, dtype=torch.float32)
        clinical_tensor = torch.tensor(clinical, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        time_tensor = torch.tensor(t_rel, dtype=torch.float32)  # For shortcut

        return img_tensor, clinical_tensor, time_tensor, target_tensor


# -------------------------------------------------------------------------
# Data Preparation Logic
# -------------------------------------------------------------------------


def preprocess_dataframe(df, is_train=True):
    """
    Augments the dataframe with Baseline_FVC and calculates Relative_Weeks.
    Finds the baseline FVC for each patient (closest to Week 0).
    """
    # Ensure sorted
    df = df.sort_values(["Patient", "Weeks"])

    patient_baselines = []
    for pid, group in df.groupby("Patient"):
        # Find row with Weeks closest to 0 to serve as baseline
        group["dist_to_0"] = group["Weeks"].abs()
        baseline_row = group.loc[group["dist_to_0"].idxmin()]

        patient_baselines.append(
            {
                "Patient": pid,
                "Baseline_FVC": baseline_row["FVC"],
                "Baseline_Week": baseline_row["Weeks"],
            }
        )

    baseline_df = pd.DataFrame(patient_baselines)

    # Merge back
    df = df.merge(baseline_df, on="Patient", how="left")

    # Calculate Relative Weeks
    df["Relative_Weeks"] = df["Weeks"] - df["Baseline_Week"]

    return df


def get_dataloaders(
    train_batch_size=Config.BATCH_SIZE,
    val_batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Main entry point to get DataLoaders.
    Handles caching of images and preprocessing of dataframes.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # 2. Preprocess DataFrames
    train_df = preprocess_dataframe(train_df, is_train=True)
    val_df = preprocess_dataframe(val_df, is_train=True)

    # 3. Cache Images
    all_patients = pd.concat([train_df["Patient"], val_df["Patient"]]).unique()

    process_and_cache_images(
        all_patients, Config.INPUT_DIR, Config.CACHE_DIR, load_cached_data
    )

    # 4. Create Datasets
    train_dataset = LungDataset(train_df, Config.CACHE_DIR, mode="train")
    val_dataset = LungDataset(val_df, Config.CACHE_DIR, mode="val")

    # 5. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates a DataLoader for the test set.
    Expands the test metadata (baseline only) to include all required prediction weeks.
    """
    test_df = pd.read_csv(Config.TEST_CSV)

    # Cache test images
    process_and_cache_images(
        test_df["Patient"].unique(),
        Config.INPUT_DIR,
        Config.CACHE_DIR,
        load_cached_data=True,
    )

    # Expand DataFrame for submission
    # We need to predict for weeks -12 to 133 for every patient
    submission_rows = []
    for _, row in test_df.iterrows():
        pid = row["Patient"]
        base_fvc = row["FVC"]
        base_week = row["Weeks"]  # Usually 0

        # Create rows for weeks -12 to 133
        for w in range(-12, 134):
            submission_rows.append(
                {
                    "Patient": pid,
                    "Weeks": w,
                    "Baseline_FVC": base_fvc,
                    "Baseline_Week": base_week,
                    "Age": row["Age"],
                    "Sex": row["Sex"],
                    "SmokingStatus": row["SmokingStatus"],
                    "Relative_Weeks": w - base_week,
                    "Patient_Week": f"{pid}_{w}",
                }
            )

    expanded_test_df = pd.DataFrame(submission_rows)

    test_dataset = LungDataset(expanded_test_df, Config.CACHE_DIR, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_loader, expanded_test_df
