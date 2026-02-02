import os
import cv2
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# -------------------------------------------------------------------------
# Constants & Encodings
# -------------------------------------------------------------------------
SEX_MAP = {"Male": 0, "Female": 1}
SMOKING_MAP = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}


def get_lung_window(dcm):
    """
    Applies lung windowing to a DICOM file.
    Window Level: -600, Window Width: 1500
    """
    # Convert to Hounsfield Units (HU)
    image = dcm.pixel_array.astype(np.float32)

    # Apply RescaleSlope and RescaleIntercept if they exist
    slope = getattr(dcm, "RescaleSlope", 1)
    intercept = getattr(dcm, "RescaleIntercept", 0)
    image = slope * image + intercept

    # Lung Window
    center = -600
    width = 1500
    lower = center - width // 2
    upper = center + width // 2

    image = np.clip(image, lower, upper)

    # Normalize to 0-1
    image = (image - lower) / (upper - lower)
    return image


def process_patient_images(patient_id, image_dir_rel, cache_dir, load_cached_data=True):
    """
    Selects 3 adaptive slices (Anchor + 2 Boundaries) from DICOMs.
    Caches the result as a .npy file.
    """
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_image_dir = os.path.join(Config.INPUT_ROOT, image_dir_rel)

    # List all DICOM files
    if not os.path.exists(full_image_dir):
        # Fallback for missing directories (should not happen based on metadata check)
        # Return a blank volume
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    files = [f for f in os.listdir(full_image_dir) if f.endswith(".dcm")]
    if not files:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Load and sort by InstanceNumber (Z-position)
    dcms = []
    for f in files:
        try:
            d = pydicom.dcmread(os.path.join(full_image_dir, f))
            dcms.append(d)
        except:
            continue

    # Sort by InstanceNumber or ImagePositionPatient Z
    dcms.sort(
        key=lambda x: int(x.InstanceNumber) if hasattr(x, "InstanceNumber") else 0
    )

    if not dcms:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Calculate Lung Area for each slice to find Anchor
    # We use a simple threshold on the normalized windowed image to estimate "lung-like" area
    # Lung is dark in CT, but after windowing and 0-1 norm, air is 0, tissue is 1.
    # Wait, lung window: -1350 (black) to 150 (white). Lung tissue is around -600 (gray).
    # Air is -1000.
    # Let's use a threshold on the raw HU or just use the middle slices if heuristic fails.

    processed_slices = []
    areas = []

    for d in dcms:
        try:
            img = get_lung_window(d)
            img_resized = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            processed_slices.append(img_resized)

            # Estimate lung area: pixels between 0.1 and 0.8 roughly (excluding pure air and bone)
            # This is a heuristic. Alternatively, just use the slice with most "structure".
            # A common heuristic is simply checking non-background pixels.
            # Since we normalized 0-1 where 0 is -1350HU (air), lung is around 0.5 (-600HU).
            # We count pixels < 0.6 and > 0.05
            area = np.sum((img_resized > 0.05) & (img_resized < 0.7))
            areas.append(area)
        except Exception:
            continue

    if not processed_slices:
        return np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    areas = np.array(areas)

    # Select Anchor (Max Area)
    if len(areas) > 0:
        idx_anchor = np.argmax(areas)
        max_area = areas[idx_anchor]
    else:
        idx_anchor = 0
        max_area = 0

    # Select Boundaries (closest slices with area < 50% of max)
    # If max_area is too small, just take indices at 25% and 75% of volume
    n_slices = len(processed_slices)

    if max_area < 100:  # Fallback for bad segmentation/empty images
        indices = [int(n_slices * 0.3), int(n_slices * 0.5), int(n_slices * 0.7)]
    else:
        # Search Up
        idx_upper = idx_anchor
        for i in range(idx_anchor + 1, n_slices):
            if areas[i] < 0.5 * max_area:
                idx_upper = i
                break

        # Search Down
        idx_lower = idx_anchor
        for i in range(idx_anchor - 1, -1, -1):
            if areas[i] < 0.5 * max_area:
                idx_lower = i
                break

        indices = [idx_lower, idx_anchor, idx_upper]

    # Gather slices
    final_volume = []
    for idx in indices:
        # Clamp index
        idx = max(0, min(idx, n_slices - 1))
        final_volume.append(processed_slices[idx])

    # Stack to (3, H, W)
    volume_np = np.stack(final_volume, axis=0).astype(np.float32)

    # 3. Save to cache
    try:
        np.save(cache_path, volume_np)
    except Exception:
        pass  # Disk full or permission error, ignore

    return volume_np


class OSICDataset(Dataset):
    def __init__(self, df, stats, mode="train", cache_dir=Config.CACHE_DIR):
        """
        Args:
            df: DataFrame containing metadata.
            stats: Dictionary containing normalization statistics (mean/std).
            mode: 'train', 'val', or 'test'.
            cache_dir: Directory to save/load processed images.
        """
        self.df = df.reset_index(drop=True)
        self.stats = stats
        self.mode = mode
        self.cache_dir = cache_dir

        # Precompute tabular features to save time in __getitem__
        self.tabular_feats = []
        self.targets = []
        self.patient_ids = []
        self.image_paths = []

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        # Group by patient to find baseline
        # Note: The input DF should already have Baseline info merged if it's train/val
        # But to be safe and robust, we recalculate baseline logic here.

        # We need to handle the case where df is just a list of queries (like submission)
        # or a list of measurements (train).

        # Strategy:
        # 1. Identify unique patients.
        # 2. For each patient, find the baseline row (min Weeks).
        # 3. Process every row in df relative to that patient's baseline.

        patient_groups = self.df.groupby("Patient")

        processed_rows = []

        for patient_id, group in patient_groups:
            # Find baseline row
            # In train/val, we have history. In test, we have 1 row (which is baseline).
            baseline_row = group.loc[group["Weeks"].idxmin()]

            base_fvc = baseline_row["FVC"]
            base_weeks = baseline_row["Weeks"]
            base_age = baseline_row["Age"]
            sex = baseline_row["Sex"]
            smoking = baseline_row["SmokingStatus"]
            image_path = baseline_row["image_path"]

            # Encode Categoricals
            sex_enc = SEX_MAP.get(sex, 0)
            smoke_enc = SMOKING_MAP.get(smoking, 1)  # Default to Never smoked

            # One-hot smoking
            smoke_oh = [0, 0, 0]
            smoke_oh[smoke_enc] = 1

            # Normalize Baseline Features using Global Stats
            base_fvc_norm = (base_fvc - self.stats["fvc_mean"]) / self.stats["fvc_std"]
            age_norm = (base_age - self.stats["age_mean"]) / self.stats["age_std"]

            for _, row in group.iterrows():
                # Relative Time
                t_rel = (row["Weeks"] - base_weeks) * Config.TIME_SCALER

                # Feature Vector: [Base_FVC, t_rel, Age, Sex, Smoke_0, Smoke_1, Smoke_2]
                feats = [base_fvc_norm, t_rel, age_norm, sex_enc] + smoke_oh

                # Target
                if self.mode != "test":
                    target_raw = row["FVC"]
                    # Z-score standardize target
                    target_norm = (target_raw - self.stats["fvc_mean"]) / self.stats[
                        "fvc_std"
                    ]
                else:
                    target_norm = 0.0  # Dummy

                self.tabular_feats.append(np.array(feats, dtype=np.float32))
                self.targets.append(target_norm)
                self.patient_ids.append(patient_id)
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.tabular_feats)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        image_path = self.image_paths[idx]

        # Load Image
        img = process_patient_images(patient_id, image_path, self.cache_dir)

        # Get Features
        tab = self.tabular_feats[idx]
        target = self.targets[idx]

        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "tabular": torch.tensor(tab, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
            "patient_id": patient_id,
        }


def get_dataloaders(batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS):
    """
    Creates DataLoaders for Train, Val, and Test sets.
    Computes standardization statistics from the Training set.
    """
    seed_everything(Config.SEED)
    Config.setup_directories()

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)
    test_df = pd.read_csv(Config.TEST_METADATA)

    # Compute Statistics from Training Data ONLY
    # We use all FVC measurements in train for target stats
    # We use unique patients in train for Age stats to avoid bias from frequent visitors?
    # Actually, standard practice is just simple column stats.

    fvc_all = train_df["FVC"].values
    age_all = train_df["Age"].values

    stats = {
        "fvc_mean": float(np.mean(fvc_all)),
        "fvc_std": float(np.std(fvc_all)),
        "age_mean": float(np.mean(age_all)),
        "age_std": float(np.std(age_all)),
    }

    print("Normalization Statistics (Train):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Create Datasets
    train_ds = OSICDataset(train_df, stats, mode="train")
    val_ds = OSICDataset(val_df, stats, mode="val")

    # For Test, we just pass the baseline metadata.
    # The inference loop will handle expanding this to multiple time steps if needed,
    # or the dataset can handle it.
    # Given the prompt "Predict every patient's FVC measurement for every possible week",
    # usually inference scripts iterate over weeks.
    # Here, we provide the dataset that returns the baseline state.
    test_ds = OSICDataset(test_df, stats, mode="test")

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,  # Process one patient at a time for inference convenience
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, stats
