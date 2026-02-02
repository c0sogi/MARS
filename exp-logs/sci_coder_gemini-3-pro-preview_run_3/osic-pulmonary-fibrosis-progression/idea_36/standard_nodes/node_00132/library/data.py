import os
import cv2
import numpy as np
import pandas as pd
import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import seed_everything

# EDA Statistics for Normalization
AGE_MEAN = 67.58
AGE_STD = 6.63


def read_dicom(path, img_size=Config.IMG_SIZE):
    """
    Reads a DICOM file, applies lung windowing, and resizes.
    """
    try:
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)

        # Rescale to Hounsfield Units (HU)
        intercept = getattr(dcm, "RescaleIntercept", -1024)
        slope = getattr(dcm, "RescaleSlope", 1)
        image = slope * image + intercept

        # Apply Lung Window
        level = Config.WINDOW_LEVEL
        width = Config.WINDOW_WIDTH
        lower = level - width // 2
        upper = level + width // 2

        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)  # Normalize to [0, 1]

        # Resize
        image = cv2.resize(image, (img_size, img_size))
        return image
    except Exception as e:
        # Return a blank image in case of read failure
        return np.zeros((img_size, img_size), dtype=np.float32)


def process_patient_images(patient_id, image_dir_rel, cache_dir, load_cached_data=True):
    """
    Loads, processes, and caches patient images (3 slices).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{patient_id}.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            # Cite debug_lesson_11: Invalidate Stale Caches When Modifying Data Pipeline Configurations
            if data.shape == (Config.IMG_SIZE, Config.IMG_SIZE, 3):
                # Cite debug_lesson_2: Explicitly Sanitize Data Types When Loading Cached Arrays
                return data.astype(np.float32)
        except Exception:
            pass  # Fallback to processing if load fails

    # 2. Process from scratch
    full_img_dir = os.path.join(Config.INPUT_DIR, image_dir_rel)

    if not os.path.exists(full_img_dir):
        # Fallback for missing directory: return zeros
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # List and sort DICOM files
    files = [f for f in os.listdir(full_img_dir) if f.endswith(".dcm")]
    # Sort by file number (assuming format 1.dcm, 2.dcm, etc.)
    try:
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        files.sort()  # Fallback to string sort

    if not files:
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.float32)

    # Load all slices and calculate lung area
    # To optimize, we read, resize, and store in memory (images are small after resize)
    loaded_images = []
    lung_areas = []

    for f in files:
        img = read_dicom(os.path.join(full_img_dir, f))
        loaded_images.append(img)
        # Heuristic for lung area: pixels with value < 0.6 (since lung is dark/low HU)
        # After windowing (-600 center), lung tissue is around 0.5. Air is ~0.2.
        # We count pixels that are "not tissue/bone" (which are bright, >0.8)
        # and "not background air" (if segmented).
        # Simple heuristic: sum of pixels in range [0.1, 0.8]
        mask = (img > 0.1) & (img < 0.9)
        lung_areas.append(np.sum(mask))

    # Slice Selection
    num_slices = len(loaded_images)
    if num_slices == 0:
        selected_indices = [0, 0, 0]
    else:
        # Find Anchor (Max Area)
        idx_max = np.argmax(lung_areas)
        max_area = lung_areas[idx_max]

        # Threshold for boundary slices
        threshold = 0.5 * max_area

        # Find valid range
        valid_indices = [i for i, area in enumerate(lung_areas) if area > threshold]

        if not valid_indices:
            valid_indices = [idx_max]

        min_valid = valid_indices[0]
        max_valid = valid_indices[-1]

        # Select 3 slices: [Lower Boundary, Anchor, Upper Boundary]
        # We ensure they are distinct if possible
        idx_lower = min_valid
        idx_upper = max_valid

        # If not enough spread, just take adjacent or duplicate
        if num_slices >= 3:
            # Try to pick equidistant if range is large, else boundaries
            selected_indices = [idx_lower, idx_max, idx_upper]
        else:
            # Pad with max if fewer slices
            selected_indices = [idx_max] * 3
            if num_slices == 2:
                selected_indices = [0, 1, 1]

    # Stack selected slices
    final_volume = np.stack(
        [loaded_images[i] for i in selected_indices], axis=-1
    )  # (H, W, 3)

    # 3. Save to cache
    np.save(cache_path, final_volume)

    return final_volume


class OSICDataset(Dataset):
    def __init__(self, df, cache_dir, mode="train", baseline_lookup=None):
        """
        Args:
            df: DataFrame containing metadata.
            cache_dir: Directory to store/load cached images.
            mode: 'train', 'val', or 'test'.
            baseline_lookup: Dict mapping PatientID -> {FVC, Weeks, Age, Sex, SmokingStatus}.
                             If None, computed from df (assuming df contains full history).
        """
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.mode = mode

        # Build or use baseline lookup
        if baseline_lookup is None:
            self.baseline_lookup = self._build_baseline_lookup(self.df)
        else:
            self.baseline_lookup = baseline_lookup

    def _build_baseline_lookup(self, df):
        lookup = {}
        # Group by patient and find the row with minimum Weeks
        for patient_id, group in df.groupby("Patient"):
            # Sort by Weeks to find baseline
            group = group.sort_values("Weeks")
            baseline_row = group.iloc[0]

            lookup[patient_id] = {
                "FVC": baseline_row["FVC"],
                "Weeks": baseline_row["Weeks"],
                "Age": baseline_row["Age"],
                "Sex": baseline_row["Sex"],
                "SmokingStatus": baseline_row["SmokingStatus"],
            }
        return lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row["Patient"]

        # 1. Load Image
        # Image path in metadata is relative, e.g., "train/ID..."
        # We extract the directory relative to input root
        image_dir_rel = row["image_path"]
        image = process_patient_images(
            patient_id, image_dir_rel, self.cache_dir, load_cached_data=True
        )

        # Transpose to (C, H, W) for PyTorch
        image = np.transpose(image, (2, 0, 1))

        # 2. Tabular Features
        # Retrieve baseline info
        base_info = self.baseline_lookup.get(patient_id)
        if base_info is None:
            # Fallback if patient not in lookup (should not happen in standard split)
            base_info = {
                "FVC": row["FVC"],
                "Weeks": row["Weeks"],
                "Age": row["Age"],
                "Sex": row["Sex"],
                "SmokingStatus": row["SmokingStatus"],
            }

        # Feature 1: Baseline FVC (Standardized)
        base_fvc = (base_info["FVC"] - Config.TARGET_MEAN) / Config.TARGET_STD

        # Feature 2: Relative Time
        # t_rel = (Current_Week - Baseline_Week) * Scale
        current_week = row["Weeks"]
        base_week = base_info["Weeks"]
        rel_time = (current_week - base_week) * Config.TIME_SCALE

        # Feature 3: Age (Standardized)
        age = (base_info["Age"] - AGE_MEAN) / AGE_STD

        # Feature 4: Sex (Encoded)
        # Male: 0, Female: 1
        sex = 0.0 if base_info["Sex"] == "Male" else 1.0

        # Feature 5: SmokingStatus (Ordinal Encoded)
        # Never smoked: 0, Ex-smoker: 1, Currently smokes: 2
        smoke_map = {"Never smoked": 0.0, "Ex-smoker": 1.0, "Currently smokes": 2.0}
        smoking = smoke_map.get(base_info["SmokingStatus"], 0.0)

        tabular = np.array([base_fvc, rel_time, age, sex, smoking], dtype=np.float32)

        # 3. Target
        # FVC (Standardized)
        target_fvc_raw = row["FVC"]
        target_fvc_scaled = (target_fvc_raw - Config.TARGET_MEAN) / Config.TARGET_STD

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "tabular": torch.tensor(tabular, dtype=torch.float32),
            "target": torch.tensor(target_fvc_scaled, dtype=torch.float32),
            "target_raw": torch.tensor(
                target_fvc_raw, dtype=torch.float32
            ),  # For metric calculation
            "patient_week": f"{patient_id}_{current_week}",
        }


def get_dataloaders(train_csv_path, val_csv_path, batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoaders for training and validation.
    """
    seed_everything(Config.SEED)

    # Load Metadata
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Cache Directory
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")

    # Create Datasets
    # Note: We build the baseline lookup from the training set primarily.
    # Ideally, validation patients are distinct, so they calculate their own baselines.
    # The OSICDataset handles baseline construction internally if not provided.

    train_dataset = OSICDataset(train_df, cache_dir=cache_dir, mode="train")

    val_dataset = OSICDataset(val_df, cache_dir=cache_dir, mode="val")

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
