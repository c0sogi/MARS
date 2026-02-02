import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Try importing pydicom, handle if missing (though analysis suggests it's present)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# Constants
CACHE_DIR = "./working/idea_26/"
INPUT_DIR = "./input"
IMG_SIZE = 224
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]


def natural_sort_key(s):
    """
    Sorts strings that contain numbers in a natural way (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def read_dicom(path):
    """
    Reads a DICOM file and returns a numpy array.
    Falls back to cv2 if pydicom is not available or fails.
    """
    if not os.path.exists(path):
        return None

    img = None
    # Method 1: pydicom
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            pass

    # Method 2: cv2
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    return img


def is_brain_slice(path):
    """
    Checks if a slice contains brain tissue (max pixel > 0).
    """
    img = read_dicom(path)
    if img is None:
        return False
    return np.max(img) > 0


def get_brain_roi_indices(file_paths):
    """
    Efficiently finds the start and end indices of the brain ROI.
    Uses a center-out check followed by binary search for boundaries to avoid reading all files.
    """
    n = len(file_paths)
    if n == 0:
        return 0, 0

    # 1. Find *any* slice with brain tissue
    # Check middle, then quarters, then strided scan
    checkpoints = [n // 2, n // 4, 3 * n // 4]
    found_idx = -1

    for idx in checkpoints:
        if is_brain_slice(file_paths[idx]):
            found_idx = idx
            break

    if found_idx == -1:
        # Fallback: Strided scan (every 10th slice)
        for idx in range(0, n, 10):
            if is_brain_slice(file_paths[idx]):
                found_idx = idx
                break

    if found_idx == -1:
        # No brain found in scan, return full range (or empty)
        return 0, n - 1

    # 2. Find Start Index (Binary Search between 0 and found_idx)
    low = 0
    high = found_idx
    start_index = found_idx

    while low <= high:
        mid = (low + high) // 2
        if is_brain_slice(file_paths[mid]):
            start_index = mid
            high = mid - 1
        else:
            low = mid + 1

    # 3. Find End Index (Binary Search between found_idx and n-1)
    low = found_idx
    high = n - 1
    end_index = found_idx

    while low <= high:
        mid = (low + high) // 2
        if is_brain_slice(file_paths[mid]):
            end_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return start_index, end_index


def get_transforms(phase="train"):
    """
    Returns Albumentations transforms.
    Strictly excludes translation (Shift) and Scaling to preserve anatomical anchoring.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


def process_metadata(df, split_name, limit_size=None):
    """
    Generates the specific file paths for the 9-channel input.
    """
    processed_rows = []

    # Optional debugging limit
    if limit_size is not None:
        df = df.head(limit_size)

    print(f"Processing {len(df)} subjects for {split_name} cache...")

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]
        subject_data = {
            "BraTS21ID": sid,
        }
        if "MGMT_value" in row:
            subject_data["MGMT_value"] = row["MGMT_value"]

        valid_subject = True

        for mod in MODALITIES:
            # Construct full path to modality folder
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)

            # Get all dcm files
            if os.path.exists(full_path):
                files = glob.glob(os.path.join(full_path, "*.dcm"))
                files.sort(key=natural_sort_key)
            else:
                files = []

            if not files:
                # If missing modality, we can't generate valid input.
                # For this competition, we assume data integrity or handle in dataset.
                # We will mark as invalid or fill with None.
                valid_subject = False
                break

            # Get ROI
            start_idx, end_idx = get_brain_roi_indices(files)
            roi_len = end_idx - start_idx

            # Calculate indices for 45%, 50%, 55%
            # Relative to ROI
            idx_45 = start_idx + int(roi_len * 0.45)
            idx_50 = start_idx + int(roi_len * 0.50)
            idx_55 = start_idx + int(roi_len * 0.55)

            # Clamp to valid range
            idx_45 = max(0, min(len(files) - 1, idx_45))
            idx_50 = max(0, min(len(files) - 1, idx_50))
            idx_55 = max(0, min(len(files) - 1, idx_55))

            # Store relative paths (to save space/portability)
            # files[i] is full path, we want relative to INPUT_DIR if possible,
            # or just absolute. Let's store absolute for simplicity in loading.
            subject_data[f"{mod}_45"] = files[idx_45]
            subject_data[f"{mod}_50"] = files[idx_50]
            subject_data[f"{mod}_55"] = files[idx_55]

        if valid_subject:
            processed_rows.append(subject_data)

    return pd.DataFrame(processed_rows)


def get_processed_metadata(
    metadata_path, split_name, load_cached_data=True, limit_size=None
):
    """
    Caching wrapper for metadata processing.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"cached_paths_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached metadata for {split_name} from {cache_file}")
        df = pd.read_parquet(cache_file)
        if limit_size is not None:
            df = df.head(limit_size)
        return df

    # Process from scratch
    print(f"Generating metadata for {split_name}...")
    df_raw = pd.read_csv(metadata_path)
    df_processed = process_metadata(df_raw, split_name, limit_size)

    # Save cache
    df_processed.to_parquet(cache_file, index=False)
    print(f"Saved cached metadata to {cache_file}")

    return df_processed


class BraTSDataset(Dataset):
    def __init__(self, df, phase="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing file paths for the 9 channels.
            phase (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.phase = phase
        self.transforms = get_transforms(phase)

        # Define channel order:
        # [FLAIR_45, T1wCE_45, T2w_45, FLAIR_50, T1wCE_50, T2w_50, FLAIR_55, T1wCE_55, T2w_55]
        # Note: T1w is available but the strategy description specifically mentioned:
        # "Channels 0-2: [FLAIR, T1wCE, T2w] at 45%" etc.
        # We will follow the strategy description strictly.
        self.channel_cols = [
            "flair_45",
            "t1wce_45",
            "t2w_45",
            "flair_50",
            "t1wce_50",
            "t2w_50",
            "flair_55",
            "t1wce_55",
            "t2w_55",
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        images = []
        for col in self.channel_cols:
            path = row[col]
            img = read_dicom(path)

            if img is None:
                # Fallback for corrupt/missing file: black image
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            else:
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32)

                # Independent Min-Max Normalization
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                else:
                    img = np.zeros_like(img)

            images.append(img)

        # Stack to (H, W, 9)
        # Albumentations expects (H, W, C)
        vol = np.stack(images, axis=-1)

        # Apply transforms
        augmented = self.transforms(image=vol)
        vol_tensor = augmented["image"]  # (9, H, W)

        # Get Label
        if self.phase != "test":
            label = torch.tensor(row["MGMT_value"], dtype=torch.float32)
            return vol_tensor, label
        else:
            # Return ID for submission tracking
            return vol_tensor, row["BraTS21ID"]
