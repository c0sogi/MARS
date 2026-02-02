import os
import re
import glob
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import set_seed

# Try importing pydicom, handle if missing (as per prompt constraints)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
CACHE_DIR = "./working/idea_10"
IMG_SIZE = 224
NUM_CHANNELS = 12  # 4 modalities * 3 slices
STRIDE = 5
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

# ------------------------------------------------------------------------------
# Helper Functions: IO & Processing
# ------------------------------------------------------------------------------


def read_dicom_raw_heuristic(path):
    """
    Fallback reader that attempts to read DICOM pixel data by inferring
    dimensions from file size (assuming uint16).
    Supports common BraTS resolutions: 256x256 and 512x512.
    """
    try:
        file_size = os.path.getsize(path)

        # Candidates for (Height, Width)
        candidates = [(512, 512), (256, 256)]

        with open(path, "rb") as f:
            content = f.read()

        for h, w in candidates:
            expected_pixels = h * w
            expected_bytes = expected_pixels * 2  # uint16 = 2 bytes

            # Header size = Total - PixelBytes
            header_size = file_size - expected_bytes

            # DICOM headers are usually between 128 bytes and 4KB
            if 128 <= header_size < 4096:
                # Extract last expected_bytes
                pixel_bytes = content[-expected_bytes:]
                arr = np.frombuffer(pixel_bytes, dtype=np.uint16)
                if arr.size == expected_pixels:
                    return arr.reshape(h, w)

    except Exception:
        pass
    return None


def load_dicom_array(path):
    """
    Reads a DICOM file and returns the raw numpy array.
    Tries pydicom first, then raw binary fallback.
    """
    img = None

    # 1. Try pydicom
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
        except Exception:
            img = None

    # 2. Try Raw Fallback if pydicom failed or missing
    if img is None:
        img = read_dicom_raw_heuristic(path)

    return img


def get_slice_intensity(path):
    """
    Returns the sum of pixel intensities for a slice.
    Used for ROI selection.
    """
    img = load_dicom_array(path)
    if img is not None:
        return np.sum(img)
    return 0.0


def load_and_process_slice(path, size=(IMG_SIZE, IMG_SIZE)):
    """
    Loads a slice, normalizes it to 0-255, and resizes it.
    Returns a (H, W) uint8 array.
    """
    img = load_dicom_array(path)

    if img is None:
        # Return black slice if unreadable
        return np.zeros(size, dtype=np.uint8)

    # Normalize Min-Max to 0-255
    if np.max(img) > np.min(img):
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)

    # Resize with Area Interpolation (Noise Suppression)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def get_sorted_files(folder_path):
    """
    Returns list of files sorted by the integer index in 'Image-N.dcm'.
    """
    if not os.path.exists(folder_path):
        return []

    files = glob.glob(os.path.join(folder_path, "*.dcm"))

    def extract_number(p):
        # Extract N from Image-N.dcm
        match = re.search(r"Image-(\d+)\.dcm", os.path.basename(p))
        if match:
            return int(match.group(1))
        return 0

    return sorted(files, key=extract_number)


def select_roi_indices(flair_path):
    """
    Determines the anchor index based on the FLAIR modality.
    Logic:
    1. Calculate sum of intensity for all slices.
    2. Smooth profile with Moving Average (window=5).
    3. Find peak within 15%-85% depth range.
    Returns: [anchor-5, anchor, anchor+5]
    """
    files = get_sorted_files(flair_path)
    num_files = len(files)

    if num_files == 0:
        return [0, 0, 0]

    # Calculate intensities
    intensities = []
    for f in files:
        intensities.append(get_slice_intensity(f))

    intensities = np.array(intensities)

    # Smooth
    window_size = 5
    if len(intensities) >= window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(intensities, kernel, mode="same")
    else:
        smoothed = intensities

    # Boundary Exclusion
    start_idx = int(num_files * 0.15)
    end_idx = int(num_files * 0.85)

    if start_idx >= end_idx:
        start_idx, end_idx = 0, num_files

    # Find peak
    valid_range = smoothed[start_idx:end_idx]
    if len(valid_range) > 0:
        peak_offset = np.argmax(valid_range)
        anchor_idx = start_idx + peak_offset
    else:
        anchor_idx = num_files // 2

    # Return indices
    return [anchor_idx - STRIDE, anchor_idx, anchor_idx + STRIDE]


# ------------------------------------------------------------------------------
# Dataset Processing
# ------------------------------------------------------------------------------


def process_dataset(metadata_df, cache_name, load_cached_data=True):
    """
    Processes the dataset defined in metadata_df.
    Caches result to ./working/idea_10/{cache_name}_data.npy
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    data_path = os.path.join(CACHE_DIR, f"{cache_name}_data.npy")
    labels_path = os.path.join(CACHE_DIR, f"{cache_name}_labels.npy")

    # Check cache validity
    if load_cached_data and os.path.exists(data_path) and os.path.exists(labels_path):
        data = np.load(data_path)
        labels = np.load(labels_path)
        # Verify length matches current metadata (crucial for reruns with larger test sets)
        if len(data) == len(metadata_df):
            return data, labels

    # Process from scratch
    data_list = []
    labels_list = []

    for _, row in metadata_df.iterrows():
        # 1. Determine ROI using FLAIR
        flair_path = os.path.join(INPUT_DIR, row["path_FLAIR"])
        roi_indices = select_roi_indices(flair_path)

        subject_slices = []

        # 2. Extract slices from all modalities
        for mod in MODALITIES:
            mod_path = os.path.join(INPUT_DIR, row[f"path_{mod}"])
            files = get_sorted_files(mod_path)
            num_files = len(files)

            for idx in roi_indices:
                # Absolute Indexing with Clamping
                if num_files > 0:
                    read_idx = min(max(0, idx), num_files - 1)
                    img = load_and_process_slice(files[read_idx])
                else:
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                subject_slices.append(img)

        # Stack to (12, H, W)
        subject_vol = np.stack(subject_slices, axis=0)
        data_list.append(subject_vol)

        # Label
        if "MGMT_value" in row:
            labels_list.append(row["MGMT_value"])
        else:
            labels_list.append(0.5)  # Placeholder

    data_array = np.array(data_list, dtype=np.uint8)
    labels_array = np.array(labels_list, dtype=np.float32)

    # Save to cache
    np.save(data_path, data_array)
    np.save(labels_path, labels_array)

    return data_array, labels_array


# ------------------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------------------


class MGMTDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # (N, 12, H, W) uint8
        self.labels = labels  # (N,) float32
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert (12, H, W) -> (H, W, 12) for Albumentations
        img = self.data[idx].transpose(1, 2, 0)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # (12, H, W) tensor
        else:
            # Manual conversion
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # Normalize 0-255 uint8 to 0.0-1.0 float
        if isinstance(img, torch.Tensor):
            img = img.float() / 255.0
        else:
            img = img.astype(np.float32) / 255.0

        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(phase):
    """
    Returns transformations.
    Train: Flip, Rotate (+/- 15 deg).
    Eval: None.
    """
    if phase == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


# ------------------------------------------------------------------------------
# Main Interface
# ------------------------------------------------------------------------------


def get_dataloaders(batch_size=32, load_cached_data=True, debug_limit=None):
    """
    Orchestrates data loading, processing, and DataLoader creation.
    """
    set_seed(42)

    # Load Metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    if debug_limit:
        train_df = train_df.head(debug_limit)
        val_df = val_df.head(debug_limit)
        test_df = test_df.head(debug_limit)

    # Process Data
    train_data, train_labels = process_dataset(train_df, "train", load_cached_data)
    val_data, val_labels = process_dataset(val_df, "val", load_cached_data)
    test_data, test_labels = process_dataset(test_df, "test", load_cached_data)

    # Create Datasets
    train_ds = MGMTDataset(train_data, train_labels, transform=get_transforms("train"))
    val_ds = MGMTDataset(val_data, val_labels, transform=get_transforms("val"))
    test_ds = MGMTDataset(test_data, test_labels, transform=get_transforms("test"))

    # Create Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
