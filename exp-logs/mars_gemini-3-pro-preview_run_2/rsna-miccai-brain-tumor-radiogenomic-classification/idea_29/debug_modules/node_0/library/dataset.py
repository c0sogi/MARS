import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    CHANNEL_CONFIG,
    ROI_DEPTH_RANGE,
    ROI_ANCHOR_MODALITY,
    IMG_SIZE,
    WORKING_DIR,
)
from library.dicom_utils import read_dicom_robust, process_image

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_image_index(filename):
    """
    Extracts the integer slice index from a DICOM filename (e.g., 'Image-123.dcm').
    Returns -1 if the pattern does not match.
    """
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return -1


def get_sorted_files(dir_path):
    """
    Returns a list of DICOM filenames in a directory, sorted by their slice index.
    """
    if not os.path.exists(dir_path):
        return []
    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]
    # Sort by the integer index extracted from the filename
    files.sort(key=lambda x: get_image_index(x))
    return files


# -----------------------------------------------------------------------------
# Raw Dataset Class (Handles DICOM IO and Logic)
# -----------------------------------------------------------------------------


class BrainTumorDataset(Dataset):
    """
    Handles the raw data ingestion, ROI selection, and Focal-Modality tensor construction.
    This class reads directly from DICOM files.
    """

    def __init__(self, df, input_dir=INPUT_DIR, transform=None):
        self.df = df
        self.input_dir = input_dir
        self.transform = transform
        self.channel_config = CHANNEL_CONFIG

    def __len__(self):
        return len(self.df)

    def _select_anchor(self, subject_dir):
        """
        Determines the anchor slice index based on the Sum of Intensity in the FLAIR modality.
        Restricted to the 15%-85% depth range.
        """
        flair_path = os.path.join(subject_dir, ROI_ANCHOR_MODALITY)
        files = get_sorted_files(flair_path)
        num_files = len(files)

        if num_files == 0:
            return 0

        # Define depth range
        start_idx = int(num_files * ROI_DEPTH_RANGE[0])
        end_idx = int(num_files * ROI_DEPTH_RANGE[1])

        # Ensure valid range
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_files

        max_intensity_sum = -1.0
        anchor_idx = start_idx  # Default to start of range

        # Iterate through the valid range to find the slice with max signal
        for i in range(start_idx, end_idx):
            file_path = os.path.join(flair_path, files[i])
            img = read_dicom_robust(file_path)

            # Calculate Sum of Intensity (Raw pixels)
            current_sum = np.sum(img)

            if current_sum > max_intensity_sum:
                max_intensity_sum = current_sum
                anchor_idx = i

        return anchor_idx

    def _load_stack(self, row, anchor_idx):
        """
        Constructs the 12-channel input tensor based on the anchor index and channel config.
        """
        channels = []
        subject_dir = os.path.join(self.input_dir, row["BraTS21ID_str"])

        # Iterate through the configuration groups
        for config in self.channel_config:
            modality = config["modality"]
            stride = config["stride"]

            # Determine relative offsets: [Anchor - Stride, Anchor, Anchor + Stride]
            offsets = [-stride, 0, stride]

            modality_dir = os.path.join(subject_dir, modality)
            modality_files = get_sorted_files(modality_dir)
            num_files = len(modality_files)

            for offset in offsets:
                target_idx = anchor_idx + offset

                # Edge Clamping: Ensure index is within bounds of THIS modality
                # Note: We use the FLAIR anchor index for all modalities (Single Reference)
                # If other modalities have different slice counts, we clamp to their limits.
                if num_files > 0:
                    target_idx = max(0, min(target_idx, num_files - 1))
                    file_path = os.path.join(modality_dir, modality_files[target_idx])

                    # Read and Process (Resize/Float32)
                    img_raw = read_dicom_robust(file_path)
                    img = process_image(img_raw)
                else:
                    # Missing modality fallback
                    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

                # Independent Per-Channel Min-Max Scaling
                # Normalize to [0, 1]
                img_min = img.min()
                img_max = img.max()
                denominator = img_max - img_min + 1e-8  # Epsilon for stability
                img = (img - img_min) / denominator

                channels.append(img)

        # Stack channels to create (12, H, W) tensor
        tensor = np.stack(channels, axis=0)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Select ROI Anchor
        # We construct the path manually to ensure we look in the right place
        subject_dir = os.path.join(
            self.input_dir,
            "train" if "train" in str(row.get("path_FLAIR", "")) else "test",
            row["BraTS21ID_str"],
        )

        # Fallback if path construction is ambiguous from metadata
        if not os.path.exists(subject_dir):
            # Try using the relative path from metadata if available
            if "path_FLAIR" in row:
                subject_dir = os.path.join(
                    self.input_dir, os.path.dirname(row["path_FLAIR"])
                )

        anchor_idx = self._select_anchor(subject_dir)

        # 2. Load Volume Stack
        tensor = self._load_stack(row, anchor_idx)

        # 3. Get Label (if available)
        label = -1.0
        if "MGMT_value" in row:
            label = float(row["MGMT_value"])

        # Note: Transforms are usually applied on the cached dataset, not here,
        # unless this is used directly.
        if self.transform:
            # Convert to torch tensor for transforms if needed, or assume numpy transforms
            tensor = self.transform(tensor)

        return tensor, label


# -----------------------------------------------------------------------------
# In-Memory Dataset (Wrapper for Cached Data)
# -----------------------------------------------------------------------------


class InMemoryDataset(Dataset):
    """
    A lightweight dataset that holds pre-loaded numpy arrays in RAM.
    Applies transforms on-the-fly during training.
    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve data from memory
        x = self.data[idx]  # Shape: (12, 224, 224)

        # Retrieve label
        if self.labels is not None:
            y = self.labels[idx]
        else:
            y = -1.0  # Dummy label for test set

        # Apply transforms
        # Expecting x to be a numpy array or torch tensor.
        # If transform expects tensor, we convert.
        x = torch.tensor(x, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# -----------------------------------------------------------------------------
# Caching & Loading Logic
# -----------------------------------------------------------------------------


def load_dataset(
    metadata_path,
    cache_path_data,
    cache_path_labels,
    load_cached_data=True,
    transform=None,
    debug_max_samples=None,
):
    """
    Loads the dataset.
    1. Checks if valid cache exists.
    2. If not, processes the raw DICOMs using BrainTumorDataset and saves to cache.
    3. Returns an InMemoryDataset containing the data.
    """

    # Load Metadata
    df = pd.read_csv(metadata_path)

    if debug_max_samples is not None:
        print(f"DEBUG: Limiting dataset to {debug_max_samples} samples.")
        df = df.head(debug_max_samples)

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path_data), exist_ok=True)

    # Check for cache
    cache_exists = os.path.exists(cache_path_data)
    if cache_path_labels:
        cache_exists = cache_exists and os.path.exists(cache_path_labels)

    # 1. Load from Cache
    if load_cached_data and cache_exists:
        print(f"Loading cached data from {cache_path_data}...")
        try:
            data_arr = np.load(cache_path_data)

            if cache_path_labels and os.path.exists(cache_path_labels):
                labels_arr = np.load(cache_path_labels)
            else:
                labels_arr = None

            # Verify consistency
            if len(data_arr) == len(df):
                print(f"Successfully loaded {len(data_arr)} samples from cache.")
                return InMemoryDataset(data_arr, labels_arr, transform)
            else:
                print(
                    f"Cache size mismatch (Cache: {len(data_arr)}, Meta: {len(df)}). Regenerating..."
                )
        except Exception as e:
            print(f"Error loading cache: {e}. Regenerating...")

    # 2. Generate from Scratch
    print("Processing raw DICOM data (this may take a while)...")

    # Initialize Raw Dataset (No transforms during caching)
    raw_ds = BrainTumorDataset(df, transform=None)

    data_list = []
    label_list = []

    # Iterate and collect
    # Using a simple loop to avoid dependency on tqdm if not strictly required,
    # but printing status every 10%
    total = len(raw_ds)
    for i in range(total):
        if i % max(1, total // 10) == 0:
            print(f"Processing sample {i}/{total}...")

        x, y = raw_ds[i]
        data_list.append(x)
        label_list.append(y)

    # Stack into arrays
    data_arr = np.stack(data_list).astype(np.float32)  # (N, 12, 224, 224)

    if "MGMT_value" in df.columns:
        labels_arr = np.array(label_list, dtype=np.float32)
    else:
        labels_arr = None

    # Save to Cache
    print(f"Saving cache to {cache_path_data}...")
    np.save(cache_path_data, data_arr)

    if cache_path_labels and labels_arr is not None:
        np.save(cache_path_labels, labels_arr)
    elif cache_path_labels:
        # Save dummy labels if requested but not available (e.g. test set consistency)
        dummy_labels = np.full(len(df), -1.0, dtype=np.float32)
        np.save(cache_path_labels, dummy_labels)
        labels_arr = dummy_labels

    print("Data processing complete.")
    return InMemoryDataset(data_arr, labels_arr, transform)
