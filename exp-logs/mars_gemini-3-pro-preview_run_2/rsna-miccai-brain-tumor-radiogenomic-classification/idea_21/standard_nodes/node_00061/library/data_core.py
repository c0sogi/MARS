import os
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.data_io import read_dicom_robust


def get_sorted_image_files(directory):
    """
    Lists and sorts DICOM files in a directory numerically based on the
    'Image-{num}.dcm' naming convention.

    Args:
        directory (str): Path to the directory containing DICOM files.

    Returns:
        list: Sorted list of filenames.
    """
    if not os.path.exists(directory):
        return []

    files = [f for f in os.listdir(directory) if f.endswith(".dcm")]

    # Sort files numerically by extracting the number after 'Image-'
    # Format is usually Image-1.dcm, Image-10.dcm, etc.
    def extract_number(filename):
        try:
            # Remove extension and split by '-'
            name_part = os.path.splitext(filename)[0]
            num_part = name_part.split("-")[-1]
            return int(num_part)
        except ValueError:
            return 0

    return sorted(files, key=extract_number)


def select_anchor_slice(flair_dir_path):
    """
    Identifies the anchor slice index based on the 'Sum of Intensity' metric
    calculated on raw pixel values within the 15%-85% depth range.

    Args:
        flair_dir_path (str): Path to the FLAIR modality directory.

    Returns:
        int: The index of the selected anchor slice (0-based relative to sorted file list).
    """
    files = get_sorted_image_files(flair_dir_path)
    num_files = len(files)

    if num_files == 0:
        return 0

    # Define depth boundaries (15% - 85%)
    start_idx = int(num_files * Config.ROI_DEPTH_MIN)
    end_idx = int(num_files * Config.ROI_DEPTH_MAX)

    # Ensure valid range
    if start_idx >= end_idx:
        start_idx = 0
        end_idx = num_files

    max_intensity_sum = -1.0
    best_index = start_idx  # Default to start if something goes wrong

    # Iterate through the defined range
    for i in range(start_idx, end_idx):
        file_path = os.path.join(flair_dir_path, files[i])

        # Read raw image (uint16)
        # We use raw values to avoid normalization artifacts affecting selection
        img = read_dicom_robust(file_path)

        # Calculate Sum of Intensity
        current_sum = np.sum(img)

        if current_sum > max_intensity_sum:
            max_intensity_sum = current_sum
            best_index = i

    return best_index


def get_roi_anchor_indices(df, load_cached_data=True):
    """
    Retrieves anchor indices for all subjects in the DataFrame.
    Implements caching using Parquet to store expensive ROI calculations.

    Args:
        df (pd.DataFrame): DataFrame containing 'BraTS21ID' and 'path_FLAIR'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Mapping from BraTS21ID (int) to anchor_index (int).
    """
    cache_file = Config.ROI_CACHE_FILE

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            print(f"Loading ROI anchors from cache: {cache_file}")
            cache_df = pd.read_parquet(cache_file)
            # Convert to dictionary
            return dict(zip(cache_df["BraTS21ID"], cache_df["anchor_index"]))
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing ROI anchors from scratch...")
    anchor_map = {}

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    total = len(df)
    for idx, row in df.iterrows():
        subject_id = row["BraTS21ID"]

        # Construct full path to FLAIR directory
        # Metadata contains relative paths, need to join with INPUT_DIR
        flair_path = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

        anchor_idx = select_anchor_slice(flair_path)
        anchor_map[subject_id] = anchor_idx

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{total} subjects for ROI selection.")

    # 3. Save to cache
    try:
        cache_df = pd.DataFrame(
            {
                "BraTS21ID": list(anchor_map.keys()),
                "anchor_index": list(anchor_map.values()),
            }
        )
        cache_df.to_parquet(cache_file, index=False)
        print(f"Saved ROI anchors to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save ROI cache: {e}")

    return anchor_map


def normalize_independent(tensor):
    """
    Applies Min-Max scaling [0, 1] independently to each channel of the input tensor.

    Args:
        tensor (np.ndarray): Input tensor of shape (Channels, Height, Width).
                             Data type should be float32.

    Returns:
        np.ndarray: Normalized tensor of shape (Channels, Height, Width).
    """
    # Ensure float32
    tensor = tensor.astype(np.float32)

    # Iterate over channels (axis 0)
    for c in range(tensor.shape[0]):
        channel_data = tensor[c]
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)

        if max_val > min_val:
            # Apply Min-Max scaling
            tensor[c] = (channel_data - min_val) / (max_val - min_val)
        else:
            # If min == max (e.g., blank image), set to 0
            tensor[c] = np.zeros_like(channel_data)

    return tensor
