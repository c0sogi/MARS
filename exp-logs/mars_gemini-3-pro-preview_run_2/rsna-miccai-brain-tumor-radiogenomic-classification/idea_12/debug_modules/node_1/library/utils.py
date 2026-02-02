import os
import numpy as np
import cv2
import random
import torch
import pandas as pd


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_robust(path, size=(224, 224)):
    """
    Reads a DICOM file using a robust binary tail-read strategy to bypass header issues.
    Converts to float32 and resizes to the target size using Area Interpolation.

    Args:
        path (str): Full path to the .dcm file.
        size (tuple): Target resolution (width, height). Default is (224, 224).

    Returns:
        np.ndarray: The processed image as a float32 array of shape (H, W).
                    Returns a zero array if reading fails.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((size[1], size[0]), dtype=np.float32)

        with open(path, "rb") as f:
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            # Heuristic based on file size for common MRI resolutions (16-bit depth)
            # 512x512 * 2 bytes = 524,288 bytes
            # 256x256 * 2 bytes = 131,072 bytes

            if file_size >= 524288:
                rows, cols = 512, 512
                pixel_bytes = 524288
            elif file_size >= 131072:
                rows, cols = 256, 256
                pixel_bytes = 131072
            else:
                # File too small to contain expected image data
                return np.zeros((size[1], size[0]), dtype=np.float32)

            # Anchor read to the end of the file
            offset = file_size - pixel_bytes
            f.seek(offset)
            data = f.read(pixel_bytes)

            # Convert binary data to numpy array
            img = np.frombuffer(data, dtype=np.int16).reshape((rows, cols))

            # Convert to float32
            img = img.astype(np.float32)

            # Resize using Area Interpolation (good for downsampling/anti-aliasing)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            return img

    except Exception:
        # Fallback for any IO or processing errors
        return np.zeros((size[1], size[0]), dtype=np.float32)


def calculate_roi_index(flair_path):
    """
    Calculates the optimal ROI anchor slice index for a given FLAIR directory.
    Uses mean intensity profile, smoothing, and depth restrictions.

    Args:
        flair_path (str): Path to the directory containing FLAIR DICOM files.

    Returns:
        int: The index of the selected slice (0-based relative to sorted filenames).
    """
    try:
        # List and sort files numerically (Image-1.dcm, Image-2.dcm, etc.)
        files = [f for f in os.listdir(flair_path) if f.endswith(".dcm")]
        if not files:
            return 0

        # Sort by the integer number in the filename "Image-X.dcm"
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))

        num_slices = len(files)
        if num_slices == 0:
            return 0

        means = []
        # Calculate mean intensity for each slice
        for f in files:
            full_path = os.path.join(flair_path, f)
            img = read_dicom_robust(full_path)
            means.append(np.mean(img))

        means = np.array(means)

        # Apply Moving Average Filter (window=5)
        window_size = 5
        if num_slices >= window_size:
            kernel = np.ones(window_size) / window_size
            # mode='same' returns output of length max(M, N), boundary effects handled by zero-padding implicitly
            smoothed = np.convolve(means, kernel, mode="same")
        else:
            smoothed = means

        # Restrict search to 15% - 85% depth range
        start_idx = int(num_slices * 0.15)
        end_idx = int(num_slices * 0.85)

        # Handle edge cases where range is invalid
        if start_idx >= end_idx:
            start_idx = 0
            end_idx = num_slices

        # Find argmax within the restricted bounds
        # Note: argmax returns index relative to the slice, so we add start_idx
        subset = smoothed[start_idx:end_idx]
        if len(subset) == 0:
            return num_slices // 2

        best_local_idx = np.argmax(subset)
        best_global_idx = start_idx + best_local_idx

        return int(best_global_idx)

    except Exception:
        # Fallback to middle slice on error
        return 0


def generate_roi_cache(
    metadata_df,
    load_cached_data=True,
    cache_dir="./working/idea_12",
    input_root="./input",
):
    """
    Generates or loads a cache of ROI indices for all subjects in the metadata.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'BraTS21ID' and 'path_FLAIR'.
        load_cached_data (bool): If True, attempts to load from disk first.
        cache_dir (str): Directory to store the cache file.
        input_root (str): Root directory for input data.

    Returns:
        dict: A dictionary mapping BraTS21ID (int) to ROI index (int).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "roi_cache.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading ROI cache from {cache_path}")
        try:
            cache_df = pd.read_parquet(cache_path)
            # Convert to dictionary
            return dict(zip(cache_df["BraTS21ID"], cache_df["roi_index"]))
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print("Computing ROI indices for all subjects...")
    results = []

    # Iterate over unique subjects
    # We use a set to avoid processing duplicates if train/val split is passed combined
    unique_subjects = metadata_df[["BraTS21ID", "path_FLAIR"]].drop_duplicates()

    for _, row in unique_subjects.iterrows():
        subject_id = row["BraTS21ID"]
        rel_path = row["path_FLAIR"]
        full_path = os.path.join(input_root, rel_path)

        roi_idx = calculate_roi_index(full_path)
        results.append({"BraTS21ID": subject_id, "roi_index": roi_idx})

    # 3. Save to cache
    result_df = pd.DataFrame(results)
    result_df.to_parquet(cache_path, index=False)
    print(f"ROI cache saved to {cache_path}")

    return dict(zip(result_df["BraTS21ID"], result_df["roi_index"]))
