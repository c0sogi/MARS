import os
import numpy as np
import pandas as pd
import cv2
import torch
from library import config

# Attempt to import pydicom, handling the case where it might be missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def read_dicom_robust(path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)):
    """
    Reads a DICOM file and returns a normalized numpy array.
    Implements a fallback mechanism if standard parsing fails.
    """
    image = None

    # Attempt 1: Pydicom
    if HAS_PYDICOM:
        try:
            dcm = pydicom.dcmread(path)
            image = dcm.pixel_array
        except Exception:
            pass

    # Attempt 2: OpenCV (Fallback)
    if image is None:
        try:
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    # Fallback: Return zeros if reading fails
    if image is None:
        return np.zeros(target_size, dtype=np.float32)

    try:
        # Ensure float32
        image = image.astype(np.float32)

        # Handle empty images
        if image.size == 0:
            return np.zeros(target_size, dtype=np.float32)

        # Min-Max Normalization to [0, 1]
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)

        # Resize to target dimensions
        if image.shape != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        return image
    except Exception:
        return np.zeros(target_size, dtype=np.float32)


def get_sorted_files(dir_path):
    """Returns sorted list of DICOM files in a directory based on numeric index."""
    if not os.path.exists(dir_path):
        return []

    files = [f for f in os.listdir(dir_path) if f.endswith(".dcm")]

    def extract_num(s):
        try:
            # Assumes format "Image-{num}.dcm"
            return int(s.split("-")[1].split(".")[0])
        except:
            return 0

    files.sort(key=extract_num)
    return [os.path.join(dir_path, f) for f in files]


def compute_consensus_anchor(flair_path):
    """
    Computes the anchor slice index using Consensus-Based Anchoring.
    Selects slice based on max(Mean_Intensity + Pixel_Variance).
    """
    files = get_sorted_files(flair_path)
    num_files = len(files)

    if num_files == 0:
        return 0

    # Boundary Exclusion: Exclude top/bottom 15%
    exclude_count = int(num_files * config.EXCLUDE_BOUNDARY_RATIO)
    start_idx = exclude_count
    end_idx = num_files - exclude_count

    # Fallback for very small volumes where exclusion removes everything
    if start_idx >= end_idx:
        return num_files // 2

    valid_files = files[start_idx:end_idx]

    means = []
    variances = []

    # Iterate through valid slices to build profiles
    for fpath in valid_files:
        try:
            img = read_dicom_robust(
                fpath, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
            )
            means.append(np.mean(img))
            variances.append(np.var(img))
        except:
            means.append(0)
            variances.append(0)

    means = np.array(means)
    variances = np.array(variances)

    # Helper to normalize profile to [0, 1]
    def normalize_profile(arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return np.zeros_like(arr)

    norm_means = normalize_profile(means)
    norm_vars = normalize_profile(variances)

    # Consensus Score
    scores = norm_means + norm_vars
    best_local_idx = np.argmax(scores)

    # Return absolute index
    return int(start_idx + best_local_idx)


def get_roi_anchors(df, load_cached_data=True):
    """
    Retrieves or computes ROI anchors for all subjects in the dataframe.
    Implements caching via Parquet.
    """
    os.makedirs(os.path.dirname(config.ROI_CACHE_PATH), exist_ok=True)

    anchors = {}

    # 1. Try to load cache
    if load_cached_data and os.path.exists(config.ROI_CACHE_PATH):
        try:
            cache_df = pd.read_parquet(config.ROI_CACHE_PATH)
            # Convert to dictionary for fast lookup
            anchors = pd.Series(
                cache_df.anchor.values, index=cache_df.BraTS21ID
            ).to_dict()
        except Exception:
            # If cache is corrupt, reset
            anchors = {}

    # 2. Compute for missing subjects
    # Check which IDs from input df are missing in the cache
    missing_ids = [uid for uid in df["BraTS21ID"] if uid not in anchors]

    if missing_ids:
        print(f"Computing ROI anchors for {len(missing_ids)} subjects...")

        # Filter df for missing subjects
        df_missing = df[df["BraTS21ID"].isin(missing_ids)]

        for idx, row in df_missing.iterrows():
            subject_id = row["BraTS21ID"]
            flair_path = os.path.join(config.INPUT_DIR, row["path_FLAIR"])

            # Compute anchor
            anchor = compute_consensus_anchor(flair_path)
            anchors[subject_id] = anchor

        # 3. Save updated cache
        save_df = pd.DataFrame(
            [{"BraTS21ID": k, "anchor": v} for k, v in anchors.items()]
        )
        save_df.to_parquet(config.ROI_CACHE_PATH)

    return anchors


def load_strided_volume(row, anchor_idx):
    """
    Loads the 12-channel tensor for a subject.
    Structure: [FLAIR(z-5, z, z+5), T1w(...), T1wCE(...), T2w(...)]
    """
    channels = []

    # Offsets for strided stacking: Anchor-5, Anchor, Anchor+5
    offsets = [-config.STRIDE, 0, config.STRIDE]

    # Iterate through modalities in fixed order
    for mod in config.MODALITIES:  # ["FLAIR", "T1w", "T1wCE", "T2w"]
        path_col = f"path_{mod}"
        dir_path = os.path.join(config.INPUT_DIR, row[path_col])
        files = get_sorted_files(dir_path)
        num_files = len(files)

        for off in offsets:
            target_idx = anchor_idx + off

            if num_files > 0:
                # Clamp index to valid range
                read_idx = max(0, min(num_files - 1, target_idx))
                fpath = files[read_idx]
                img = read_dicom_robust(
                    fpath, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)
                )
            else:
                # Handle missing modality
                img = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)

            channels.append(img)

    # Stack channels to create (12, H, W) volume
    volume = np.stack(channels, axis=0)

    # Return as torch tensor
    return torch.tensor(volume, dtype=torch.float32)
