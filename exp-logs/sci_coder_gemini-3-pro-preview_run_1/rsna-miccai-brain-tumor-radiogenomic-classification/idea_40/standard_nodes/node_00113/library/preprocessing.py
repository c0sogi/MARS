import os
import glob
import numpy as np
import pandas as pd
import cv2
from library.config import INPUT_DIR, IMG_SIZE, PLANES, CACHE_DIR, WORKING_DIR
from library.utils import (
    read_dicom_image,
    get_brain_bbox,
    get_center_of_mass_z,
    min_max_scale,
    save_numpy_cache,
    load_numpy_cache,
)


def load_dicom_volume(folder_rel_path):
    """
    Reads a directory of DICOM files into a 3D numpy array.
    Args:
        folder_rel_path: Relative path to the modality folder (e.g., 'train/00000/FLAIR')
    Returns:
        volume: 3D numpy array (Depth, Height, Width) or None if empty/error.
    """
    full_path = os.path.join(INPUT_DIR, folder_rel_path)
    if not os.path.exists(full_path):
        return None

    # Get all dicom files
    files = glob.glob(os.path.join(full_path, "*.dcm"))
    if not files:
        return None

    # Sort files to ensure correct depth order.
    # Usually Image-X.dcm, sorting by the integer X is safest.
    # If filename structure varies, standard string sort might fail (1, 10, 2),
    # but BraTS data usually follows Image-N pattern.
    try:
        files.sort(
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1])
        )
    except Exception:
        files.sort()  # Fallback to lexicographical sort

    images = []
    for f in files:
        img = read_dicom_image(f)
        if img is not None:
            images.append(img)

    if not images:
        return None

    return np.array(images)


def compute_geometry(volume):
    """
    Calculates the Z-axis Median Index and the XY Bounding Box.
    Cite Lesson 00036: Independent heuristics (median index) are more robust than CoM or ROI-based projection.
    """
    threshold = 0

    # 1. Calculate Z-axis Median Index (Anchor)
    # We use the geometric middle instead of Center of Mass to avoid intensity-based noise.
    z_center = volume.shape[0] // 2

    # 2. Calculate 3D Bounding Box to get XY limits (Centering/Cropping)
    # Cite Lesson 00065: Aligning the brain via bbox allows us to exclude Shift/Scale augmentations safely.
    _, _, y_min, y_max, x_min, x_max = get_brain_bbox(volume, threshold=threshold)

    return z_center, (y_min, y_max, x_min, x_max)


def extract_expert_slice(volume, z_com, bbox, plane_offset, target_size):
    """
    Extracts a specific slice relative to CoM, crops to ROI, resizes, and normalizes.
    Args:
        volume: 3D numpy array.
        z_com: Z-axis center of mass index.
        bbox: (y_min, y_max, x_min, x_max) tuple.
        plane_offset: Float relative offset (e.g., -0.15, 0.0, 0.15).
        target_size: Int, output spatial dimension (H=W).
    Returns:
        processed_slice: 2D numpy array (target_size, target_size), float32, [0, 1].
    """
    depth = volume.shape[0]

    # Calculate target slice index
    # Formula: Z_target = Z_com + (offset * total_depth)
    slice_idx = int(z_com + (plane_offset * depth))

    # Clip to valid range
    slice_idx = np.clip(slice_idx, 0, depth - 1)

    # Extract raw slice
    raw_slice = volume[slice_idx]

    # Crop to Bounding Box (Content-Adaptive Resolution Enhancement)
    y_min, y_max, x_min, x_max = bbox

    # Handle empty volume case
    if y_min >= y_max or x_min >= x_max:
        return np.zeros((target_size, target_size), dtype=np.float32)

    cropped_slice = raw_slice[y_min:y_max, x_min:x_max]

    # Resize to target size
    # cv2.resize expects (width, height)
    resized_slice = cv2.resize(
        cropped_slice, (target_size, target_size), interpolation=cv2.INTER_LINEAR
    )

    # Normalize to [0, 1]
    normalized_slice = min_max_scale(resized_slice)

    return normalized_slice


def process_subject(row, img_size=IMG_SIZE):
    """
    Process a single subject row from metadata.
    Returns a dictionary containing the 3 expert inputs (Lower, Center, Upper).
    Each input is (H, W, 3) corresponding to [FLAIR, T1wCE, T2w].
    """
    # Modalities to use for the 3 channels
    # Note: T1w is available but we select FLAIR, T1wCE, T2w as per strategy
    modalities = ["flair", "t1wce", "t2w"]

    # Container for extracted slices: {modality: {plane: slice}}
    extracted_data = {mod: {} for mod in modalities}

    # Define planes from config
    # PLANES = {"lower": -0.15, "center": 0.0, "upper": 0.15}

    for mod in modalities:
        path_col = f"{mod}_path"
        rel_path = row[path_col]

        # Load Volume
        volume = load_dicom_volume(rel_path)

        if volume is None:
            # Fallback for missing data: Black images
            for plane_name in PLANES.keys():
                extracted_data[mod][plane_name] = np.zeros(
                    (img_size, img_size), dtype=np.float32
                )
            continue

        # Compute Geometry (Independent Anchoring)
        z_com, bbox = compute_geometry(volume)

        # Extract Slices for each plane
        for plane_name, offset in PLANES.items():
            img_slice = extract_expert_slice(volume, z_com, bbox, offset, img_size)
            extracted_data[mod][plane_name] = img_slice

    # Stack channels for each Expert
    # Expert Input Shape: (H, W, 3) -> Channels: [FLAIR, T1wCE, T2w]

    expert_inputs = {}
    for plane_name in PLANES.keys():
        # Stack along last axis
        channels = [
            extracted_data["flair"][plane_name],
            extracted_data["t1wce"][plane_name],
            extracted_data["t2w"][plane_name],
        ]
        # Result shape: (224, 224, 3)
        expert_inputs[plane_name] = np.stack(channels, axis=-1)

    return expert_inputs


def process_dataset(metadata_df, load_cached_data=True, save_name="train"):
    """
    Main processing function.
    Args:
        metadata_df: DataFrame containing file paths and labels (if train/val).
        load_cached_data: Bool, whether to attempt loading from cache.
        save_name: String identifier for cache files (e.g., 'train', 'val', 'test').
    Returns:
        images: Numpy array of shape (N, 3, H, W, C).
                Dimension 1 (size 3) corresponds to [Lower, Center, Upper] experts.
        ids: Numpy array of BraTS21IDs.
        labels: Numpy array of targets (or None if test).
    """
    # Define cache filenames
    cache_images_name = f"cache_{save_name}_images.npy"
    cache_ids_name = f"cache_{save_name}_ids.npy"
    cache_labels_name = f"cache_{save_name}_targets.npy"

    # 1. Try Load from Cache
    if load_cached_data:
        print(f"Attempting to load {save_name} data from cache...")
        c_images = load_numpy_cache(cache_images_name, CACHE_DIR)
        c_ids = load_numpy_cache(cache_ids_name, CACHE_DIR)
        c_labels = load_numpy_cache(cache_labels_name, CACHE_DIR)

        # Check if all needed files exist
        # Labels might be None for test set, so we check logic accordingly
        if c_images is not None and c_ids is not None:
            if "MGMT_value" in metadata_df.columns:
                if c_labels is not None:
                    print("Cache hit! Data loaded successfully.")
                    return c_images, c_ids, c_labels
            else:
                print("Cache hit! Data loaded successfully (no labels).")
                return c_images, c_ids, None

        print("Cache miss or incomplete. Processing from scratch...")

    # 2. Process from Scratch
    print(f"Processing {len(metadata_df)} subjects for {save_name} set...")

    all_images = []
    all_ids = []
    all_labels = []

    # Iterate DataFrame
    for idx, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]

        # Process images
        # Returns dict: {'lower': (H,W,3), 'center': (H,W,3), 'upper': (H,W,3)}
        expert_dict = process_subject(row, IMG_SIZE)

        # Stack the experts into one array for this subject
        # Dynamically handle planes based on config to support single-plane optimization
        plane_keys = sorted(PLANES.keys())
        subject_stack = np.stack([expert_dict[k] for k in plane_keys], axis=0)

        all_images.append(subject_stack)
        all_ids.append(sid)

        if "MGMT_value" in row:
            all_labels.append(row["MGMT_value"])

    # Convert to numpy arrays
    # Final Shape: (N, 3, H, W, 3)
    arr_images = np.array(all_images, dtype=np.float32)
    arr_ids = np.array(all_ids, dtype=np.int64)

    if all_labels:
        arr_labels = np.array(all_labels, dtype=np.float32)
    else:
        arr_labels = None

    # 3. Save to Cache
    print(f"Saving processed {save_name} data to cache in {CACHE_DIR}...")
    save_numpy_cache(arr_images, cache_images_name, CACHE_DIR)
    save_numpy_cache(arr_ids, cache_ids_name, CACHE_DIR)
    if arr_labels is not None:
        save_numpy_cache(arr_labels, cache_labels_name, CACHE_DIR)

    return arr_images, arr_ids, arr_labels
