import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import (
    INPUT_DIR,
    WORKING_DIR,
    IMG_SIZE,
    RELATIVE_DEPTHS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
)


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm, Image-2.dcm, Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def min_max_normalize(img):
    """
    Scales the image to [0, 1].
    """
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def load_dicom_slice(path, resize_to=IMG_SIZE):
    """
    Reads a DICOM file and returns a float32 numpy array.
    Attempts pydicom first, then cv2. Resizes to the target dimension.
    """
    if not os.path.exists(path):
        return np.zeros((resize_to, resize_to), dtype=np.float32)

    img = None

    # Attempt 1: pydicom
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
    except Exception:
        pass

    # Attempt 2: cv2
    if img is None:
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except Exception:
            pass

    if img is None:
        return np.zeros((resize_to, resize_to), dtype=np.float32)

    # Resize
    if img.shape[0] != resize_to or img.shape[1] != resize_to:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    return img


def get_modality_roi(modality_path):
    """
    Scans all DICOM files in a modality folder to find the start and end indices
    of the brain tissue (pixels > 0).

    Returns:
        tuple: (start_index, end_index, sorted_file_names)
    """
    if not os.path.exists(modality_path):
        return 0, 0, []

    files = [f for f in os.listdir(modality_path) if f.endswith(".dcm")]
    files.sort(key=natural_sort_key)

    if not files:
        return 0, 0, []

    # If there are too many files, checking every single one might be slow,
    # but for ROI accuracy we should check. Given the constraints and hardware,
    # we proceed with a linear scan.

    has_tissue = []

    for f in files:
        f_path = os.path.join(modality_path, f)
        # We need to read the image to check for max > 0
        # Optimization: Use pydicom with stop_before_pixels=False to check if we can skip?
        # No, we need pixel data.
        img = load_dicom_slice(f_path, resize_to=IMG_SIZE)
        if np.max(img) > 0:
            has_tissue.append(True)
        else:
            has_tissue.append(False)

    # Find first and last True
    indices = [i for i, x in enumerate(has_tissue) if x]

    if not indices:
        # Fallback: use whole range if no tissue detected (unlikely but possible in bad data)
        return 0, len(files) - 1, files

    return indices[0], indices[-1], files


def select_sicav_indices(start_idx, end_idx, depths=RELATIVE_DEPTHS):
    """
    Calculates file indices corresponding to specific relative depths within the ROI.
    """
    length = end_idx - start_idx
    indices = []
    for d in depths:
        # Calculate relative position
        rel_pos = start_idx + (length * d)
        idx = int(round(rel_pos))
        # Clamp to valid range
        idx = max(start_idx, min(end_idx, idx))
        indices.append(idx)
    return indices


def process_subject(row, input_dir):
    """
    Generates the 9-channel input tensor for a single subject using SICAV strategy.

    Channels 0-2: [FLAIR, T1wCE, T2w] at 40% ROI depth
    Channels 3-5: [FLAIR, T1wCE, T2w] at 50% ROI depth
    Channels 6-8: [FLAIR, T1wCE, T2w] at 60% ROI depth
    """
    # Modalities of interest for the 9-channel tensor
    # Note: T1w is excluded based on the strategy description.
    modalities = ["flair", "t1wce", "t2w"]

    # Store images for each modality at each depth
    # Structure: depth_images[depth_index][modality_index]
    depth_images = {d_idx: [] for d_idx in range(len(RELATIVE_DEPTHS))}

    for mod in modalities:
        # Construct full path
        rel_path = row[f"{mod}_path"]
        full_path = os.path.join(input_dir, rel_path)

        # 1. Get ROI
        start, end, files = get_modality_roi(full_path)

        if not files:
            # Handle missing modality with zeros
            for d_idx in range(len(RELATIVE_DEPTHS)):
                depth_images[d_idx].append(
                    np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                )
            continue

        # 2. Select Indices
        indices = select_sicav_indices(start, end, RELATIVE_DEPTHS)

        # 3. Load and Normalize Images
        for d_idx, file_idx in enumerate(indices):
            file_name = files[file_idx]
            file_path = os.path.join(full_path, file_name)

            img = load_dicom_slice(file_path, resize_to=IMG_SIZE)
            img = min_max_normalize(img)
            depth_images[d_idx].append(img)

    # Stack into (224, 224, 9)
    # Order:
    # D0: FLAIR, T1wCE, T2w
    # D1: FLAIR, T1wCE, T2w
    # D2: FLAIR, T1wCE, T2w

    channels = []
    for d_idx in range(len(RELATIVE_DEPTHS)):
        channels.extend(depth_images[d_idx])

    # Stack along the last axis (channels)
    tensor = np.stack(channels, axis=-1)
    return tensor


def load_data(split_name, load_cached_data=True):
    """
    Main function to load and process the dataset.

    Args:
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (ids, images, labels)
            ids: numpy array of BraTS21IDs
            images: numpy array of shape (N, IMG_SIZE, IMG_SIZE, 9)
            labels: numpy array of targets (or zeros for test)
    """
    # Define cache paths
    cache_ids_path = os.path.join(WORKING_DIR, f"cached_{split_name}_ids.npy")
    cache_images_path = os.path.join(WORKING_DIR, f"cached_{split_name}_images.npy")
    cache_labels_path = os.path.join(WORKING_DIR, f"cached_{split_name}_labels.npy")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_ids_path)
            and os.path.exists(cache_images_path)
            and os.path.exists(cache_labels_path)
        ):

            print(f"Loading {split_name} data from cache...")
            ids = np.load(cache_ids_path)
            images = np.load(cache_images_path)
            labels = np.load(cache_labels_path)
            return ids, images, labels

    # 2. Process from scratch
    print(f"Processing {split_name} data from scratch (SICAV Strategy)...")

    # Select metadata file
    if split_name == "train":
        meta_path = TRAIN_METADATA_PATH
    elif split_name == "val":
        meta_path = VAL_METADATA_PATH
    elif split_name == "test":
        meta_path = TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown split: {split_name}")

    df = pd.read_csv(meta_path)

    ids_list = []
    images_list = []
    labels_list = []

    total = len(df)
    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]

        # Print progress every 50 subjects to track runtime
        if idx % 50 == 0:
            print(f"Processing subject {idx}/{total} (ID: {sid})")

        try:
            tensor = process_subject(row, INPUT_DIR)

            ids_list.append(sid)
            images_list.append(tensor)

            if "MGMT_value" in row:
                labels_list.append(row["MGMT_value"])
            else:
                labels_list.append(0.0)  # Placeholder for test

        except Exception as e:
            print(f"Error processing subject {sid}: {e}")
            # In case of error, we skip or add placeholder.
            # To maintain array integrity, we'll add a zero tensor.
            ids_list.append(sid)
            images_list.append(np.zeros((IMG_SIZE, IMG_SIZE, 9), dtype=np.float32))
            labels_list.append(0.0)

    # Convert to numpy arrays
    ids = np.array(ids_list)
    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    # 3. Save to cache
    print(f"Saving {split_name} data to cache at {WORKING_DIR}...")
    np.save(cache_ids_path, ids)
    np.save(cache_images_path, images)
    np.save(cache_labels_path, labels)

    return ids, images, labels
