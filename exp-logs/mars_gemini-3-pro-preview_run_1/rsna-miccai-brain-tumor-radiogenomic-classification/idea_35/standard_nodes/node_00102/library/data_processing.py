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


def process_subject(row, input_dir):
    """
    Generates the 3-channel input tensor for a single subject using Middle Slice strategy.
    Channels: [FLAIR, T1wCE, T2w]
    Cite solution_lesson_node_00002: Middle slice is a strong baseline.
    Cite solution_lesson_node_00036: Independent slice selection is robust.
    """
    modalities = ["flair", "t1wce", "t2w"]
    channels = []

    for mod in modalities:
        # Construct full path
        rel_path = row[f"{mod}_path"]
        full_path = os.path.join(input_dir, rel_path)

        # Check existence and list files
        if os.path.exists(full_path):
            files = [f for f in os.listdir(full_path) if f.endswith(".dcm")]
            files.sort(key=natural_sort_key)
        else:
            files = []

        if not files:
            # Handle missing modality with zeros
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            # Select Middle Slice
            idx = len(files) // 2
            file_name = files[idx]
            file_path = os.path.join(full_path, file_name)

            img = load_dicom_slice(file_path, resize_to=IMG_SIZE)
            img = min_max_normalize(img)

        channels.append(img)

    # Stack along the last axis (channels) -> (224, 224, 3)
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
    print(f"Processing {split_name} data from scratch (Middle Slice Strategy)...")

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
