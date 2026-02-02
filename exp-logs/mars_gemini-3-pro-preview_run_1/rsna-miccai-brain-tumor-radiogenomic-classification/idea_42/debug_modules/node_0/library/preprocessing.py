import os
import glob
import re
import numpy as np
import pandas as pd
import cv2
import torch
from library import config, utils

# Attempt to import pydicom, handle if missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def extract_file_index(filepath):
    """
    Extracts the integer index from a DICOM filename (e.g., Image-123.dcm).
    Used for sorting slices spatially along the Z-axis.
    """
    filename = os.path.basename(filepath)
    match = re.search(r"Image-(\d+)\.dcm", filename)
    if match:
        return int(match.group(1))
    return 0


def load_dicom_file(path):
    """
    Reads a single DICOM file into a numpy array.
    Prioritizes pydicom for accuracy, falls back to OpenCV if necessary.
    """
    # Method 1: pydicom (Preferred for raw pixel data and header handling)
    if HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(path)
            return ds.pixel_array
        except Exception:
            pass

    # Method 2: OpenCV (Fallback for specific formats)
    try:
        # cv2.IMREAD_UNCHANGED is crucial to preserve 16-bit depth
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img
    except Exception:
        pass

    return None


def load_dicom_volume(folder_path):
    """
    Loads all DICOM files from a folder and stacks them into a 3D volume.
    Sorts files by numerical index in filename to ensure Z-axis consistency.
    Returns: 3D numpy array (Depth, Height, Width) or None if empty.
    """
    if not os.path.exists(folder_path):
        return None

    # Get all .dcm files
    files = glob.glob(os.path.join(folder_path, "*.dcm"))
    if not files:
        return None

    # Sort by index (Image-1, Image-2, ... Image-10) to maintain spatial order
    files.sort(key=extract_file_index)

    slices = []
    for f in files:
        img = load_dicom_file(f)
        if img is not None:
            slices.append(img)

    if not slices:
        return None

    # Stack into volume
    try:
        volume = np.stack(slices, axis=0)
        return volume
    except ValueError:
        # Handle rare case where slices might have inconsistent shapes
        # Return the median slice duplicated to form a minimal volume
        return np.array(slices)


def get_z_center_of_mass(volume):
    """
    Calculates the Center of Mass (CoM) along the Z-axis based on pixel intensity.
    This identifies the slice containing the most significant brain tissue structure,
    avoiding empty apical or basal slices.
    Returns the index of the slice closest to the CoM.
    """
    if volume is None or volume.shape[0] == 0:
        return 0

    # Calculate total intensity per slice (mass proxy)
    # volume shape: (Depth, Height, Width)
    slice_masses = np.sum(volume, axis=(1, 2))

    total_mass = np.sum(slice_masses)

    if total_mass == 0:
        # If volume is completely black, return middle slice
        return volume.shape[0] // 2

    # Weighted sum of indices
    z_indices = np.arange(volume.shape[0])
    center_z = np.sum(z_indices * slice_masses) / total_mass

    return int(np.round(center_z))


def crop_roi_and_resize(image, target_size=(224, 224)):
    """
    Crops the image to the bounding box of non-zero pixels (brain tissue)
    and resizes it to the target size. This maximizes resolution of the ROI.
    """
    if image is None:
        return np.zeros(target_size, dtype=np.float32)

    # Find non-zero pixels
    rows = np.any(image, axis=1)
    cols = np.any(image, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, image.shape[0])
    xmin, xmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, image.shape[1])

    # Check if image is effectively empty
    if ymax <= ymin or xmax <= xmin:
        return np.zeros(target_size, dtype=np.float32)

    # Crop to Bounding Box
    cropped = image[ymin : ymax + 1, xmin : xmax + 1]

    # Resize
    # Use INTER_AREA for downsampling (preserving intensity sum)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    return resized


def normalize_image(image):
    """
    Min-Max normalization to [0, 1].
    Preserves relative intensity differences within the slice.
    """
    if image is None:
        return image

    img_min = image.min()
    img_max = image.max()

    if img_max - img_min == 0:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - img_min) / (img_max - img_min)
    return image.astype(np.float32)


def preprocess_subject(row, input_dir, img_size):
    """
    Process a single subject:
    1. Load volumes for FLAIR, T1wCE, T2w independently.
    2. Find CoM slice for each modality.
    3. Crop ROI and resize.
    4. Normalize.
    5. Stack into (H, W, 3).
    """
    channels = []

    # Map config modalities to metadata column names
    # config.SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]
    modality_map = {"FLAIR": "flair_path", "T1wCE": "t1wce_path", "T2w": "t2w_path"}

    for mod_name in config.SELECTED_MODALITIES:
        col_name = modality_map.get(mod_name)
        rel_path = row[col_name]
        full_path = os.path.join(input_dir, rel_path)

        # 1. Load Volume
        volume = load_dicom_volume(full_path)

        if volume is not None:
            # 2. Get CoM Slice
            z_idx = get_z_center_of_mass(volume)
            # Clamp index within bounds
            z_idx = max(0, min(z_idx, volume.shape[0] - 1))
            img_slice = volume[z_idx]

            # 3. Crop & Resize
            processed_img = crop_roi_and_resize(img_slice, (img_size, img_size))

            # 4. Normalize
            processed_img = normalize_image(processed_img)
        else:
            # Fallback for missing modality: Black image
            processed_img = np.zeros((img_size, img_size), dtype=np.float32)

        channels.append(processed_img)

    # 5. Stack Channels
    # Result shape: (H, W, C) -> (224, 224, 3)
    stacked_img = np.stack(channels, axis=-1)
    return stacked_img


def process_dataframe(df, description="Processing"):
    """
    Iterates through a dataframe and processes all subjects.
    """
    images = []
    ids = []
    labels = []

    print(f"Starting {description} for {len(df)} subjects...")

    for idx, row in df.iterrows():
        sid = row["BraTS21ID"]

        # Process image
        img = preprocess_subject(row, config.INPUT_DIR, config.IMG_SIZE)

        images.append(img)
        ids.append(sid)

        if "MGMT_value" in row:
            labels.append(row["MGMT_value"])

    images = np.array(images, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    if labels:
        labels = np.array(labels, dtype=np.float32)
        return images, ids, labels
    else:
        return images, ids, None


def prepare_datasets(load_cached_data=True):
    """
    Main entry point. Loads metadata, checks cache, processes data if needed,
    and returns numpy arrays for Train, Val, and Test sets.
    """
    logger = utils.get_logger("Preprocessing")

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Define Cache Paths
    path_train_imgs = os.path.join(config.WORKING_DIR, config.CACHE_TRAIN_IMAGES)
    path_train_lbls = os.path.join(config.WORKING_DIR, config.CACHE_TRAIN_LABELS)
    path_val_imgs = os.path.join(config.WORKING_DIR, config.CACHE_VAL_IMAGES)
    path_val_lbls = os.path.join(config.WORKING_DIR, config.CACHE_VAL_LABELS)
    path_test_imgs = os.path.join(config.WORKING_DIR, config.CACHE_TEST_IMAGES)
    path_test_ids = os.path.join(config.WORKING_DIR, config.CACHE_TEST_IDS)

    # Check if cache exists
    cache_exists = (
        os.path.exists(path_train_imgs)
        and os.path.exists(path_train_lbls)
        and os.path.exists(path_val_imgs)
        and os.path.exists(path_val_lbls)
        and os.path.exists(path_test_imgs)
        and os.path.exists(path_test_ids)
    )

    if load_cached_data and cache_exists:
        logger.info("Loading datasets from cache...")
        train_images = np.load(path_train_imgs)
        train_labels = np.load(path_train_lbls)
        val_images = np.load(path_val_imgs)
        val_labels = np.load(path_val_lbls)
        test_images = np.load(path_test_imgs)
        test_ids = np.load(path_test_ids)

        logger.info(
            f"Loaded Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}"
        )
        return (
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_ids),
        )

    # If not cached, process from scratch
    logger.info("Cache not found or ignored. Processing datasets from scratch...")

    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Process Train
    logger.info("Processing Training Set...")
    train_images, _, train_labels = process_dataframe(df_train, "Train")

    # Process Val
    logger.info("Processing Validation Set...")
    val_images, _, val_labels = process_dataframe(df_val, "Val")

    # Process Test
    logger.info("Processing Test Set...")
    test_images, test_ids, _ = process_dataframe(df_test, "Test")

    # Save to Cache
    logger.info(f"Saving processed data to {config.WORKING_DIR}...")
    np.save(path_train_imgs, train_images)
    np.save(path_train_lbls, train_labels)
    np.save(path_val_imgs, val_images)
    np.save(path_val_lbls, val_labels)
    np.save(path_test_imgs, test_images)
    np.save(path_test_ids, test_ids)

    logger.info("Data processing complete.")
    logger.info(
        f"Train: {train_images.shape}, Val: {val_images.shape}, Test: {test_images.shape}"
    )

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_ids),
    )
