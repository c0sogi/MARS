import os
import glob
import numpy as np
import pandas as pd
import cv2
import pydicom
from library import config, utils


def load_dicom_volume(path):
    """
    Reads a directory of DICOM files, sorts them by InstanceNumber,
    and returns the 3D volume as a numpy array.
    """
    if not os.path.exists(path):
        return None

    files = glob.glob(os.path.join(path, "*.dcm"))
    if not files:
        return None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            # Use InstanceNumber for Z-axis ordering; fallback to -1 if missing
            instance_num = (
                int(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else -1
            )
            pixel_array = ds.pixel_array
            slices.append((instance_num, pixel_array))
        except Exception:
            continue

    if not slices:
        return None

    # Sort by InstanceNumber to reconstruct the volume correctly
    slices.sort(key=lambda x: x[0])

    # Stack images into a 3D volume (Depth, Height, Width)
    volume = np.stack([s[1] for s in slices])
    return volume


def get_brain_centroid(volume):
    """
    Calculates the Z-axis center of mass of the brain tissue.
    Assumes background pixels are 0.
    """
    if volume is None or volume.size == 0:
        return 0

    # Create a mask for brain tissue (pixels > 0)
    # volume shape: (Depth, Height, Width)
    mask = volume > 0

    # Sum pixels per slice to get tissue amount per z-index
    slice_sums = np.sum(mask, axis=(1, 2))
    total_tissue = np.sum(slice_sums)

    if total_tissue == 0:
        # Fallback to geometric center if volume is empty/black
        return volume.shape[0] // 2

    # Calculate center of mass: sum(z * mass) / sum(mass)
    z_indices = np.arange(len(slice_sums))
    centroid = np.sum(z_indices * slice_sums) / total_tissue

    return int(round(centroid))


def extract_middle_slice(volume, centroid):
    """
    Extracts 3 slices around the anatomical centroid.
    Cite solution_lesson_node_00015: Deterministic geometric heuristics outperform stochastic sampling.
    """
    if volume is None or volume.shape[0] == 0:
        return None

    depth = volume.shape[0]
    # Cite debug_lesson_5: Bind Cache Identity to Data Generation Hyperparameters
    indices = [centroid - config.STRIDE, centroid, centroid + config.STRIDE]
    slices = []
    for idx in indices:
        idx = max(0, min(idx, depth - 1))
        slices.append(volume[idx])
    return slices


def normalize_slice(img):
    """
    Performs Min-Max scaling to [0, 1] for a single slice.
    """
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val > 0:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)

    return img


def resize_slice(img, size):
    """
    Resizes image to (size, size) using area interpolation.
    """
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def process_subject(row):
    """
    Process a single subject:
    1. Iterate through modalities (FLAIR, T1wCE, T2w).
    2. Load volume, find anatomical centroid, extract 1 middle slice.
    3. Resize and Normalize each slice.
    4. Stack into a 3-channel volume.
    """
    channels = []

    # Order defined in config: ["FLAIR", "T1wCE", "T2w"]
    for mod in config.MODALITIES:
        # Construct full path from metadata relative path
        rel_path = row[f"{mod.lower()}_path"]
        full_path = os.path.join(config.INPUT_DIR, rel_path)

        volume = load_dicom_volume(full_path)

        if volume is not None:
            centroid = get_brain_centroid(volume)
            # Cite solution_lesson_node_00015: Using deterministic middle slice
            slices = extract_middle_slice(volume, centroid)
        else:
            # Handle missing volume: create 3 empty slices to match slab depth
            slices = [
                np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
                for _ in range(3)
            ]

        # Process extracted slices
        for s in slices:
            s_resized = resize_slice(s, config.IMG_SIZE)
            s_norm = normalize_slice(s_resized)
            channels.append(s_norm)

    # Stack channels along the last axis -> (H, W, 3)
    # Channel order: [FLAIR, T1wCE, T2w]
    stack = np.stack(channels, axis=-1)

    return stack


def process_dataset(metadata_df, dataset_name, load_cached_data=True):
    """
    Main function to process a dataset (train/val/test).
    Handles caching using .npy files to avoid re-processing.
    """
    logger = utils.get_logger(f"data_processing_{dataset_name}")

    # Define cache file paths
    cache_ids_path = os.path.join(config.CACHE_DIR, f"{dataset_name}_ids.npy")
    cache_imgs_path = os.path.join(config.CACHE_DIR, f"{dataset_name}_images.npy")
    cache_lbls_path = os.path.join(config.CACHE_DIR, f"{dataset_name}_labels.npy")

    has_labels = "MGMT_value" in metadata_df.columns

    # 1. Try to load from cache
    if load_cached_data:
        # Check if ID and Image cache exists
        if os.path.exists(cache_ids_path) and os.path.exists(cache_imgs_path):
            # If labels are expected, check for them too
            if has_labels and not os.path.exists(cache_lbls_path):
                pass  # Cache incomplete, proceed to processing
            else:
                logger.info(
                    f"Loading cached {dataset_name} data from {config.CACHE_DIR}"
                )
                ids = np.load(cache_ids_path)
                images = np.load(cache_imgs_path)

                if has_labels:
                    labels = np.load(cache_lbls_path)
                    return ids, images, labels
                else:
                    return ids, images

    # 2. Process from scratch
    logger.info(f"Processing {dataset_name} data from scratch...")

    ids_list = []
    images_list = []
    labels_list = []

    # Handle Debug Mode
    if config.DEBUG:
        metadata_df = metadata_df.head(config.DEBUG_DATASET_SIZE)
        logger.info(f"DEBUG MODE: Processing only {len(metadata_df)} samples.")

    count = 0
    for idx, row in metadata_df.iterrows():
        sid = row["BraTS21ID"]

        try:
            img_stack = process_subject(row)

            ids_list.append(sid)
            images_list.append(img_stack)

            if has_labels:
                labels_list.append(row["MGMT_value"])

            count += 1
            if count % 50 == 0:
                logger.info(f"Processed {count}/{len(metadata_df)} subjects")

        except Exception as e:
            logger.error(f"Error processing subject {sid}: {e}")
            continue

    # Convert lists to numpy arrays
    ids_np = np.array(ids_list)
    images_np = np.array(images_list, dtype=np.float32)  # Shape: (N, 224, 224, 3)

    # Transpose to PyTorch format: (N, Channels, Height, Width)
    images_np = np.transpose(images_np, (0, 3, 1, 2))

    # 3. Save to cache
    logger.info(f"Saving {dataset_name} data to cache...")
    np.save(cache_ids_path, ids_np)
    np.save(cache_imgs_path, images_np)

    if has_labels:
        labels_np = np.array(labels_list, dtype=np.float32)
        np.save(cache_lbls_path, labels_np)
        return ids_np, images_np, labels_np
    else:
        return ids_np, images_np
