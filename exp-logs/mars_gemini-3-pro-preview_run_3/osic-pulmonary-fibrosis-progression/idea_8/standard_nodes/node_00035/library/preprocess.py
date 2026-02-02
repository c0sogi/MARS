import os
import glob
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    IMG_SIZE,
    STATS,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    WORKING_DIR,
)
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(42)


def load_scan(path):
    """
    Loads all DICOM files from a directory, sorts them by Z-position,
    and handles missing files gracefully.
    """
    if not os.path.exists(path):
        return []

    slices = []
    for s in os.listdir(path):
        if s.endswith(".dcm"):
            try:
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
            except Exception:
                continue

    if not slices:
        return []

    # Sort by ImagePositionPatient Z coordinate (index 2)
    # If ImagePositionPatient is missing, fall back to InstanceNumber
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda x: float(x.InstanceNumber))

    return slices


def get_pixels_hu(slices):
    """
    Converts a list of pydicom datasets to a numpy array of Hounsfield Units.
    Handles slope/intercept conversion and sets outside-scan pixels to 0 (air).
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_and_resize(img, img_size=IMG_SIZE):
    """
    Applies lung windowing and resizes the image.
    Window: Level -600, Width 1500 (Range -1350 to 150)
    """
    # Lung Window
    L = -600
    W = 1500
    min_hu = L - W // 2
    max_hu = L + W // 2

    img = (img - min_hu) / (max_hu - min_hu)
    img = np.clip(img, 0, 1)

    # Resize
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size))

    return img


def select_adaptive_slices(image_3d):
    """
    Selects 3 slices (Basal, Middle, Apical) based on lung area.
    1. Middle (Anchor): Slice with maximum lung area.
    2. Apical/Basal: Slices where area drops to ~50% of max.
    """
    # Threshold to approximate lung area (HU < -320 is typically air/lung)
    # We use a simplified check here since we already have HU
    # Air is -1000, Tissue is > -100. -320 is a safe cutoff.
    lung_mask = image_3d < -320
    area_per_slice = np.sum(lung_mask, axis=(1, 2))

    num_slices = len(image_3d)
    if num_slices < 3:
        # Fallback for very few slices: duplicate
        indices = [0, 0, 0] if num_slices == 1 else [0, 0, 1]
    else:
        # Find Anchor (Max Area)
        anchor_idx = np.argmax(area_per_slice)
        max_area = area_per_slice[anchor_idx]
        target_area = max_area * 0.5

        # Find Apical (Top, higher index usually in sorted DICOM if Head-First)
        # We search upwards from anchor
        apical_idx = num_slices - 1
        for i in range(anchor_idx, num_slices):
            if area_per_slice[i] < target_area:
                apical_idx = i
                break

        # Find Basal (Bottom, lower index)
        # We search downwards from anchor
        basal_idx = 0
        for i in range(anchor_idx, -1, -1):
            if area_per_slice[i] < target_area:
                basal_idx = i
                break

        # Ensure distinct indices if possible, else fallback to quantiles
        if basal_idx == anchor_idx:
            basal_idx = max(0, anchor_idx - 1)
        if apical_idx == anchor_idx:
            apical_idx = min(num_slices - 1, anchor_idx + 1)

        indices = sorted([basal_idx, anchor_idx, apical_idx])

    # Extract, Resize, Normalize
    selected_slices = []
    for idx in indices:
        slice_img = image_3d[idx]
        processed = normalize_and_resize(slice_img)
        selected_slices.append(processed)

    return np.stack(selected_slices)  # (3, H, W)


def process_patient(patient_id, image_dir, save_path):
    """
    Loads, processes, and saves a patient's CT scan.
    """
    patient_dir = os.path.join(image_dir, patient_id)
    slices = load_scan(patient_dir)

    if not slices:
        # Create a blank volume if loading fails (robustness)
        # Shape (3, IMG_SIZE, IMG_SIZE)
        processed_vol = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        try:
            vol_hu = get_pixels_hu(slices)
            processed_vol = select_adaptive_slices(vol_hu)
        except Exception as e:
            # Fallback for processing errors
            processed_vol = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    np.save(save_path, processed_vol.astype(np.float32))


def run_preprocessing(load_cached_data=True):
    """
    Main entry point for preprocessing.
    Processes and caches CT scans for all patients.
    """
    print("Starting Preprocessing...")

    # Load Metadata
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Process Images
    # Get unique patients from all sets
    all_patients = pd.concat(
        [train_df[["Patient"]], val_df[["Patient"]], test_df[["Patient"]]]
    )["Patient"].unique()

    print(f"Processing images for {len(all_patients)} patients...")

    # Determine image source directory (INPUT_DIR/train or INPUT_DIR/test)
    # The metadata contains relative paths, but we can infer based on ID presence
    # Actually, simpler: check both folders or use the metadata 'image_path' if we loaded it.
    # Since we just have IDs here, we check existence.

    count = 0
    for patient_id in all_patients:
        save_path = os.path.join(CACHE_DIR, f"{patient_id}.npy")

        # Check cache
        if load_cached_data and os.path.exists(save_path):
            continue

        # Determine path
        train_path = os.path.join(INPUT_DIR, "train", patient_id)
        test_path = os.path.join(INPUT_DIR, "test", patient_id)

        if os.path.exists(train_path):
            image_dir = os.path.join(INPUT_DIR, "train")
        elif os.path.exists(test_path):
            image_dir = os.path.join(INPUT_DIR, "test")
        else:
            # Should not happen given metadata is valid
            continue

        process_patient(patient_id, image_dir, save_path)
        count += 1
        if count % 50 == 0:
            print(f"Processed {count} images...")

    print("Preprocessing complete.")
