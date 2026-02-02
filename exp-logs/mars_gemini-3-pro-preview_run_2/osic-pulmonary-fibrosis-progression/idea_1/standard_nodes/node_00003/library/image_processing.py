import os
import numpy as np
import pandas as pd
import warnings
from library.config import (
    INPUT_DIR,
    RADIOMICS_CACHE_PATH,
    HU_MIN,
    HU_MAX,
    RADIOMICS_FEATURES,
    CACHE_DIR,
)

# Attempt to import pydicom for DICOM processing
try:
    import pydicom

    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    warnings.warn(
        "pydicom module not found. Radiomics features will be approximated using file statistics."
    )


def load_scan(path):
    """
    Loads all DICOM slices from a directory and sorts them.

    Args:
        path (str): Path to the directory containing .dcm files.

    Returns:
        list: A list of pydicom datasets (if available) or file paths, sorted by instance/position.
    """
    if not os.path.exists(path):
        return []

    files = [f for f in os.listdir(path) if f.lower().endswith(".dcm")]
    if not files:
        return []

    if PYDICOM_AVAILABLE:
        slices = []
        for f in files:
            try:
                ds = pydicom.dcmread(os.path.join(path, f))
                slices.append(ds)
            except Exception:
                continue

        # Sort slices
        # Try sorting by ImagePositionPatient[2] (Z-axis), then InstanceNumber, then filename
        try:
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except AttributeError:
                slices.sort(key=lambda x: x.filename)
        return slices
    else:
        # Fallback: just return sorted filenames to ensure deterministic order
        files.sort()
        return [os.path.join(path, f) for f in files]


def get_pixels_hu(scans):
    """
    Converts a list of pydicom datasets to a 3D numpy array of Hounsfield Units.
    """
    if not scans or not PYDICOM_AVAILABLE:
        return np.array([])

    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Convert to Hounsfield Units (HU)
    # The intercept is usually -1024, so air is approximately -1000
    for slice_number in range(len(scans)):
        intercept = scans[slice_number].RescaleIntercept
        slope = scans[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def extract_global_stats(path):
    """
    Extracts global radiomics features from a patient's CT scan.

    Features:
        - Lung_Volume: Number of voxels (or slices in fallback) within HU threshold.
        - Mean_Density: Mean HU value of segmented tissue.
        - Density_Variance: Standard deviation of HU value of segmented tissue.

    Args:
        path (str): Full path to the patient's DICOM directory.

    Returns:
        dict: Dictionary containing the extracted features.
    """
    # Default fallback values
    stats = {"Lung_Volume": 0, "Mean_Density": 0.0, "Density_Variance": 0.0}

    scans = load_scan(path)

    if not scans:
        return stats

    if PYDICOM_AVAILABLE:
        try:
            # Get 3D volume in HU
            hu_volume = get_pixels_hu(scans)

            if hu_volume.size == 0:
                stats["Lung_Volume"] = len(scans)  # Fallback to slice count
                return stats

            # Segment lung tissue based on thresholds
            # Lung tissue is typically between -1000 and -400 HU
            mask = (hu_volume >= HU_MIN) & (hu_volume <= HU_MAX)

            if np.sum(mask) == 0:
                # No lung tissue found (possible bad threshold or artifacts)
                # Fallback to whole image stats or just slice count
                stats["Lung_Volume"] = len(scans)
                return stats

            segmented_voxels = hu_volume[mask]

            stats["Lung_Volume"] = int(np.sum(mask))
            stats["Mean_Density"] = float(np.mean(segmented_voxels))
            stats["Density_Variance"] = float(np.std(segmented_voxels))

        except Exception as e:
            # In case of memory errors or corruption, fallback to slice count
            stats["Lung_Volume"] = len(scans)

    else:
        # Fallback if pydicom is not installed
        # Use number of slices as a proxy for volume
        stats["Lung_Volume"] = len(scans)
        stats["Mean_Density"] = 0.0
        stats["Density_Variance"] = 0.0

    return stats


def process_all_scans(metadata_df, load_cached_data=True):
    """
    Iterates over unique patients in the metadata, extracts features,
    and returns a DataFrame. Implements caching.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'Patient' and 'dcm_path'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame with 'Patient' and radiomics features.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    if load_cached_data and os.path.exists(RADIOMICS_CACHE_PATH):
        print(f"Loading radiomics features from cache: {RADIOMICS_CACHE_PATH}")
        try:
            cached_df = pd.read_parquet(RADIOMICS_CACHE_PATH)
            # Verify it covers the patients we need (optional, but good practice)
            # For simplicity, we assume the cache is valid if it exists.
            return cached_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing features.")

    print("Extracting radiomics features from DICOM files...")

    # Get unique patients and their paths
    # We drop duplicates to process each patient only once
    unique_patients = metadata_df[["Patient", "dcm_path"]].drop_duplicates()

    results = []

    for idx, row in unique_patients.iterrows():
        patient_id = row["Patient"]
        rel_path = row["dcm_path"]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # Extract features
        feats = extract_global_stats(full_path)
        feats["Patient"] = patient_id
        results.append(feats)

        # Optional: Print progress every N patients could go here,
        # but we are asked to be silent/minimal.

    # Create DataFrame
    radiomics_df = pd.DataFrame(results)

    # Ensure columns are in expected order (Patient + Features)
    cols = ["Patient"] + RADIOMICS_FEATURES
    radiomics_df = radiomics_df[cols]

    # Save to cache
    print(f"Saving radiomics features to cache: {RADIOMICS_CACHE_PATH}")
    radiomics_df.to_parquet(RADIOMICS_CACHE_PATH, index=False)

    return radiomics_df
