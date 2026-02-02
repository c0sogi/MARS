import os
import glob
import re
import numpy as np
import cv2
import pandas as pd
from library.config import (
    IMG_SIZE,
    NUM_SLICES_PER_MODALITY,
    NUM_MODALITIES,
    CIRCUIT_BREAKER_THRESHOLD,
    WORKING_DIR,
)

# Attempt to import pydicom for fallback; handle environment where it might be missing
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def natural_sort_key(s):
    """
    Key for natural sorting of filenames (e.g., Image-1.dcm before Image-10.dcm).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", os.path.basename(s))
    ]


def read_dicom_raw(filepath):
    """
    Reads a DICOM file using a 'Raw Binary Tail-Read' heuristic to bypass expensive header parsing.
    Falls back to pydicom or manual parsing if the heuristic fails.

    Args:
        filepath (str): Path to the .dcm file.

    Returns:
        np.ndarray: 2D numpy array of the image data (uint16).
    """
    try:
        file_size = os.path.getsize(filepath)

        # Heuristic candidates: (Height, Width)
        # Assuming uint16 (2 bytes per pixel)
        # 512x512 = 262144 pixels -> 524288 bytes
        # 256x256 = 65536 pixels -> 131072 bytes
        candidates = [(512, 512), (256, 256)]

        with open(filepath, "rb") as f:
            data = f.read()

        # 1. Try Raw Binary Tail-Read
        for h, w in candidates:
            expected_data_size = h * w * 2
            # Check if file is slightly larger than data (header overhead)
            # A valid DICOM usually has a header > 128 bytes.
            # We allow up to 8KB header for standard MRI DICOMs.
            diff = file_size - expected_data_size

            if 128 <= diff < 8192:
                # Extract tail
                raw_bytes = data[-expected_data_size:]
                arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                if arr.size == h * w:
                    return arr.reshape((h, w))

        # 2. Fallback: pydicom (if available)
        if HAS_PYDICOM:
            try:
                ds = pydicom.dcmread(filepath)
                return ds.pixel_array
            except Exception:
                pass  # Fall through to manual parse

        # 3. Fallback: Manual Tag Parsing (Last Resort)
        # Look for PixelData tag (7FE0, 0010) -> LE: E0 7F 10 00
        tag_marker = b"\xe0\x7f\x10\x00"
        idx = data.find(tag_marker)
        if idx != -1:
            # Attempt to determine start of data
            # Check VR (Value Representation) at idx + 4
            vr = data[idx + 4 : idx + 6]
            if vr in [b"OB", b"OW"]:
                # Explicit VR: Tag(4) + VR(2) + Reserved(2) + Len(4)
                start = idx + 12
            else:
                # Implicit VR or other: Tag(4) + Len(4) (usually)
                # Or Explicit with short length?
                # Safe bet for Implicit LE is Tag+Len=8 bytes
                start = idx + 8

            # Try to match remaining data to a candidate shape
            remaining = len(data) - start
            for h, w in candidates:
                if remaining == h * w * 2:
                    arr = np.frombuffer(data[start:], dtype=np.uint16)
                    return arr.reshape((h, w))

        raise ValueError("All reading methods failed.")

    except Exception as e:
        raise ValueError(f"Could not read DICOM {filepath}: {e}")


def get_image_plane(data):
    """
    Resizes image to the global IMG_SIZE (256x256) if necessary.
    """
    if data.shape != (IMG_SIZE, IMG_SIZE):
        data = cv2.resize(data, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return data


def load_and_preprocess_dataset(df, cache_prefix, input_dir, load_cached_data=True):
    """
    Loads, processes, and caches the dataset. Implements Circuit Breaker and Pre-Caching.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        cache_prefix (str): Prefix for cache files (e.g., 'train', 'val', 'test').
        input_dir (str): Base input directory.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (data_array, labels_array/ids_array)
    """
    # Define cache paths
    cache_data_path = os.path.join(WORKING_DIR, f"{cache_prefix}_data.npy")
    cache_meta_path = os.path.join(
        WORKING_DIR, f"{cache_prefix}_labels.npy"
    )  # Stores labels or IDs

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(cache_data_path)
        and os.path.exists(cache_meta_path)
    ):
        print(f"Loading {cache_prefix} set from cache...")
        try:
            data = np.load(cache_data_path)
            meta = np.load(cache_meta_path)
            return data, meta
        except Exception as e:
            print(f"Cache load failed ({e}). Reprocessing...")

    # 2. Reprocess from Scratch
    print(f"Processing {cache_prefix} set from raw DICOMs...")

    processed_data = []
    processed_meta = []  # labels for train/val, IDs for test

    failed_count = 0
    total_subjects = len(df)

    # Pre-calculate modality paths
    # We assume the dataframe has 'BraTS21ID' and paths like 'path_FLAIR'

    for idx, row in df.iterrows():
        subject_id = row["BraTS21ID"]

        try:
            # ---------------------------------------------------------
            # A. Anchor Selection (Fidelity-Aligned)
            # ---------------------------------------------------------
            # Use FLAIR to find the geometric center of the tumor/brain
            flair_path_rel = row["path_FLAIR"]
            flair_dir = os.path.join(input_dir, flair_path_rel)

            # Get all DICOM files
            flair_files = sorted(
                glob.glob(os.path.join(flair_dir, "*.dcm")), key=natural_sort_key
            )

            if not flair_files:
                raise FileNotFoundError(f"No FLAIR files for {subject_id}")

            num_files = len(flair_files)

            # Restrict search to 15% - 85% depth to avoid noise at ends
            start_idx = int(num_files * 0.15)
            end_idx = int(num_files * 0.85)
            search_files = flair_files[start_idx:end_idx]

            if not search_files:
                search_files = flair_files  # Fallback if too few files

            # Find slice with maximum intensity integral
            max_integral = -1
            best_file = search_files[len(search_files) // 2]  # Default to middle

            # Optimization: Check every 2nd slice to speed up anchor search
            for fpath in search_files[::2]:
                img = read_dicom_raw(fpath)
                integral = np.sum(img)
                if integral > max_integral:
                    max_integral = integral
                    best_file = fpath

            # Extract the integer ID of the anchor slice (e.g., Image-123.dcm -> 123)
            anchor_filename = os.path.basename(best_file)
            anchor_id = int(re.search(r"(\d+)", anchor_filename).group(1))

            # ---------------------------------------------------------
            # B. Volume Extraction (Stacked Modalities)
            # ---------------------------------------------------------
            # We need 3 slices per modality: [Anchor-5, Anchor, Anchor+5]
            offsets = [-5, 0, 5]
            target_ids = [anchor_id + off for off in offsets]

            subject_channels = []

            modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]

            for mod in modalities:
                mod_path_rel = row[f"path_{mod}"]
                mod_dir = os.path.join(input_dir, mod_path_rel)

                # We need specific files: Image-{id}.dcm
                # If a file doesn't exist (out of bounds), clamp to nearest available
                # To do this efficiently, we list all available IDs first
                all_files = glob.glob(os.path.join(mod_dir, "*.dcm"))
                if not all_files:
                    # If a modality is completely missing, fill with zeros
                    for _ in range(NUM_SLICES_PER_MODALITY):
                        subject_channels.append(
                            np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
                        )
                    continue

                # Map ID -> Filename
                id_to_file = {}
                available_ids = []
                for f in all_files:
                    fid = int(re.search(r"(\d+)", os.path.basename(f)).group(1))
                    id_to_file[fid] = f
                    available_ids.append(fid)

                available_ids.sort()
                min_id, max_id = available_ids[0], available_ids[-1]

                for tid in target_ids:
                    # Clamping logic
                    if tid < min_id:
                        fetch_id = min_id
                    elif tid > max_id:
                        fetch_id = max_id
                    elif tid in id_to_file:
                        fetch_id = tid
                    else:
                        # ID inside range but missing? Find nearest.
                        # (Simple linear search since list is small)
                        fetch_id = min(available_ids, key=lambda x: abs(x - tid))

                    # Read and Process
                    img_raw = read_dicom_raw(id_to_file[fetch_id])
                    img_resized = get_image_plane(img_raw)

                    # Normalize [0, 1] per channel independently
                    img_float = img_resized.astype(np.float32)
                    if img_float.max() > 0:
                        img_float /= img_float.max()

                    subject_channels.append(img_float)

            # Stack channels: (12, 256, 256)
            subject_volume = np.stack(subject_channels, axis=0)
            processed_data.append(subject_volume)

            # Handle Label or ID
            if "MGMT_value" in row:
                processed_meta.append(row["MGMT_value"])
            else:
                processed_meta.append(subject_id)

        except Exception as e:
            failed_count += 1
            # print(f"Failed subject {subject_id}: {e}") # Optional logging

    # ---------------------------------------------------------
    # C. Circuit Breaker
    # ---------------------------------------------------------
    fail_ratio = failed_count / total_subjects if total_subjects > 0 else 0
    if fail_ratio > CIRCUIT_BREAKER_THRESHOLD:
        raise RuntimeError(
            f"Circuit Breaker Triggered! Failure ratio {fail_ratio:.4f} "
            f"exceeds threshold {CIRCUIT_BREAKER_THRESHOLD}. "
            "Pipeline stopped to prevent silent failure."
        )

    if len(processed_data) == 0:
        raise RuntimeError("No data processed successfully.")

    # Convert to numpy
    data_array = np.array(processed_data, dtype=np.float32)
    meta_array = np.array(processed_meta)  # int or float

    # 3. Save to Cache
    os.makedirs(WORKING_DIR, exist_ok=True)
    np.save(cache_data_path, data_array)
    np.save(cache_meta_path, meta_array)

    print(f"Successfully processed and cached {len(data_array)} samples.")
    return data_array, meta_array
