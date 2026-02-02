import os
import re
import numpy as np
import pandas as pd
import pydicom
import cv2
from library.config import INPUT_DIR, IMG_SIZE, SELECTED_MODALITIES


def natural_sort_key(s):
    """
    Sorts strings containing numbers in a natural way (e.g., Image-1, Image-2, Image-10).
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_middle_slice(folder_path):
    """
    Selects and reads exactly the middle DICOM file from a folder.
    This deterministic heuristic is robust for registered datasets (Cite Lesson 00036).
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    if not files:
        raise ValueError(f"No DICOM files found in {folder_path}")

    # Sort files naturally to ensure correct Z-ordering
    files.sort(key=natural_sort_key)

    # Select middle file index
    idx = len(files) // 2
    fpath = os.path.join(folder_path, files[idx])

    try:
        ds = pydicom.dcmread(fpath)
        return ds.pixel_array
    except Exception as e:
        raise IOError(f"Failed to read DICOM file {fpath}: {e}")


def process_subject(row):
    """
    Loads modalities using Median Slice heuristic, processes, and stacks into a 3-channel image.
    Returns: (H, W, 3) float32 image, label, BraTS21ID
    """
    channels = []

    # SELECTED_MODALITIES defined in config (e.g., ["FLAIR", "T1wCE", "T2w"])
    for mod in SELECTED_MODALITIES:
        # Map config modality name to metadata column name (e.g., FLAIR -> flair_path)
        col_name = f"{mod.lower()}_path"
        if col_name not in row:
            raise KeyError(f"Metadata row missing column: {col_name}")

        rel_path = row[col_name]
        full_path = os.path.join(INPUT_DIR, rel_path)

        # 1. Load Middle Slice (Cite Lesson 00036, Lesson 00015)
        # Replaces unstable Center-of-Mass logic with robust geometric median
        slc = get_middle_slice(full_path)

        # 2. Preprocessing
        # Resize to target dimensions (IMG_SIZE x IMG_SIZE)
        if slc.shape[:2] != (IMG_SIZE, IMG_SIZE):
            slc = cv2.resize(slc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # Convert to float32 for precision
        slc = slc.astype(np.float32)

        # Independent Min-Max Normalization to [0, 1] (Cite Lesson 00017)
        min_val = slc.min()
        max_val = slc.max()
        if max_val > min_val:
            slc = (slc - min_val) / (max_val - min_val)
        else:
            slc = np.zeros_like(slc)

        channels.append(slc)

    # Stack channels: (H, W, C)
    # This format is standard for image libraries (Albumentations/OpenCV)
    img_multichannel = np.stack(channels, axis=-1)

    # Get Label and ID
    # Test data might not have MGMT_value, handle gracefully
    label = row["MGMT_value"] if "MGMT_value" in row else -1.0
    braTS21ID = row["BraTS21ID"]

    return img_multichannel, label, braTS21ID


def generate_dataset(
    metadata_csv_path, cache_file, load_cached_data=True, debug=False, debug_size=50
):
    """
    Generates the dataset by processing all subjects in the metadata file.
    Implements caching to speed up subsequent runs and avoid re-processing raw DICOMs.
    """
    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}...")
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            print(f"Loaded {len(data['ids'])} samples from cache.")
            return data["images"], data["labels"], data["ids"]
        except Exception as e:
            print(f"Failed to load cache: {e}. Regenerating dataset...")

    # 2. Load Metadata
    df = pd.read_csv(metadata_csv_path)

    if debug:
        print(f"Debug mode: Processing first {debug_size} samples.")
        df = df.head(debug_size)

    images = []
    labels = []
    ids = []

    print(f"Processing {len(df)} subjects from {metadata_csv_path}...")

    # 3. Process Subjects
    for idx, row in df.iterrows():
        try:
            img, lbl, sid = process_subject(row)
            images.append(img)
            labels.append(lbl)
            ids.append(sid)
        except Exception as e:
            # Strict integrity implies we track failures, but we skip the bad subject
            # to allow the pipeline to continue for the valid majority.
            print(f"Skipping Subject {row.get('BraTS21ID', 'Unknown')}: {e}")
            continue

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    # 4. Save to Cache
    if len(images) > 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, {"images": images, "labels": labels, "ids": ids})
        print(f"Saved processed dataset to {cache_file}")
    else:
        print("Warning: No images processed successfully. Cache not saved.")

    return images, labels, ids
