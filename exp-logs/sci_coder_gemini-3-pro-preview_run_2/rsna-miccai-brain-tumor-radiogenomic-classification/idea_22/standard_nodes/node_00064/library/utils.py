import os
import numpy as np
import cv2
import pandas as pd
from library import config


def read_dicom_raw(path):
    """
    Reads a DICOM file using raw binary tail-read to bypass header parsing.
    Assumes Little Endian uint16 data at the end of the file.
    Infers resolution (512x512 or 256x256) based on file size.
    """
    try:
        file_size = os.path.getsize(path)

        # Determine resolution based on file size
        # 512x512x2 = 524288 bytes
        # 256x256x2 = 131072 bytes
        # Allow for variable header size
        if file_size >= 524288:
            rows, cols = 512, 512
            expected_bytes = 524288
        elif file_size >= 131072:
            rows, cols = 256, 256
            expected_bytes = 131072
        else:
            # Fallback heuristic for smaller images
            possible_pixels = file_size // 2
            root = int(np.sqrt(possible_pixels))
            if root * root * 2 <= file_size:
                rows, cols = root, root
                expected_bytes = root * root * 2
            else:
                return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

        with open(path, "rb") as f:
            f.seek(-expected_bytes, 2)  # Seek from end
            buffer = f.read(expected_bytes)

        img = np.frombuffer(buffer, dtype=np.uint16).copy()
        img = img.reshape((rows, cols))
        return img.astype(np.float32)

    except Exception:
        # Return zero image on failure to avoid crashing the pipeline
        return np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)


def resize_image(image):
    """Resizes image to config.IMG_SIZE using Area interpolation."""
    return cv2.resize(
        image, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA
    )


def min_max_normalize(image):
    """Applies independent min-max normalization to [0, 1]."""
    _min = image.min()
    _max = image.max()
    if _max > _min:
        return (image - _min) / (_max - _min)
    else:
        return np.zeros_like(image)


def get_sorted_image_files(folder_path):
    """Returns sorted list of .dcm files in a folder."""
    if not os.path.exists(folder_path):
        return []
    files = [f for f in os.listdir(folder_path) if f.endswith(".dcm")]
    # Sort by the integer number in Image-X.dcm
    try:
        files.sort(key=lambda x: int(x.split("-")[1].split(".")[0]))
    except:
        files.sort()  # Fallback
    return files


def get_best_slice_index(flair_full_path):
    """
    Determines the anchor slice index using Sum of Intensity on FLAIR.
    Cite solution_lesson_node_00038: Prefer integral statistics (Sum) over extremal (Max).
    Cite solution_lesson_node_00026: Use Intensity-Based Heuristics (Sum) on hyperintense pathology (FLAIR).
    """
    files = get_sorted_image_files(flair_full_path)
    if not files:
        return 0

    slice_sums = []
    # We need to read all files to find the best one.
    for f in files:
        path = os.path.join(flair_full_path, f)
        img = read_dicom_raw(path)
        slice_sums.append(np.sum(img))

    slice_sums = np.array(slice_sums)
    total_energy = np.sum(slice_sums)

    if total_energy == 0:
        return len(files) // 2

    # Adaptive Safe Zone: 10% to 90% cumulative energy
    # This avoids selecting slices at the very start/end which might be noisy but have high sum due to artifacts
    cumsum = np.cumsum(slice_sums)
    start_idx = np.argmax(cumsum >= 0.10 * total_energy)
    end_idx = np.argmax(cumsum >= 0.90 * total_energy)

    if start_idx >= end_idx:
        # Fallback to full range if safe zone is invalid
        start_idx = 0
        end_idx = len(files) - 1

    # Find slice with maximum intensity sum within the safe zone
    valid_sums = slice_sums[start_idx : end_idx + 1]
    if len(valid_sums) == 0:
        return len(files) // 2

    local_argmax = np.argmax(valid_sums)
    best_idx = start_idx + local_argmax

    return best_idx


def process_patient(row):
    """
    Executes the Robust ROI Pipeline for a single patient.
    Returns:
        X: (12, 224, 224) float32 tensor
        y: scalar label (or -1 if test)
    """
    # Construct full paths
    paths = {
        "FLAIR": os.path.join(config.INPUT_DIR, row["path_FLAIR"]),
        "T1w": os.path.join(config.INPUT_DIR, row["path_T1w"]),
        "T1wCE": os.path.join(config.INPUT_DIR, row["path_T1wCE"]),
        "T2w": os.path.join(config.INPUT_DIR, row["path_T2w"]),
    }

    # Get Anchor from FLAIR using Sum Intensity (Robust Heuristic)
    anchor_idx = get_best_slice_index(paths["FLAIR"])

    # Stacking
    channels = []

    # Order: FLAIR, T1w, T1wCE, T2w
    # This order is crucial for the Grouped Convolutional Stem (groups=4)
    for mod in config.MODALITIES:
        mod_path = paths[mod]
        files = get_sorted_image_files(mod_path)
        num_files = len(files)

        if num_files == 0:
            # Handle missing modality with zeros
            for _ in range(config.NUM_SLICES):
                channels.append(
                    np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)
                )
            continue

        # Determine slice indices: Anchor - Stride, Anchor, Anchor + Stride
        indices = [anchor_idx - config.STRIDE, anchor_idx, anchor_idx + config.STRIDE]

        for idx in indices:
            # Edge Clamping
            clamp_idx = max(0, min(idx, num_files - 1))

            file_path = os.path.join(mod_path, files[clamp_idx])
            img = read_dicom_raw(file_path)
            img = resize_image(img)
            img = min_max_normalize(img)
            channels.append(img)

    # Stack channels
    X = np.stack(channels, axis=0).astype(np.float32)  # (12, 224, 224)

    # Get Label
    if "MGMT_value" in row:
        y = float(row["MGMT_value"])
    else:
        y = -1.0

    return X, y


def get_dataset(df, cache_prefix, load_cached_data=True):
    """
    Generates or loads the dataset.
    Args:
        df: DataFrame containing metadata.
        cache_prefix: 'train', 'val', or 'test'.
        load_cached_data: Boolean to use cache.
    Returns:
        X: numpy array
        y: numpy array
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    path_data = os.path.join(cache_dir, f"{cache_prefix}_data.npy")
    path_labels = os.path.join(cache_dir, f"{cache_prefix}_labels.npy")

    if load_cached_data and os.path.exists(path_data) and os.path.exists(path_labels):
        print(f"Loading cached {cache_prefix} data from {cache_dir}...")
        X = np.load(path_data)
        y = np.load(path_labels)
        return X, y

    print(f"Processing {cache_prefix} data ({len(df)} subjects)...")

    X_list = []
    y_list = []

    for i, (idx, row) in enumerate(df.iterrows()):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(df)}")

        X_subj, y_subj = process_patient(row)
        X_list.append(X_subj)
        y_list.append(y_subj)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"Saving {cache_prefix} cache to {cache_dir}...")
    np.save(path_data, X)
    np.save(path_labels, y)

    return X, y
