import os
import random
import numpy as np
import torch
import pandas as pd
import cv2


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the available device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_dicom_manual(path):
    """
    Reads a DICOM file assuming standard uncompressed pixel data.
    Uses file size heuristics to determine dimensions (512x512 or 256x256).
    Returns a numpy array (uint16) or a zero array if reading fails.
    """
    try:
        if not os.path.exists(path):
            return np.zeros((256, 256), dtype=np.uint16)

        file_size = os.path.getsize(path)

        # Heuristic for image size based on file size + header
        # 512x512x2 = 524288 bytes
        # 256x256x2 = 131072 bytes

        if file_size > 524288:
            shape = (512, 512)
            pixel_bytes = 512 * 512 * 2
        elif file_size > 131072:
            shape = (256, 256)
            pixel_bytes = 256 * 256 * 2
        else:
            # Fallback for unexpected sizes (return black image)
            return np.zeros((256, 256), dtype=np.uint16)

        # Calculate offset to read the last N bytes corresponding to pixel data
        offset = file_size - pixel_bytes
        if offset < 0:
            return np.zeros((256, 256), dtype=np.uint16)

        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(pixel_bytes)

        # Load as uint16 (standard for MRI)
        img = np.frombuffer(data, dtype=np.uint16).reshape(shape)
        return img

    except Exception:
        return np.zeros((256, 256), dtype=np.uint16)


def process_patient(row, input_dir="./input"):
    """
    Processes a single patient's data for the Siamese Network.
    - Loads 4 modalities (FLAIR, T1w, T1wCE, T2w).
    - Performs Global Volumetric Normalization per modality.
    - Samples 32 slices uniformly from the 10%-90% depth range.
    - Resizes slices to 256x256.

    Returns:
        np.ndarray: Shape (4, 32, 256, 256), dtype float32.
    """
    modalities = ["flair_paths", "t1w_paths", "t1wce_paths", "t2w_paths"]
    patient_data = []

    for mod_col in modalities:
        paths = row[mod_col]

        # Handle missing modality case
        if paths is None or len(paths) == 0:
            patient_data.append(np.zeros((32, 256, 256), dtype=np.float32))
            continue

        # Load all slices for this modality to form a volume
        vol_slices = []
        for p in paths:
            full_path = os.path.join(input_dir, p)
            img = read_dicom_manual(full_path)
            vol_slices.append(img)

        if len(vol_slices) == 0:
            patient_data.append(np.zeros((32, 256, 256), dtype=np.float32))
            continue

        volume = np.stack(vol_slices)  # Shape: (Depth, H, W)

        # Global Volumetric Normalization (0-1 range)
        v_min = volume.min()
        v_max = volume.max()
        if v_max - v_min > 0:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = np.zeros_like(volume, dtype=np.float32)

        # High-Density Uniform Sampling (32 slices)
        depth = volume.shape[0]
        if depth < 1:
            patient_data.append(np.zeros((32, 256, 256), dtype=np.float32))
            continue

        # Select indices from 10% to 90% of the volume depth
        start = int(depth * 0.1)
        end = int(depth * 0.9)
        if end <= start:
            start = 0
            end = depth - 1

        # Generate 32 uniformly spaced indices
        indices = np.linspace(start, end, 32)
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, depth - 1)

        sampled_slices = volume[indices]  # Shape: (32, H, W)

        # Resize each slice to 256x256
        resized_slices = []
        for i in range(32):
            slc = sampled_slices[i]
            # cv2.resize expects (width, height)
            if slc.shape[0] != 256 or slc.shape[1] != 256:
                slc = cv2.resize(slc, (256, 256), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(slc)

        patient_data.append(np.stack(resized_slices))

    # Stack modalities to get (4, 32, 256, 256)
    return np.stack(patient_data).astype(np.float32)


def load_dataset(metadata_path, cache_dir, load_cached=True, input_dir="./input"):
    """
    Loads the dataset, using a caching mechanism to save processed arrays.

    Args:
        metadata_path (str): Path to the parquet metadata file.
        cache_dir (str): Directory to store/load .npy cache files.
        load_cached (bool): Whether to attempt loading from cache.
        input_dir (str): Root directory of the input data.

    Returns:
        tuple: (X, y, ids)
            X (np.ndarray): Input data of shape (N, 4, 32, 256, 256).
            y (np.ndarray or None): Target labels of shape (N,). None for test set.
            ids (np.ndarray): BraTS21IDs of shape (N,).
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    base_name = os.path.splitext(os.path.basename(metadata_path))[0]
    cache_X_path = os.path.join(cache_dir, f"cached_{base_name}_X.npy")
    cache_y_path = os.path.join(cache_dir, f"cached_{base_name}_y.npy")
    cache_ids_path = os.path.join(cache_dir, f"cached_{base_name}_ids.npy")

    # Try loading from cache
    if load_cached and os.path.exists(cache_X_path) and os.path.exists(cache_ids_path):
        print(f"Loading cached data for {base_name} from {cache_dir}...")
        X = np.load(cache_X_path)
        ids = np.load(cache_ids_path)
        if os.path.exists(cache_y_path):
            y = np.load(cache_y_path)
        else:
            y = None
        return X, y, ids

    # Process data from scratch
    print(f"Processing data from {metadata_path}...")
    df = pd.read_parquet(metadata_path)

    X_list = []
    y_list = []
    ids_list = []

    for idx, row in df.iterrows():
        p_data = process_patient(row, input_dir=input_dir)
        X_list.append(p_data)
        ids_list.append(str(row["BraTS21ID"]))

        if "MGMT_value" in row:
            y_list.append(row["MGMT_value"])

    X = np.stack(X_list)  # Shape: (N, 4, 32, 256, 256)
    ids = np.array(ids_list)

    if len(y_list) > 0:
        y = np.array(y_list, dtype=np.float32)
    else:
        y = None

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(cache_X_path, X)
    np.save(cache_ids_path, ids)
    if y is not None:
        np.save(cache_y_path, y)

    return X, y, ids
