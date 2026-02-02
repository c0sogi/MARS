import os
import numpy as np
import pandas as pd
import librosa
import cv2
from joblib import Parallel, delayed
from library.config import Config


def apply_instance_standardization(image_tensor: np.ndarray) -> np.ndarray:
    """
    Applies per-sample standardization to the image tensor.
    Formula: (x - mean) / (std + epsilon)

    Args:
        image_tensor (np.ndarray): Input tensor of shape (Channels, Height, Width).

    Returns:
        np.ndarray: Standardized tensor.
    """
    mean = np.mean(image_tensor)
    std = np.std(image_tensor)
    # Add epsilon to prevent division by zero
    return (image_tensor - mean) / (std + 1e-6)


def generate_spectrogram(df_segment: pd.DataFrame) -> np.ndarray:
    """
    Converts a dataframe of sensor readings into a stacked Log-Mel Spectrogram tensor.
    Resizes the output to the configured image size (e.g., 224x224).

    Args:
        df_segment (pd.DataFrame): Dataframe containing sensor data (approx 60001 rows).

    Returns:
        np.ndarray: Stacked spectrograms of shape (10, 224, 224).
    """
    # Initialize tensor dimensions
    target_h, target_w = Config.IMG_SIZE
    num_sensors = len(Config.SENSOR_COLS)

    # Pre-allocate array
    output_tensor = np.zeros((num_sensors, target_h, target_w), dtype=np.float32)

    # Fill NaNs with 0 (equilibrium) before processing
    df_segment = df_segment.fillna(0)

    for i, sensor_col in enumerate(Config.SENSOR_COLS):
        signal = df_segment[sensor_col].values.astype(np.float32)

        # Compute Mel Spectrogram
        # Result shape: (n_mels, time_steps)
        melspec = librosa.feature.melspectrogram(
            y=signal,
            sr=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            fmin=Config.F_MIN,
            fmax=Config.F_MAX,
        )

        # Convert to Log-Mel (dB)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # Resize to target dimensions
        # cv2.resize expects dsize=(width, height)
        # Input log_melspec is (height, width) implicitly
        log_melspec = log_melspec.astype(np.float32)
        resized_spec = cv2.resize(
            log_melspec,
            dsize=(target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )

        output_tensor[i, :, :] = resized_spec

    return output_tensor


def _process_single_image(row, input_dir):
    """
    Helper function to process a single CSV file into a standardized spectrogram tensor.
    Executed in parallel.
    """
    segment_id = row["segment_id"]
    file_path = row["file_path"]
    full_path = os.path.join(input_dir, file_path)

    try:
        # Load data (float32 to save memory)
        df = pd.read_csv(full_path, dtype="float32")

        # Generate Spectrogram
        spec_tensor = generate_spectrogram(df)

        # Apply Instance Standardization
        spec_tensor = apply_instance_standardization(spec_tensor)

        return spec_tensor, segment_id
    except Exception as e:
        print(f"Error processing vision segment {segment_id}: {e}")
        return None, None


def process_vision_dataset(dataset_type: str = "train", load_cached_data: bool = True):
    """
    Main function to process the dataset into vision tensors.
    Handles caching, parallel execution, and target alignment.

    Args:
        dataset_type (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to load from cache if available.

    Returns:
        X (np.ndarray): Tensor of shape (N, 10, 224, 224).
        y (np.ndarray): Target array (N,) or None.
        ids (np.ndarray): Array of segment IDs corresponding to X.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    x_path = os.path.join(cache_dir, f"{dataset_type}_vision_X.npy")
    y_path = os.path.join(cache_dir, f"{dataset_type}_vision_y.npy")
    ids_path = os.path.join(cache_dir, f"{dataset_type}_vision_ids.npy")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(x_path) and os.path.exists(ids_path):
        print(f"Loading {dataset_type} vision data from cache: {x_path}")
        X = np.load(x_path)
        ids = np.load(ids_path)

        y = None
        if os.path.exists(y_path):
            y = np.load(y_path)

        return X, y, ids

    # 2. Process from Scratch
    print(f"Generating {dataset_type} vision data from scratch...")

    # Load Metadata
    if dataset_type == "train":
        meta_path = Config.TRAIN_METADATA_PATH
    elif dataset_type == "val":
        meta_path = Config.VAL_METADATA_PATH
    elif dataset_type == "test":
        meta_path = Config.TEST_METADATA_PATH
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    df_meta = pd.read_csv(meta_path)

    # Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_meta = df_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # Parallel Processing
    # joblib preserves the order of the input iterator in the output list
    results = Parallel(n_jobs=Config.NUM_WORKERS, verbose=0)(
        delayed(_process_single_image)(row, Config.INPUT_DIR)
        for _, row in df_meta.iterrows()
    )

    # Filter out failures
    valid_results = [r for r in results if r[0] is not None]

    if not valid_results:
        raise RuntimeError(f"No valid data processed for {dataset_type}")

    X_list = [r[0] for r in valid_results]
    ids_list = [r[1] for r in valid_results]

    # Convert to numpy arrays
    X = np.stack(X_list).astype(np.float32)
    ids = np.array(ids_list)

    # Handle Targets
    y = None
    if "time_to_eruption" in df_meta.columns and dataset_type != "test":
        # Create a mapping from ID to target to ensure alignment
        id_to_target = dict(zip(df_meta["segment_id"], df_meta["time_to_eruption"]))

        # Map targets for the successfully processed IDs
        y_list = [id_to_target[seg_id] for seg_id in ids_list]
        y = np.array(y_list, dtype=np.float32)

    # 3. Save to Cache
    print(f"Saving {dataset_type} vision data to cache...")
    np.save(x_path, X)
    np.save(ids_path, ids)
    if y is not None:
        np.save(y_path, y)

    return X, y, ids
