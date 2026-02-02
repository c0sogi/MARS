import os
import numpy as np
import pandas as pd
from library.config import INPUT_DIR, CACHED_X_PATH, CACHED_Y_PATH, SEED, WORKING_DIR
from library.utils import load_normalized_image


def extract_patch_data(metadata_path, patch_size, num_samples, load_cached_data=True):
    """
    Extracts patch features and pixel targets from images listed in the metadata.

    This function implements a caching mechanism. If cache files exist and
    load_cached_data is True, it loads them. Otherwise, it processes the images,
    extracts patches, saves the result to cache, and returns the data.

    Args:
        metadata_path (str): Path to the metadata CSV file (e.g., train.csv).
        patch_size (int): The dimension (k) of the square patch (k x k).
        num_samples (int): The total number of patches to sample from the dataset.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: A tuple (X, y) containing:
            - X (np.ndarray): Feature matrix of shape (num_samples, patch_size^2).
            - y (np.ndarray): Target vector of shape (num_samples,).
    """
    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- 1. Caching Logic ---
    if load_cached_data:
        if os.path.exists(CACHED_X_PATH) and os.path.exists(CACHED_Y_PATH):
            print(f"Loading cached data from {WORKING_DIR}...")
            try:
                X = np.load(CACHED_X_PATH)
                y = np.load(CACHED_Y_PATH)
                print(f"Loaded cached data: X shape={X.shape}, y shape={y.shape}")
                return X, y
            except Exception as e:
                print(f"Failed to load cache: {e}. Proceeding to recompute.")
        else:
            print("Cache files not found. Computing from scratch...")
    else:
        print("Caching disabled or forced recompute. Computing from scratch...")

    # --- 2. Data Extraction ---
    print(f"Reading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Ensure we only process rows with valid targets
    if "target_path" in df.columns:
        df = df.dropna(subset=["target_path"])
    else:
        raise ValueError(
            "Metadata provided does not contain 'target_path'. Cannot extract targets."
        )

    num_images = len(df)
    if num_images == 0:
        raise ValueError("No valid images found in metadata.")

    # Determine samples per image to reach the target num_samples
    # We sample slightly more to ensure we meet the requirement after shuffling/trimming
    samples_per_image = int(np.ceil(num_samples / num_images))

    print(
        f"Processing {num_images} images. Target samples per image: {samples_per_image}"
    )

    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        input_rel_path = row["input_path"]
        target_rel_path = row["target_path"]

        input_full_path = os.path.join(INPUT_DIR, input_rel_path)
        target_full_path = os.path.join(INPUT_DIR, target_rel_path)

        try:
            # Load images normalized to [0, 1]
            img_in = load_normalized_image(input_full_path)
            img_tar = load_normalized_image(target_full_path)
        except Exception as e:
            print(
                f"Warning: Could not load image pair for {input_rel_path}. Error: {e}"
            )
            continue

        h, w = img_in.shape

        # Ensure image is large enough for patch extraction
        if h < patch_size or w < patch_size:
            continue

        # Generate random top-left coordinates for patches
        # We sample valid crops without padding for training
        rows = np.random.randint(0, h - patch_size + 1, size=samples_per_image)
        cols = np.random.randint(0, w - patch_size + 1, size=samples_per_image)

        # Extract patches
        for r, c in zip(rows, cols):
            patch_in = img_in[r : r + patch_size, c : c + patch_size]
            patch_tar = img_tar[r : r + patch_size, c : c + patch_size]

            # Add channel dimension (1, H, W)
            X_list.append(patch_in[np.newaxis, :, :])
            y_list.append(patch_tar[np.newaxis, :, :])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Trim to the exact requested number of samples
    if len(X) > num_samples:
        X = X[:num_samples]
        y = y[:num_samples]

    print(f"Extraction complete. Final shapes: X={X.shape}, y={y.shape}")

    # --- 3. Save to Cache ---
    print(f"Saving processed data to cache at {WORKING_DIR}...")
    np.save(CACHED_X_PATH, X)
    np.save(CACHED_Y_PATH, y)

    return X, y
