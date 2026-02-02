import os
import cv2
import numpy as np
import pandas as pd
import torch
from library.config import INPUT_DIR, WORKING_DIR, DEVICE, set_seed


def get_device():
    """
    Returns the PyTorch device configured in the library config.
    """
    return torch.device(DEVICE)


def load_image(path):
    """
    Loads an image from the specified relative path, converts it to grayscale,
    and normalizes pixel intensities to the [0, 1] range.

    Args:
        path (str): Relative path to the image from INPUT_DIR.

    Returns:
        np.ndarray: Normalized grayscale image of shape (H, W).
    """
    full_path = os.path.join(INPUT_DIR, path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Load image (unchanged to detect format, but task specifies grayscale output)
    img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Failed to load image: {full_path}")

    # Convert to grayscale if necessary
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize to [0, 1] float32
    img = img.astype(np.float32) / 255.0

    return img


def load_dataset_images(metadata_df, cache_name, load_cached_data=True):
    """
    Loads all images defined in the metadata DataFrame into memory.
    Implements a caching mechanism using .npz files to speed up subsequent runs.

    Args:
        metadata_df (pd.DataFrame): DataFrame containing 'id', 'noisy_image_path',
                                    and optionally 'clean_image_path'.
        cache_name (str): Identifier for the cache file (e.g., 'train_cache').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (noisy_imgs, clean_imgs)
            - noisy_imgs (dict): {id: np.array}
            - clean_imgs (dict): {id: np.array} (Empty if clean paths not provided)
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(WORKING_DIR, f"{cache_name}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading dataset from cache: {cache_path}")
        try:
            with np.load(cache_path) as data:
                noisy_imgs = {}
                clean_imgs = {}
                for key in data.files:
                    # Keys are prefixed to distinguish noisy vs clean
                    if key.startswith("n_"):
                        noisy_imgs[key[2:]] = data[key]
                    elif key.startswith("c_"):
                        clean_imgs[key[2:]] = data[key]
            return noisy_imgs, clean_imgs
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing dataset...")

    # 2. Compute from scratch if cache missing or failed
    print(f"Loading dataset images from source...")
    noisy_imgs = {}
    clean_imgs = {}

    for _, row in metadata_df.iterrows():
        img_id = str(row["id"])

        # Load Noisy Image
        if "noisy_image_path" in row and pd.notna(row["noisy_image_path"]):
            noisy_imgs[img_id] = load_image(row["noisy_image_path"])

        # Load Clean Image (if available)
        if "clean_image_path" in row and pd.notna(row["clean_image_path"]):
            clean_imgs[img_id] = load_image(row["clean_image_path"])

    # 3. Save to cache
    save_dict = {}
    for k, v in noisy_imgs.items():
        save_dict[f"n_{k}"] = v
    for k, v in clean_imgs.items():
        save_dict[f"c_{k}"] = v

    np.savez_compressed(cache_path, **save_dict)
    print(f"Dataset cached to: {cache_path}")

    return noisy_imgs, clean_imgs


def calculate_rmse(y_true, y_pred):
    """
    Calculates the global Root Mean Squared Error (RMSE) between true and predicted images.
    Supports inputs as lists of arrays or dictionaries of arrays.

    Args:
        y_true: List of np.arrays or Dict {id: np.array}
        y_pred: List of np.arrays or Dict {id: np.array}

    Returns:
        float: The RMSE value.
    """
    # Align dictionaries if provided
    if isinstance(y_true, dict) and isinstance(y_pred, dict):
        common_keys = sorted(list(set(y_true.keys()) & set(y_pred.keys())))
        y_true = [y_true[k] for k in common_keys]
        y_pred = [y_pred[k] for k in common_keys]

    if not y_true:
        return 0.0

    # Flatten all pixels from all images into single arrays for global RMSE
    y_true_flat = np.concatenate([y.flatten() for y in y_true])
    y_pred_flat = np.concatenate([y.flatten() for y in y_pred])

    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    return np.sqrt(mse)


def create_submission(predictions, output_path):
    """
    Formats the predictions into the specific pixel-wise CSV format required for submission.

    Format:
        id,value
        {img_id}_{row}_{col},value

    Args:
        predictions (dict): Dictionary mapping image IDs to predicted numpy arrays.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating submission file...")
    records = []

    # Process each image
    for img_id in sorted(predictions.keys()):
        img = predictions[img_id]
        h, w = img.shape

        # Create coordinate grids (1-based indexing as per sample submission)
        # rows: [[1, 1, ...], [2, 2, ...]]
        # cols: [[1, 2, ...], [1, 2, ...]]
        rows, cols = np.indices((h, w))
        rows = rows + 1
        cols = cols + 1

        # Flatten arrays
        rows_flat = rows.flatten()
        cols_flat = cols.flatten()
        vals_flat = img.flatten()

        # Use pandas for efficient vectorized string operations
        chunk_df = pd.DataFrame({"r": rows_flat, "c": cols_flat, "value": vals_flat})

        # Construct ID column: "imgID_row_col"
        chunk_df["id"] = (
            f"{img_id}_" + chunk_df["r"].astype(str) + "_" + chunk_df["c"].astype(str)
        )

        records.append(chunk_df[["id", "value"]])

    if not records:
        print("Warning: No predictions found to submit.")
        return

    # Concatenate all image records
    full_df = pd.concat(records, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    full_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
