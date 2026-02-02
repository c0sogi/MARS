import os
import numpy as np
import pandas as pd
import rasterio
import cv2
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from library.config import Config
from library.utils import get_logger

logger = get_logger("dicom_converter")


def _process_single_dicom(args):
    """
    Helper function to process a single DICOM file.
    Reads using rasterio, validates, normalizes, resizes, and saves as PNG.

    Args:
        args (tuple): (src_path, dest_path, img_size)

    Returns:
        tuple: (src_path, success, error_message)
    """
    src_path, dest_path, img_size = args

    # Cache hit check for the image file itself
    if os.path.exists(dest_path):
        return src_path, True, None

    try:
        # Use rasterio to read DICOM
        # Note: rasterio requires GDAL with DICOM driver which is standard in this env
        with rasterio.open(src_path) as src:
            # Read the first band
            img = src.read(1)

        # --- Validation ---
        # Ensure pixel value sum is greater than zero
        if np.sum(img) == 0:
            raise ValueError("Image is empty (pixel sum is 0).")

        # Ensure variance is non-trivial (prevent flat images)
        if np.var(img) < 1.0:
            raise ValueError("Image has trivial variance (flat image).")

        # --- Normalization ---
        # Normalize to 0-255 range
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)

        if max_val - min_val > 0:
            img = (img - min_val) / (max_val - min_val)
        else:
            # This case should be caught by variance check, but as a safeguard
            img = np.zeros_like(img)

        img = (img * 255).astype(np.uint8)

        # --- Resizing ---
        if img_size is not None:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # --- Saving ---
        # Ensure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        cv2.imwrite(dest_path, img)

        return src_path, True, None

    except Exception as e:
        return src_path, False, str(e)


def convert_and_cache_data(load_cached_data=True):
    """
    Main function to preprocess the dataset.

    1. Checks if processed metadata (Parquet) exists.
    2. If not (or if load_cached_data is False), loads original CSVs.
    3. Converts all referenced DICOM files to PNGs in parallel.
    4. Validates every image during conversion.
    5. Updates metadata to point to the new PNG files.
    6. Saves the updated metadata to Parquet files.

    Args:
        load_cached_data (bool): If True, attempts to load processed metadata from disk.

    Returns:
        tuple: (train_df, val_df, test_df) - DataFrames with updated 'file_path' columns.
    """

    # Define cache paths for metadata
    # Using Config.WORKING_DIR which is ./working/idea_2
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "train_processed.parquet")
    val_cache_path = os.path.join(cache_dir, "val_processed.parquet")
    test_cache_path = os.path.join(cache_dir, "test_processed.parquet")

    # --- 1. Try Loading Cached Metadata ---
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            logger.info("Loading processed metadata from cache...")
            try:
                train_df = pd.read_parquet(train_cache_path)
                val_df = pd.read_parquet(val_cache_path)
                test_df = pd.read_parquet(test_cache_path)
                return train_df, val_df, test_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-running preprocessing.")
        else:
            logger.info("Cache not found. Starting preprocessing...")
    else:
        logger.info("Ignoring cache. Starting preprocessing...")

    # --- 2. Load Original Metadata ---
    logger.info("Loading original metadata...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)
    test_df = pd.read_csv(Config.TEST_META_PATH)

    # --- 3. Prepare Conversion Tasks ---
    # Identify unique files across all splits to avoid redundant processing
    all_dfs = [train_df, val_df, test_df]
    unique_files = set()
    for df in all_dfs:
        unique_files.update(df["file_path"].unique())

    logger.info(f"Found {len(unique_files)} unique DICOM files to process.")

    # Create mapping: src_path -> dest_path
    # Images are saved to Config.CACHE_DIR
    tasks = []
    src_to_dest = {}

    for src_path in unique_files:
        # Generate destination path
        # Example: ./input/train/abc.dicom -> ./working/idea_2/cache/abc.png
        filename = os.path.basename(src_path)
        name, _ = os.path.splitext(filename)
        dest_filename = f"{name}.png"
        dest_path = os.path.join(Config.CACHE_DIR, dest_filename)

        src_to_dest[src_path] = dest_path
        tasks.append((src_path, dest_path, Config.IMG_SIZE))

    # --- 4. Execute Conversion in Parallel ---
    logger.info(f"Starting conversion with {Config.NUM_WORKERS} workers...")

    # Ensure image cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    failed_files = []

    with ProcessPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(_process_single_dicom, t) for t in tasks]

        # Iterate over results as they complete
        for future in as_completed(futures):
            src, success, err = future.result()
            if not success:
                failed_files.append((src, err))

    # Check for failures
    if failed_files:
        logger.error(f"Failed to process {len(failed_files)} files.")
        for src, err in failed_files[:5]:
            logger.error(f"  Error processing {src}: {err}")
        if len(failed_files) > 5:
            logger.error(f"  ... and {len(failed_files) - 5} more.")

        # Strict requirement: Raise exception on failure
        raise RuntimeError(
            f"Preprocessing failed for {len(failed_files)} images. Aborting."
        )

    logger.info("All images processed and validated successfully.")

    # --- 5. Update Metadata ---
    logger.info("Updating metadata with new file paths...")

    def update_paths(df, mapping):
        df_copy = df.copy()
        df_copy["file_path"] = df_copy["file_path"].map(mapping)
        return df_copy

    train_df = update_paths(train_df, src_to_dest)
    val_df = update_paths(val_df, src_to_dest)
    test_df = update_paths(test_df, src_to_dest)

    # --- 6. Save Metadata to Cache ---
    logger.info("Saving processed metadata to parquet...")
    train_df.to_parquet(train_cache_path, index=False)
    val_df.to_parquet(val_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    logger.info("Preprocessing complete.")

    return train_df, val_df, test_df
