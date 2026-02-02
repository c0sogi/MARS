import os
import numpy as np
import pandas as pd
import cv2
import pydicom
import rasterio
import logging
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
from library.config import Config
from library.utils import get_logger, set_seed


def read_dicom_robust(path):
    """
    Attempts to read a DICOM file using multiple libraries (Pydicom, Rasterio, OpenCV, TensorFlow).
    Returns a numpy array or None if all methods fail.
    """
    # Method 0: Pydicom (Specialized for DICOM)
    try:
        ds = pydicom.dcmread(path)
        return ds.pixel_array
    except Exception:
        pass

    # Method 1: Rasterio (GDAL based)
    try:
        with rasterio.open(path) as src:
            img = src.read(1)
            return img
    except Exception:
        pass

    # Method 2: OpenCV
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            # Handle multi-channel if necessary, though DICOM is usually 1-channel
            if len(img.shape) == 3:
                img = img[:, :, 0]
            return img
    except Exception:
        pass

    # Method 3: TensorFlow (IO)
    try:
        # Read file content
        file_content = tf.io.read_file(path)
        # Try decoding as image (sometimes works for standard formats wrapped in DICOM)
        # or use specialized decode if available. Standard tf.io.decode_image might fail on pure DICOM.
        # This is a last resort fallback.
        img_tensor = tf.io.decode_image(
            file_content, channels=1, expand_animations=False
        )
        return img_tensor.numpy().squeeze()
    except Exception:
        pass

    return None


def process_single_image(args):
    """
    Worker function to process a single image.
    Args:
        args: tuple (image_id, src_path, cache_dir, img_size)
    Returns:
        tuple: (image_id, success_flag, mean_val, std_val, new_file_path)
    """
    image_id, src_path, cache_dir, img_size = args
    dest_path = os.path.join(cache_dir, f"{image_id}.png")

    # If file exists, we can skip processing but need to return stats if needed
    # For simplicity in this pipeline, if it exists, we assume it's valid.
    # We will re-read it to calculate stats if it's a training image.
    if os.path.exists(dest_path):
        try:
            img = cv2.imread(dest_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Corrupt cached image")
            mean_val = np.mean(img)
            std_val = np.std(img)
            return image_id, True, mean_val, std_val, dest_path
        except Exception:
            # If corrupt, re-process
            pass

    # Read DICOM
    img = read_dicom_robust(src_path)

    if img is None:
        return image_id, False, 0, 0, None

    # Normalization (Min-Max to 0-255)
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)

    img = img.astype(np.uint8)

    # Resize
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Save
    cv2.imwrite(dest_path, img)

    # Stats
    mean_val = np.mean(img)
    std_val = np.std(img)

    return image_id, True, mean_val, std_val, dest_path


class DicomPreprocessor:
    def __init__(self):
        self.logger = get_logger(os.path.join(Config.LOG_DIR, "preprocess.log"))
        set_seed(Config.SEED)
        self.cache_dir = Config.CACHE_DIR
        self.img_size = Config.IMG_SIZE

        # Determine number of workers (leave some CPUs for system)
        self.num_workers = max(1, os.cpu_count() - 2)

    def _process_subset(self, df, subset_name, calc_stats=False):
        """
        Process a subset of data (train/val/test).
        """
        self.logger.info(f"Processing {subset_name} set...")

        # Get unique images to process
        unique_imgs = df[["image_id", "file_path"]].drop_duplicates()
        tasks = []

        for _, row in unique_imgs.iterrows():
            tasks.append(
                (row["image_id"], row["file_path"], self.cache_dir, self.img_size)
            )

        results = {}
        pixel_means = []
        pixel_stds = []

        # Run parallel processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Map returns results in order, but as_completed allows tracking progress if needed
            # We use map for simplicity or as_completed for logging
            futures = [executor.submit(process_single_image, t) for t in tasks]

            completed_count = 0
            total_count = len(futures)

            for future in as_completed(futures):
                img_id, success, mean_val, std_val, new_path = future.result()
                completed_count += 1

                if completed_count % 1000 == 0:
                    self.logger.info(
                        f"  Processed {completed_count}/{total_count} images."
                    )

                if success:
                    results[img_id] = new_path
                    if calc_stats:
                        pixel_means.append(mean_val)
                        pixel_stds.append(std_val)
                else:
                    self.logger.warning(f"Failed to process image {img_id}")

        # Update DataFrame with new paths
        # We map image_id to new_path. If missing, we might have to drop the row or handle it.
        # Here we filter out rows where image processing failed.

        original_len = len(df)
        df["file_path"] = df["image_id"].map(results)
        df = df.dropna(subset=["file_path"])

        if len(df) < original_len:
            self.logger.warning(
                f"Dropped {original_len - len(df)} rows due to processing failures in {subset_name}."
            )

        stats = {}
        if calc_stats and pixel_means:
            stats["mean"] = np.mean(pixel_means)
            stats["std"] = np.mean(pixel_stds)
            self.logger.info(
                f"  {subset_name} Stats - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}"
            )

        return df, stats

    def run(self, load_cached_data=True):
        """
        Main execution method.
        """
        # Check if parquet files exist
        if (
            load_cached_data
            and os.path.exists(Config.PROCESSED_TRAIN_PKL)
            and os.path.exists(Config.PROCESSED_VAL_PKL)
            and os.path.exists(Config.PROCESSED_TEST_PKL)
        ):

            self.logger.info("Loading cached processed data...")
            train_df = pd.read_parquet(Config.PROCESSED_TRAIN_PKL)
            val_df = pd.read_parquet(Config.PROCESSED_VAL_PKL)
            test_df = pd.read_parquet(Config.PROCESSED_TEST_PKL)
            return train_df, val_df, test_df

        self.logger.info("Starting offline data preprocessing...")

        # Load Metadata
        df_train = pd.read_csv(Config.TRAIN_META)
        df_val = pd.read_csv(Config.VAL_META)
        df_test = pd.read_csv(Config.TEST_META)

        # Process Train (Calculate Stats)
        df_train_proc, train_stats = self._process_subset(
            df_train, "Train", calc_stats=True
        )

        # Process Val
        df_val_proc, _ = self._process_subset(df_val, "Val", calc_stats=False)

        # Process Test
        df_test_proc, _ = self._process_subset(df_test, "Test", calc_stats=False)

        # Save to Parquet
        self.logger.info("Saving processed metadata to Parquet...")
        df_train_proc.to_parquet(Config.PROCESSED_TRAIN_PKL, index=False)
        df_val_proc.to_parquet(Config.PROCESSED_VAL_PKL, index=False)
        df_test_proc.to_parquet(Config.PROCESSED_TEST_PKL, index=False)

        # Save stats to a simple text file for reference or loading
        stats_path = os.path.join(Config.WORKING_DIR, "dataset_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"mean,{train_stats.get('mean', 0)}\n")
            f.write(f"std,{train_stats.get('std', 1)}\n")

        self.logger.info("Preprocessing completed successfully.")
        return df_train_proc, df_val_proc, df_test_proc
