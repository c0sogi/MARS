import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_logger

# Initialize Logger
logger = get_logger("DataProcessing")


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg/Ship classification.
    Handles on-the-fly normalization and augmentation.
    """

    def __init__(self, images, angles, labels=None, stats=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 75, 75, 3)
            angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,). Defaults to None.
            stats (dict): Dictionary containing 'min' and 'max' arrays for normalization.
            transform (bool): Whether to apply data augmentation.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.stats = stats
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Retrieve image (H, W, C)
        image = self.images[idx].copy()
        angle = self.angles[idx]

        # 1. Normalization (Global Min-Max Scaling)
        # Formula: (x - min) / (max - min)
        # We do not clip values, allowing outliers > 1.0 as per instructions.
        if self.stats is not None:
            min_vals = self.stats["min"]
            max_vals = self.stats["max"]
            # Broadcasting: (75, 75, 3) - (3,)
            image = (image - min_vals) / (max_vals - min_vals + 1e-8)

        # 2. Augmentation
        if self.transform:
            # Random Horizontal Flip
            if np.random.rand() < 0.5:
                image = np.fliplr(image)

            # Random Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            image = np.rot90(image, k=k)
            # Note: np.rot90 rotates first two axes, which are H, W.
            # Since image is (H, W, C), this is correct.

        # 3. To Tensor
        # Convert (H, W, C) -> (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float()

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            # Return -1 or similar for test set if no labels
            return image_tensor, angle_tensor, torch.tensor(-1.0)


def _process_json_to_dict(json_path):
    """Helper to load JSON and convert to a dict mapped by ID."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _construct_images(data_list):
    """
    Constructs 3-channel images from raw data list.
    Channels: Band 1, Band 2, Mean(Band 1, Band 2)
    Returns: np.ndarray of shape (N, 75, 75, 3)
    """
    images = []
    for item in data_list:
        # Band 1 and Band 2 are flattened lists of 5625 elements
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)

        # Channel 3: Arithmetic Mean
        b3 = (b1 + b2) / 2.0

        # Stack along the last axis -> (75, 75, 3)
        img = np.dstack((b1, b2, b3))
        images.append(img)

    return np.array(images, dtype=np.float32)


def load_and_process_data(load_cached_data=True):
    """
    Main function to load, process, and split data.
    Implements caching mechanism.
    """
    cache_path = Config.CACHE_PATH
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)

            # Reconstruct dictionaries
            train_data = {
                "images": data["X_train"],
                "angles": data["angle_train"],
                "labels": data["y_train"],
            }
            val_data = {
                "images": data["X_val"],
                "angles": data["angle_val"],
                "labels": data["y_val"],
            }
            test_data = {
                "images": data["X_test"],
                "angles": data["angle_test"],
                "ids": data["ids_test"],
            }
            global_stats = data["global_stats"].item()

            # Handle Debug Mode
            if Config.DEBUG:
                logger.info(
                    f"DEBUG mode: Truncating datasets to {Config.DEBUG_SIZE} samples."
                )
                for d in [train_data, val_data]:
                    d["images"] = d["images"][: Config.DEBUG_SIZE]
                    d["angles"] = d["angles"][: Config.DEBUG_SIZE]
                    d["labels"] = d["labels"][: Config.DEBUG_SIZE]
                test_data["images"] = test_data["images"][: Config.DEBUG_SIZE]
                test_data["angles"] = test_data["angles"][: Config.DEBUG_SIZE]
                test_data["ids"] = test_data["ids"][: Config.DEBUG_SIZE]

            return train_data, val_data, test_data, global_stats

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing data...")

    # 2. Process from Scratch
    logger.info("Processing data from scratch...")

    # Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    df_val_meta = pd.read_csv(Config.VAL_META)
    df_test_meta = pd.read_csv(Config.TEST_META)

    # Load Raw JSONs
    logger.info("Loading raw JSON files...")
    raw_train = _process_json_to_dict(Config.TRAIN_JSON)
    raw_test = _process_json_to_dict(Config.TEST_JSON)

    # Helper to extract data based on metadata DataFrame
    def extract_data(df, raw_dict, is_test=False):
        data_list = []
        angles = []
        labels = []
        ids = []

        for _, row in df.iterrows():
            img_id = row["id"]
            item = raw_dict[img_id]
            data_list.append(item)
            ids.append(img_id)

            # Handle Incidence Angle
            # Metadata CSV has 'inc_angle', raw JSON has it too.
            # We use the one from metadata which has already parsed 'na' to NaN (or we handle it here).
            # The metadata CSV stores numeric values, NaNs are empty or nan.
            ang = row["inc_angle"]
            angles.append(ang)

            if not is_test:
                labels.append(row["is_iceberg"])

        images = _construct_images(data_list)
        return (
            images,
            np.array(angles, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(ids),
        )

    # Extract Splits
    logger.info("Constructing image arrays...")
    X_train, angle_train, y_train, _ = extract_data(
        df_train_meta, raw_train, is_test=False
    )
    X_val, angle_val, y_val, _ = extract_data(df_val_meta, raw_train, is_test=False)
    X_test, angle_test, _, ids_test = extract_data(df_test_meta, raw_test, is_test=True)

    # Impute Missing Incidence Angles
    # Calculate mean from training set (ignoring NaNs)
    angle_mean = np.nanmean(angle_train)
    logger.info(f"Imputing missing incidence angles with mean: {angle_mean:.4f}")

    # Fill NaNs
    angle_train = np.nan_to_num(angle_train, nan=angle_mean)
    angle_val = np.nan_to_num(angle_val, nan=angle_mean)
    angle_test = np.nan_to_num(angle_test, nan=angle_mean)

    # Compute Global Statistics for Normalization
    # Using Train + Val (Full Training Data)
    logger.info("Computing global statistics...")
    X_full_train = np.concatenate([X_train, X_val], axis=0)

    # Compute min/max per channel (axis 0, 1, 2 are N, H, W. Channel is axis 3)
    # Actually shape is (N, 75, 75, 3). We want stats per channel.
    # Reshape to (N*H*W, 3)
    flat_pixels = X_full_train.reshape(-1, 3)
    min_vals = np.min(flat_pixels, axis=0)
    max_vals = np.max(flat_pixels, axis=0)

    global_stats = {"min": min_vals, "max": max_vals}
    logger.info(f"Global Min: {min_vals}, Global Max: {max_vals}")

    # Cache Data
    logger.info(f"Saving processed data to {cache_path}")
    np.savez(
        cache_path,
        X_train=X_train,
        angle_train=angle_train,
        y_train=y_train,
        X_val=X_val,
        angle_val=angle_val,
        y_val=y_val,
        X_test=X_test,
        angle_test=angle_test,
        ids_test=ids_test,
        global_stats=global_stats,
    )

    # Prepare Return Structures
    train_data = {"images": X_train, "angles": angle_train, "labels": y_train}
    val_data = {"images": X_val, "angles": angle_val, "labels": y_val}
    test_data = {"images": X_test, "angles": angle_test, "ids": ids_test}

    # Handle Debug Mode (Post-processing check)
    if Config.DEBUG:
        logger.info(f"DEBUG mode: Truncating datasets to {Config.DEBUG_SIZE} samples.")
        for d in [train_data, val_data]:
            d["images"] = d["images"][: Config.DEBUG_SIZE]
            d["angles"] = d["angles"][: Config.DEBUG_SIZE]
            d["labels"] = d["labels"][: Config.DEBUG_SIZE]
        test_data["images"] = test_data["images"][: Config.DEBUG_SIZE]
        test_data["angles"] = test_data["angles"][: Config.DEBUG_SIZE]
        test_data["ids"] = test_data["ids"][: Config.DEBUG_SIZE]

    return train_data, val_data, test_data, global_stats
