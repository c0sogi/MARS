import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg/Ship classification task.
    Handles on-the-fly augmentation and tensor conversion.
    """

    def __init__(self, images, inc_angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Input images with shape (N, 3, 75, 75).
            inc_angles (np.ndarray): Incidence angles with shape (N,).
            labels (np.ndarray, optional): Target labels with shape (N,).
            transform (bool): If True, applies random augmentations.
        """
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Copy image to ensure we don't modify the original array during augmentation
        image = self.images[idx].copy()  # Shape: (3, 75, 75)
        inc_angle = self.inc_angles[idx]

        if self.transform:
            # Rotational Invariance: Random rotation of 0, 90, 180, or 270 degrees
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(1, 2))

            # Horizontal Flip (Random)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2)  # Axis 2 is Width

        # Convert to PyTorch tensors
        # Image: (3, 75, 75) float32
        image_tensor = torch.from_numpy(image).float()
        # Inc Angle: scalar float32
        inc_angle_tensor = torch.tensor(inc_angle, dtype=torch.float32)

        if self.labels is not None:
            # Label: scalar float32 (for BCELoss)
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, inc_angle_tensor, label_tensor
        else:
            return image_tensor, inc_angle_tensor


def load_json_data(file_path):
    """Helper to load JSON data into a DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def process_raw_images(df):
    """
    Extracts bands from DataFrame, reshapes them, and creates the 3rd channel (Mean).
    Returns:
        images (np.ndarray): (N, 3, 75, 75)
        ids (np.ndarray): (N,)
    """
    # Convert lists to numpy arrays and reshape to (N, 75, 75)
    # Using np.array(list) is efficient here
    band_1 = np.stack([np.array(b) for b in df["band_1"]]).reshape(-1, 75, 75)
    band_2 = np.stack([np.array(b) for b in df["band_2"]]).reshape(-1, 75, 75)

    # Construct Channel 3: Arithmetic Mean of Band 1 and Band 2
    band_3 = (band_1 + band_2) / 2.0

    # Stack channels: (N, 3, 75, 75)
    images = np.stack([band_1, band_2, band_3], axis=1)
    ids = df["id"].values

    return images, ids


def load_data(load_cached_data=True):
    """
    Main data loading function.
    1. Checks for cached .npz file.
    2. If not found, loads raw JSONs and Metadata CSVs.
    3. Processes images (Reshape, Channel 3 creation).
    4. Aligns data with Train/Val/Test splits defined in metadata.
    5. Computes global stats for Band 3 from Training set.
    6. Applies Global Min-Max normalization.
    7. Imputes missing incidence angles.
    8. Caches the processed data.
    9. Returns numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(Config.CACHE_PATH):
        try:
            print(f"Loading cached processed data from {Config.CACHE_PATH}")
            data = np.load(Config.CACHE_PATH)

            # Unpack
            X_train, y_train, inc_train = (
                data["X_train"],
                data["y_train"],
                data["inc_train"],
            )
            X_val, y_val, inc_val = data["X_val"], data["y_val"], data["inc_val"]
            X_test, inc_test, ids_test = (
                data["X_test"],
                data["inc_test"],
                data["ids_test"],
            )

            # Handle Debug Slicing (Post-Cache Load)
            if Config.DEBUG:
                print(
                    f"DEBUG Mode: Slicing data to {Config.MAX_DEBUG_SAMPLES} samples."
                )
                X_train, y_train, inc_train = (
                    X_train[: Config.MAX_DEBUG_SAMPLES],
                    y_train[: Config.MAX_DEBUG_SAMPLES],
                    inc_train[: Config.MAX_DEBUG_SAMPLES],
                )
                X_val, y_val, inc_val = (
                    X_val[: Config.MAX_DEBUG_SAMPLES],
                    y_val[: Config.MAX_DEBUG_SAMPLES],
                    inc_val[: Config.MAX_DEBUG_SAMPLES],
                )
                X_test, inc_test, ids_test = (
                    X_test[: Config.MAX_DEBUG_SAMPLES],
                    inc_test[: Config.MAX_DEBUG_SAMPLES],
                    ids_test[: Config.MAX_DEBUG_SAMPLES],
                )

            return (
                X_train,
                y_train,
                inc_train,
                X_val,
                y_val,
                inc_val,
                X_test,
                inc_test,
                ids_test,
            )

        except Exception as e:
            print(f"Failed to load cache ({e}). Recomputing from scratch...")

    # 2. Process from Scratch
    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    test_meta = pd.read_csv(Config.TEST_META_PATH)

    # Load Raw JSONs
    print("Loading raw JSON files...")
    df_train_raw = load_json_data(Config.TRAIN_JSON)
    df_test_raw = load_json_data(Config.TEST_JSON)

    # Process Images (Raw dB values)
    # Note: df_train_raw contains both training and validation samples
    print("Constructing image channels...")
    train_imgs_raw, train_ids_raw = process_raw_images(df_train_raw)
    test_imgs_raw, test_ids_raw = process_raw_images(df_test_raw)

    # Create ID -> Index Mappings
    train_id_map = {id_: i for i, id_ in enumerate(train_ids_raw)}
    test_id_map = {id_: i for i, id_ in enumerate(test_ids_raw)}

    # Helper to extract subsets based on metadata IDs
    def get_subset(meta_df, id_map, img_source):
        indices = [id_map[id_] for id_ in meta_df["id"].values]
        return img_source[indices]

    # Extract Raw Subsets
    X_train_raw = get_subset(train_meta, train_id_map, train_imgs_raw)
    X_val_raw = get_subset(val_meta, train_id_map, train_imgs_raw)
    X_test_raw = get_subset(test_meta, test_id_map, test_imgs_raw)

    # 3. Normalization
    # Calculate Band 3 Statistics from Training Set ONLY
    print("Computing normalization statistics...")
    b3_train = X_train_raw[:, 2, :, :]
    b3_min = b3_train.min()
    b3_max = b3_train.max()

    # Define Global Min-Max Scaling Function
    def normalize(X):
        X_norm = X.copy()
        # Band 1
        X_norm[:, 0] = (X[:, 0] - Config.BAND1_MIN) / (
            Config.BAND1_MAX - Config.BAND1_MIN
        )
        # Band 2
        X_norm[:, 1] = (X[:, 1] - Config.BAND2_MIN) / (
            Config.BAND2_MAX - Config.BAND2_MIN
        )
        # Band 3 (Computed stats)
        X_norm[:, 2] = (X[:, 2] - b3_min) / (b3_max - b3_min)
        return X_norm

    print("Applying Global Normalization...")
    X_train = normalize(X_train_raw)
    X_val = normalize(X_val_raw)
    X_test = normalize(X_test_raw)

    # 4. Handle Incidence Angles
    # Use metadata values (NaNs represent missing 'na' values)
    train_inc = train_meta["inc_angle"].values
    val_inc = val_meta["inc_angle"].values
    test_inc = test_meta["inc_angle"].values

    # Impute NaNs with Mean of Training Set
    inc_mean = np.nanmean(train_inc)
    train_inc = np.nan_to_num(train_inc, nan=inc_mean)
    val_inc = np.nan_to_num(val_inc, nan=inc_mean)
    test_inc = np.nan_to_num(test_inc, nan=inc_mean)

    # 5. Targets and IDs
    y_train = train_meta["is_iceberg"].values
    y_val = val_meta["is_iceberg"].values
    ids_test = test_meta["id"].values

    # 6. Save to Cache
    print(f"Saving processed data to {Config.CACHE_PATH}...")
    np.savez(
        Config.CACHE_PATH,
        X_train=X_train,
        y_train=y_train,
        inc_train=train_inc,
        X_val=X_val,
        y_val=y_val,
        inc_val=val_inc,
        X_test=X_test,
        inc_test=test_inc,
        ids_test=ids_test,
    )

    # 7. Handle Debug Slicing (Post-Processing)
    if Config.DEBUG:
        print(f"DEBUG Mode: Slicing data to {Config.MAX_DEBUG_SAMPLES} samples.")
        X_train, y_train, inc_train = (
            X_train[: Config.MAX_DEBUG_SAMPLES],
            y_train[: Config.MAX_DEBUG_SAMPLES],
            inc_train[: Config.MAX_DEBUG_SAMPLES],
        )
        X_val, y_val, inc_val = (
            X_val[: Config.MAX_DEBUG_SAMPLES],
            y_val[: Config.MAX_DEBUG_SAMPLES],
            inc_val[: Config.MAX_DEBUG_SAMPLES],
        )
        X_test, inc_test, ids_test = (
            X_test[: Config.MAX_DEBUG_SAMPLES],
            inc_test[: Config.MAX_DEBUG_SAMPLES],
            ids_test[: Config.MAX_DEBUG_SAMPLES],
        )

    return (
        X_train,
        y_train,
        inc_train,
        X_val,
        y_val,
        inc_val,
        X_test,
        inc_test,
        ids_test,
    )
