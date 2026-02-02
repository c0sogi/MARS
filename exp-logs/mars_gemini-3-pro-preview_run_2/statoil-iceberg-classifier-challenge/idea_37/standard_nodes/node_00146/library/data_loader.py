import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import INPUT_DIR, METADATA_DIR, CACHE_DIR, BATCH_SIZE, IMAGE_SIZE
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_loader")


class GlobalScaler:
    """
    Applies independent per-channel Min-Max scaling using statistics derived
    from the entire training dataset. Does not clip values, allowing outliers
    in validation/test sets to exceed the [0, 1] range.
    """

    def __init__(self):
        self.mins = None
        self.maxs = None
        self.fitted = False

    def fit(self, images):
        """
        Compute min and max for each channel across the entire dataset.
        Args:
            images (np.ndarray): Shape (N, 3, H, W)
        """
        # Reshape to (N, 3, H*W) to aggregate over spatial dimensions and samples
        flattened = images.reshape(images.shape[0], 3, -1)
        self.mins = flattened.min(axis=(0, 2))  # Shape (3,)
        self.maxs = flattened.max(axis=(0, 2))  # Shape (3,)
        self.fitted = True
        logger.info(f"Global Scaler Fitted. Mins: {self.mins}, Maxs: {self.maxs}")

    def transform(self, images):
        """
        Apply scaling: (x - min) / (max - min)
        Args:
            images (np.ndarray): Shape (N, 3, H, W)
        Returns:
            np.ndarray: Scaled images
        """
        if not self.fitted:
            raise RuntimeError("GlobalScaler must be fitted before transform.")

        # Reshape for broadcasting: (1, 3, 1, 1)
        mins = self.mins.reshape(1, 3, 1, 1)
        maxs = self.maxs.reshape(1, 3, 1, 1)

        # Avoid division by zero if max == min (unlikely in this data but good practice)
        denom = maxs - mins
        denom[denom == 0] = 1.0

        return (images - mins) / denom


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg vs Ship classification.
    Constructs 3-channel images and handles augmentations.
    """

    def __init__(self, images, angles, labels=None, transform=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75), float32
            angles (np.ndarray): Shape (N,), float32
            labels (np.ndarray, optional): Shape (N,), int/float. Defaults to None.
            transform (bool): Whether to apply training augmentations.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        img = self.images[idx]  # (3, 75, 75)
        angle = self.angles[idx]

        # Apply Augmentations (only if transform is True)
        if self.transform:
            # 1. Random Rotation (0, 90, 180, 270 degrees)
            # k is number of times to rotate by 90 degrees
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2))

            # 2. Random Horizontal Flip
            if np.random.random() > 0.5:
                # Axis 2 is Width (C, H, W)
                img = img[:, :, ::-1]

        # Ensure positive strides for PyTorch (fixing negative stride from flip)
        img = np.ascontiguousarray(img)

        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor([angle], dtype=torch.float32)

        if self.labels is not None:
            # Labels are binary 0 or 1, float is needed for BCELoss
            label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            return img_tensor, angle_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, applies global scaling,
    imputes missing incidence angles, and caches the result.
    """
    # Define cache paths
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_npz = os.path.join(CACHE_DIR, "processed_data.npz")
    cache_train_ids = os.path.join(CACHE_DIR, "train_ids.parquet")
    cache_test_ids = os.path.join(CACHE_DIR, "test_ids.parquet")

    # Check if cache exists and should be loaded
    if (
        load_cached_data
        and os.path.exists(cache_npz)
        and os.path.exists(cache_train_ids)
    ):
        logger.info(f"Loading cached processed data from {CACHE_DIR}...")
        try:
            data = np.load(cache_npz)
            train_ids_df = pd.read_parquet(cache_train_ids)
            test_ids_df = pd.read_parquet(cache_test_ids)

            return {
                "train_images": data["train_images"],
                "train_angles": data["train_angles"],
                "train_labels": data["train_labels"],
                "train_ids": train_ids_df["id"].values,
                "test_images": data["test_images"],
                "test_angles": data["test_angles"],
                "test_ids": test_ids_df["id"].values,
            }
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reprocessing from scratch.")

    logger.info("Processing data from scratch...")

    # 1. Load Raw Data
    train_path = os.path.join(INPUT_DIR, "train.json")
    test_path = os.path.join(INPUT_DIR, "test.json")

    # Using pandas to read json is efficient for this structure
    df_train = pd.read_json(train_path)
    df_test = pd.read_json(test_path)

    logger.info(f"Raw Train Shape: {df_train.shape}, Raw Test Shape: {df_test.shape}")

    # 2. Helper to extract images
    def extract_images(df):
        # Stack lists into numpy arrays
        b1 = np.stack([np.array(b) for b in df["band_1"]])
        b2 = np.stack([np.array(b) for b in df["band_2"]])

        # Reshape to (N, 75, 75)
        b1 = b1.reshape(-1, 75, 75)
        b2 = b2.reshape(-1, 75, 75)

        # Compute Band 3: Mean of Band 1 and Band 2
        b3 = (b1 + b2) / 2.0

        # Stack to (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1)
        return images.astype(np.float32)

    logger.info("Constructing 3-channel images...")
    train_images = extract_images(df_train)
    test_images = extract_images(df_test)

    # 3. Global Scaling
    logger.info("Applying Global Scaling...")
    scaler = GlobalScaler()
    scaler.fit(train_images)  # Fit only on training data

    train_images = scaler.transform(train_images)
    test_images = scaler.transform(test_images)

    # 4. Handle Incidence Angles
    logger.info("Processing incidence angles...")
    # Convert 'na' to NaN
    train_angles = pd.to_numeric(df_train["inc_angle"], errors="coerce").values
    test_angles = pd.to_numeric(df_test["inc_angle"], errors="coerce").values

    # Compute mean from valid training angles
    valid_mask = ~np.isnan(train_angles)
    angle_mean = np.mean(train_angles[valid_mask])
    logger.info(f"Imputing missing angles with mean: {angle_mean:.4f}")

    # Impute
    train_angles[np.isnan(train_angles)] = angle_mean
    test_angles[np.isnan(test_angles)] = (
        angle_mean  # Assuming test might have NaNs too, or just for safety
    )

    train_angles = train_angles.astype(np.float32)
    test_angles = test_angles.astype(np.float32)

    # 5. Extract Labels and IDs
    train_labels = df_train["is_iceberg"].values.astype(np.float32)
    train_ids = df_train["id"].values
    test_ids = df_test["id"].values

    # 6. Cache Data
    logger.info("Caching processed data...")
    np.savez(
        cache_npz,
        train_images=train_images,
        train_angles=train_angles,
        train_labels=train_labels,
        test_images=test_images,
        test_angles=test_angles,
    )

    # Save IDs separately using pandas/parquet to avoid pickle issues with strings in npz
    pd.DataFrame({"id": train_ids}).to_parquet(cache_train_ids)
    pd.DataFrame({"id": test_ids}).to_parquet(cache_test_ids)

    return {
        "train_images": train_images,
        "train_angles": train_angles,
        "train_labels": train_labels,
        "train_ids": train_ids,
        "test_images": test_images,
        "test_angles": test_angles,
        "test_ids": test_ids,
    }


def get_data_loaders(load_cached_data=True, batch_size=BATCH_SIZE):
    """
    Generates PyTorch DataLoaders for Train, Validation, and Test sets.
    Uses metadata files to split the processed training data.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        batch_size (int): Batch size for loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Get Processed Data (Cached or Computed)
    data = process_and_cache_data(load_cached_data=load_cached_data)

    # 2. Load Metadata for Splits
    meta_train = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    meta_val = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    meta_test = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 3. Create ID to Index Mapping for Training Data
    # The processed 'train_images' contains all data from train.json.
    # We need to pick specific indices for train and val splits.
    train_id_to_idx = {id_: i for i, id_ in enumerate(data["train_ids"])}

    # Helper to gather arrays based on IDs
    def gather_subset(meta_df, id_map, images, angles, labels=None):
        indices = [id_map[id_] for id_ in meta_df["id"].values]
        sub_images = images[indices]
        sub_angles = angles[indices]
        sub_labels = labels[indices] if labels is not None else None
        return sub_images, sub_angles, sub_labels

    # 4. Construct Subsets
    logger.info("Constructing Train/Val/Test subsets...")

    # Train Split
    X_train, ang_train, y_train = gather_subset(
        meta_train,
        train_id_to_idx,
        data["train_images"],
        data["train_angles"],
        data["train_labels"],
    )

    # Validation Split
    X_val, ang_val, y_val = gather_subset(
        meta_val,
        train_id_to_idx,
        data["train_images"],
        data["train_angles"],
        data["train_labels"],
    )

    # Test Split (Directly from test data, but we ensure order matches metadata)
    test_id_to_idx = {id_: i for i, id_ in enumerate(data["test_ids"])}
    # Note: labels are None for test
    X_test, ang_test, _ = gather_subset(
        meta_test, test_id_to_idx, data["test_images"], data["test_angles"], None
    )

    # 5. Create Datasets
    # Train: Transform=True (Augmentation)
    train_dataset = IcebergDataset(X_train, ang_train, y_train, transform=True)

    # Val: Transform=False
    val_dataset = IcebergDataset(X_val, ang_val, y_val, transform=False)

    # Test: Transform=False
    test_dataset = IcebergDataset(X_test, ang_test, labels=None, transform=False)

    logger.info(
        f"Dataset Sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # 6. Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
