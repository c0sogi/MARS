import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    CACHE_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    DEBUG_DATA_LIMIT,
    ROTATION_ANGLES,
    DO_HORIZONTAL_FLIP,
)
from library.utils import seed_everything


def process_json_to_numpy(json_path, is_test=False):
    """
    Parses the raw JSON file and converts bands to a 3-channel numpy array.
    Returns dictionaries mapping ID to image, angle, and label.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {}
    angles = {}
    labels = {}

    for item in data:
        img_id = item["id"]

        # Extract bands
        band_1 = np.array(item["band_1"]).reshape(75, 75)
        band_2 = np.array(item["band_2"]).reshape(75, 75)

        # Construct 3rd channel: Mean of Band 1 and Band 2
        band_3 = (band_1 + band_2) / 2.0

        # Stack to (3, 75, 75)
        # Using channel-first format for PyTorch
        img = np.stack([band_1, band_2, band_3], axis=0).astype(np.float32)
        images[img_id] = img

        # Extract angle
        # "na" values are handled later during dataset construction or imputation
        angle = item["inc_angle"]
        if angle == "na":
            angles[img_id] = np.nan
        else:
            angles[img_id] = float(angle)

        # Extract label if available
        if not is_test:
            labels[img_id] = int(item["is_iceberg"])

    return images, angles, labels


def get_global_stats(images_array):
    """
    Computes global min and max per channel for the provided images.
    images_array shape: (N, 3, 75, 75)
    Returns: min_vals (3, 1, 1), max_vals (3, 1, 1)
    """
    # Reshape to (Channels, N * H * W) to compute stats per channel
    # Or just use axis=(0, 2, 3)
    min_vals = np.min(images_array, axis=(0, 2, 3)).reshape(3, 1, 1)
    max_vals = np.max(images_array, axis=(0, 2, 3)).reshape(3, 1, 1)
    return min_vals, max_vals


class IcebergDataset(Dataset):
    def __init__(
        self,
        images,
        angles,
        labels=None,
        ids=None,
        global_stats=None,
        transform=False,
        angle_mean=None,
    ):
        """
        Args:
            images: Numpy array of shape (N, 3, 75, 75)
            angles: Numpy array of shape (N,)
            labels: Numpy array of shape (N,) or None
            ids: List or array of IDs
            global_stats: Tuple (min_vals, max_vals) for normalization
            transform: Boolean, whether to apply augmentations
            angle_mean: Float, value to impute NaN angles
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.global_stats = global_stats
        self.transform = transform
        self.angle_mean = angle_mean if angle_mean is not None else 0.0

        # Unpack stats
        if self.global_stats is not None:
            self.min_val, self.max_val = self.global_stats
            # Ensure they are float32
            self.min_val = self.min_val.astype(np.float32)
            self.max_val = self.max_val.astype(np.float32)
            self.denom = self.max_val - self.min_val
            # Avoid division by zero
            self.denom[self.denom == 0] = 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx].copy()  # (3, 75, 75)
        angle = self.angles[idx]

        # Impute missing angle
        if np.isnan(angle):
            angle = self.angle_mean

        # Normalization
        # (img - min) / (max - min)
        # No hard clipping as per instructions
        if self.global_stats is not None:
            img = (img - self.min_val) / self.denom

        # Augmentation
        if self.transform:
            # Horizontal Flip
            if DO_HORIZONTAL_FLIP and np.random.random() < 0.5:
                # Flip along width axis (axis 2)
                img = np.flip(img, axis=2).copy()

            # Discrete Rotation
            if len(ROTATION_ANGLES) > 0:
                # 0, 90, 180, 270 correspond to k=0, 1, 2, 3 for rot90
                # We rotate in the H, W plane (axes 1 and 2)
                k = np.random.randint(0, 4)
                if k > 0:
                    img = np.rot90(img, k, axes=(1, 2)).copy()

        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        else:
            # For test set, return ID as well to help with submission mapping if needed
            # But standard DataLoader collate might struggle with strings mixed with tensors if not careful.
            # We will return ID only if needed, but usually prediction loop handles alignment via order.
            # The prompt implies we need to predict for each ID.
            # We'll return the ID string. Default collate handles list of strings fine.
            return img_tensor, angle_tensor, self.ids[idx]


def load_data(
    load_cached_data=True, batch_size=BATCH_SIZE, debug_limit=DEBUG_DATA_LIMIT
):
    """
    Main function to load and process data.

    Args:
        load_cached_data (bool): Whether to try loading from cache.
        batch_size (int): Batch size for DataLoaders.
        debug_limit (int or None): Limit dataset size for debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    data_loaded = False

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(CACHE_PATH):
        print(f"Loading cached data from {CACHE_PATH}...")
        try:
            cached = np.load(CACHE_PATH, allow_pickle=True)

            train_images = cached["train_images"]
            train_angles = cached["train_angles"]
            train_labels = cached["train_labels"]
            train_ids = cached["train_ids"]

            val_images = cached["val_images"]
            val_angles = cached["val_angles"]
            val_labels = cached["val_labels"]
            val_ids = cached["val_ids"]

            test_images = cached["test_images"]
            test_angles = cached["test_angles"]
            test_ids = cached["test_ids"]

            global_min = cached["global_min"]
            global_max = cached["global_max"]
            angle_mean = float(cached["angle_mean"])

            data_loaded = True
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing data...")
            data_loaded = False

    # 2. Process from Scratch if needed
    if not data_loaded:
        print("Processing raw JSON data...")

        # Load Metadata
        df_train_meta = pd.read_csv(TRAIN_META_PATH)
        df_val_meta = pd.read_csv(VAL_META_PATH)
        df_test_meta = pd.read_csv(TEST_META_PATH)

        # Parse Raw JSON
        # We load all training data (which includes val) from train.json
        raw_train_imgs, raw_train_angles, raw_train_lbls = process_json_to_numpy(
            TRAIN_JSON, is_test=False
        )
        # Load test data
        raw_test_imgs, raw_test_angles, _ = process_json_to_numpy(
            TEST_JSON, is_test=True
        )

        # Helper to assemble arrays based on metadata IDs
        def assemble_split(df, raw_imgs, raw_angles, raw_lbls=None):
            ids = df["id"].values
            N = len(ids)

            images = np.zeros((N, 3, 75, 75), dtype=np.float32)
            angles = np.zeros((N,), dtype=np.float32)
            labels = np.zeros((N,), dtype=np.float32) if raw_lbls is not None else None

            valid_ids = []

            for i, img_id in enumerate(ids):
                if img_id in raw_imgs:
                    images[i] = raw_imgs[img_id]
                    angles[i] = raw_angles[img_id]
                    if raw_lbls is not None:
                        labels[i] = raw_lbls[img_id]
                    valid_ids.append(img_id)
                else:
                    # This should not happen if metadata is correct
                    print(f"Warning: ID {img_id} not found in raw data.")

            return images, angles, labels, np.array(valid_ids)

        # Assemble Splits
        train_images, train_angles, train_labels, train_ids = assemble_split(
            df_train_meta, raw_train_imgs, raw_train_angles, raw_train_lbls
        )
        val_images, val_angles, val_labels, val_ids = assemble_split(
            df_val_meta, raw_train_imgs, raw_train_angles, raw_train_lbls
        )
        test_images, test_angles, _, test_ids = assemble_split(
            df_test_meta, raw_test_imgs, raw_test_angles, None
        )

        # Compute Statistics (Train only)
        print("Computing global statistics from training set...")
        global_min, global_max = get_global_stats(train_images)

        # Compute Angle Mean (Train only, ignoring NaNs)
        angle_mean = np.nanmean(train_angles)

        # Save to Cache
        print(f"Saving processed data to {CACHE_PATH}...")
        np.savez(
            CACHE_PATH,
            train_images=train_images,
            train_angles=train_angles,
            train_labels=train_labels,
            train_ids=train_ids,
            val_images=val_images,
            val_angles=val_angles,
            val_labels=val_labels,
            val_ids=val_ids,
            test_images=test_images,
            test_angles=test_angles,
            test_ids=test_ids,
            global_min=global_min,
            global_max=global_max,
            angle_mean=angle_mean,
        )

    # 3. Apply Debug Limit
    if debug_limit is not None:
        print(f"Debug mode: Limiting data to {debug_limit} samples.")
        train_images = train_images[:debug_limit]
        train_angles = train_angles[:debug_limit]
        train_labels = train_labels[:debug_limit]
        train_ids = train_ids[:debug_limit]

        val_images = val_images[:debug_limit]
        val_angles = val_angles[:debug_limit]
        val_labels = val_labels[:debug_limit]
        val_ids = val_ids[:debug_limit]

        # Keep test set full or limit? Usually limit for full pipeline debug
        test_images = test_images[:debug_limit]
        test_angles = test_angles[:debug_limit]
        test_ids = test_ids[:debug_limit]

    # 4. Create Datasets
    global_stats = (global_min, global_max)

    train_dataset = IcebergDataset(
        train_images,
        train_angles,
        train_labels,
        train_ids,
        global_stats=global_stats,
        transform=True,
        angle_mean=angle_mean,
    )

    val_dataset = IcebergDataset(
        val_images,
        val_angles,
        val_labels,
        val_ids,
        global_stats=global_stats,
        transform=False,
        angle_mean=angle_mean,
    )

    test_dataset = IcebergDataset(
        test_images,
        test_angles,
        labels=None,
        ids=test_ids,
        global_stats=global_stats,
        transform=False,
        angle_mean=angle_mean,
    )

    # 5. Create DataLoaders
    # Use generator for reproducibility in shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_everything,
        generator=g,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_everything,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_everything,
        pin_memory=True,
    )

    print(f"Data loaded successfully.")
    print(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}"
    )
    print(f"Global Stats - Min: {global_min.flatten()}, Max: {global_max.flatten()}")
    print(f"Angle Mean Imputation: {angle_mean}")

    return train_loader, val_loader, test_loader
