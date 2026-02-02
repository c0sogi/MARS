import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for Iceberg vs Ship classification.
    Handles 3-channel radar images and scalar incidence angles.
    """

    def __init__(self, images, inc_angles, labels=None, augment=False):
        """
        Args:
            images (np.ndarray): Preprocessed images of shape (N, 75, 75, 3).
            inc_angles (np.ndarray): Incidence angles of shape (N,).
            labels (np.ndarray, optional): Target labels of shape (N,). Defaults to None.
            augment (bool): Whether to apply geometric augmentations.
        """
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]  # Shape: (75, 75, 3)
        inc = self.inc_angles[idx]

        # Convert to Tensor and rearrange dimensions to (C, H, W)
        # Input is (H, W, C), PyTorch expects (C, H, W)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        inc_tensor = torch.tensor([inc], dtype=torch.float32)

        # Apply Geometric Augmentations
        if self.augment:
            # Random 90-degree rotations (k=0, 1, 2, 3)
            k = np.random.randint(0, 4)
            img_tensor = torch.rot90(img_tensor, k, [1, 2])

            # Random Horizontal Flip (Probability 0.5)
            if np.random.random() > 0.5:
                img_tensor = torch.flip(img_tensor, [2])  # Dim 2 is Width

        # Return tuple (image, angle, label) or (image, angle)
        if self.labels is not None:
            label_tensor = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return img_tensor, inc_tensor, label_tensor
        else:
            return img_tensor, inc_tensor


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, performs feature engineering (Band 3 creation),
    imputation, and scaling. Caches the result as an .npz file.

    Returns:
        dict: Dictionary containing 'ids', 'images', 'inc_angles', 'labels'.
    """
    cache_path = Config.PROCESSED_DATA_FILE

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # allow_pickle=True is necessary for loading arrays of strings (ids)
            data = np.load(cache_path, allow_pickle=True)
            return {
                "ids": data["ids"],
                "images": data["images"],
                "inc_angles": data["inc_angles"],
                "labels": data["labels"],
            }
        except Exception as e:
            print(f"Cache load failed: {e}. Reprocessing from scratch...")

    # 2. Process data from scratch
    print("Processing data from scratch...")

    # Load Raw JSONs
    with open(Config.TRAIN_JSON, "r") as f:
        train_raw = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_raw = json.load(f)

    # Helper function to extract and format data from JSON list
    def extract_features(raw_list, is_train=True):
        ids = []
        b1_list = []
        b2_list = []
        inc_list = []
        labels = []

        for item in raw_list:
            ids.append(item["id"])

            # Reshape flattened bands to 75x75
            b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
            b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
            b1_list.append(b1)
            b2_list.append(b2)

            # Handle Incidence Angle
            inc = item["inc_angle"]
            if inc == "na":
                inc_list.append(np.nan)
            else:
                inc_list.append(float(inc))

            # Handle Label
            if is_train:
                labels.append(item["is_iceberg"])
            else:
                labels.append(np.nan)

        return (
            np.array(ids),
            np.stack(b1_list),
            np.stack(b2_list),
            np.array(inc_list, dtype=np.float32),
            np.array(labels, dtype=np.float32),
        )

    # Extract data
    train_ids, train_b1, train_b2, train_inc, train_y = extract_features(
        train_raw, is_train=True
    )
    test_ids, test_b1, test_b2, test_inc, test_y = extract_features(
        test_raw, is_train=False
    )

    # Concatenate train and test for unified processing
    all_ids = np.concatenate([train_ids, test_ids])
    all_b1 = np.concatenate([train_b1, test_b1])
    all_b2 = np.concatenate([train_b2, test_b2])
    all_inc = np.concatenate([train_inc, test_inc])
    all_y = np.concatenate([train_y, test_y])

    # Construct Band 3 (Average of Band 1 and Band 2)
    all_b3 = (all_b1 + all_b2) / 2.0

    # Stack into (N, 75, 75, 3)
    all_images = np.stack([all_b1, all_b2, all_b3], axis=-1)

    # --- Imputation ---
    # Calculate mean incidence angle from training set only
    train_count = len(train_ids)
    inc_mean = np.nanmean(all_inc[:train_count])

    # Fill NaNs in the entire dataset
    nan_mask = np.isnan(all_inc)
    all_inc[nan_mask] = inc_mean

    # --- Independent Per-Channel Min-Max Scaling ---
    # Calculate statistics on training set only
    train_imgs = all_images[:train_count]

    # Compute min/max per channel (axis 0=batch, 1=height, 2=width)
    # Result shape: (3,)
    min_vals = train_imgs.min(axis=(0, 1, 2))
    max_vals = train_imgs.max(axis=(0, 1, 2))

    # Apply scaling to all data: (X - min) / (max - min)
    # Broadcasting handles the shape matching
    all_images = (all_images - min_vals) / (max_vals - min_vals)

    # Cast to float32 for model compatibility
    all_images = all_images.astype(np.float32)

    # --- Cache Results ---
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path, ids=all_ids, images=all_images, inc_angles=all_inc, labels=all_y
    )

    return {
        "ids": all_ids,
        "images": all_images,
        "inc_angles": all_inc,
        "labels": all_y,
    }


def get_loader(
    df_metadata,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    augment=False,
    load_cached_data=True,
):
    """
    Creates a PyTorch DataLoader for the data subset defined in df_metadata.

    Args:
        df_metadata (pd.DataFrame): Dataframe containing at least an 'id' column.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        augment (bool): Whether to apply augmentations.
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1. Load full processed dataset
    data_dict = process_and_cache_data(load_cached_data=load_cached_data)

    # 2. Handle Debug Mode (Subset metadata)
    if Config.DEBUG:
        df_metadata = df_metadata.head(Config.SUBSET_SIZE)

    # 3. Filter data based on metadata IDs
    requested_ids = df_metadata["id"].values

    # Create ID -> Index map for fast lookup
    id_to_idx = {id_str: i for i, id_str in enumerate(data_dict["ids"])}

    # Identify indices corresponding to the requested IDs
    indices = []
    for rid in requested_ids:
        if rid in id_to_idx:
            indices.append(id_to_idx[rid])

    if not indices:
        raise ValueError("No matching IDs found in the processed data.")

    # Extract the subset
    subset_images = data_dict["images"][indices]
    subset_inc = data_dict["inc_angles"][indices]
    subset_labels = data_dict["labels"][indices]

    # 4. Handle Labels
    # If any label in the subset is NaN (test set), we treat the whole subset as unlabeled
    if np.isnan(subset_labels).any():
        final_labels = None
    else:
        final_labels = subset_labels

    # 5. Create Dataset and DataLoader
    dataset = IcebergDataset(subset_images, subset_inc, final_labels, augment=augment)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True
    )

    return loader
