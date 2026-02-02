import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


def process_and_cache_data(load_cached_data=True):
    """
    Loads raw JSON data, processes it (reshape, scale, impute), and caches it to disk.
    Returns the processed numpy arrays.
    """
    cache_path = Config.PROCESSED_DATA_PATH

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["train_images"],
            data["train_inc_angles"],
            data["train_labels"],
            data["train_ids"],
            data["test_images"],
            data["test_inc_angles"],
            data["test_ids"],
        )

    print("Processing data from scratch...")

    # 2. Load Raw Data
    with open(Config.TRAIN_JSON, "r") as f:
        train_data_raw = json.load(f)
    with open(Config.TEST_JSON, "r") as f:
        test_data_raw = json.load(f)

    # 3. Helper function to process bands and metadata
    def extract_features(data_list):
        ids = np.array([item["id"] for item in data_list])

        # Extract bands and reshape to (N, 75, 75)
        b1 = np.array([item["band_1"] for item in data_list]).reshape(-1, 75, 75)
        b2 = np.array([item["band_2"] for item in data_list]).reshape(-1, 75, 75)

        # Create 3rd band (Mean)
        b3 = (b1 + b2) / 2.0

        # Stack channels: (N, 3, 75, 75)
        images = np.stack([b1, b2, b3], axis=1)

        # Extract incidence angles, replacing 'na' with NaN
        inc_angles = []
        for item in data_list:
            val = item["inc_angle"]
            if val == "na":
                inc_angles.append(np.nan)
            else:
                inc_angles.append(float(val))
        inc_angles = np.array(inc_angles)

        return ids, images, inc_angles

    # Process Train and Test
    train_ids, train_images, train_inc_angles = extract_features(train_data_raw)
    test_ids, test_images, test_inc_angles = extract_features(test_data_raw)

    # Extract Labels for Train
    train_labels = np.array([item["is_iceberg"] for item in train_data_raw])

    # 4. Imputation (Incidence Angle)
    # Compute median from training set (ignoring NaNs)
    inc_median = np.nanmedian(train_inc_angles)

    # Fill NaNs in both sets
    train_inc_angles = np.nan_to_num(train_inc_angles, nan=inc_median)
    test_inc_angles = np.nan_to_num(test_inc_angles, nan=inc_median)

    # 5. Min-Max Scaling
    # Compute global min/max per channel from training data
    # Channel 0
    min_c0 = train_images[:, 0, :, :].min()
    max_c0 = train_images[:, 0, :, :].max()
    # Channel 1
    min_c1 = train_images[:, 1, :, :].min()
    max_c1 = train_images[:, 1, :, :].max()
    # Channel 2
    min_c2 = train_images[:, 2, :, :].min()
    max_c2 = train_images[:, 2, :, :].max()

    def apply_scaling(imgs):
        imgs[:, 0, :, :] = (imgs[:, 0, :, :] - min_c0) / (max_c0 - min_c0)
        imgs[:, 1, :, :] = (imgs[:, 1, :, :] - min_c1) / (max_c1 - min_c1)
        imgs[:, 2, :, :] = (imgs[:, 2, :, :] - min_c2) / (max_c2 - min_c2)
        return imgs

    train_images = apply_scaling(train_images)
    test_images = apply_scaling(test_images)

    # 6. Save to Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        train_images=train_images,
        train_inc_angles=train_inc_angles,
        train_labels=train_labels,
        train_ids=train_ids,
        test_images=test_images,
        test_inc_angles=test_inc_angles,
        test_ids=test_ids,
    )

    print(f"Data processed and saved to {cache_path}")
    return (
        train_images,
        train_inc_angles,
        train_labels,
        train_ids,
        test_images,
        test_inc_angles,
        test_ids,
    )


class IcebergDataset(Dataset):
    def __init__(self, images, inc_angles, labels=None, ids=None, train_mode=False):
        """
        Args:
            images (np.ndarray): Shape (N, 3, 75, 75)
            inc_angles (np.ndarray): Shape (N,)
            labels (np.ndarray, optional): Shape (N,)
            ids (np.ndarray, optional): Shape (N,)
            train_mode (bool): Whether to apply augmentation.
        """
        self.images = images
        self.inc_angles = inc_angles
        self.labels = labels
        self.ids = ids
        self.train_mode = train_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get data
        image = self.images[idx]
        inc_angle = self.inc_angles[idx]

        # Convert to Tensor
        # image is (3, 75, 75) float
        image_tensor = torch.from_numpy(image).float()
        inc_angle_tensor = torch.tensor(inc_angle, dtype=torch.float32)

        # Apply Augmentation if in training mode
        if self.train_mode:
            # Random Rotation: 0, 90, 180, 270 degrees
            # k is number of times to rotate by 90 degrees
            k = torch.randint(0, 4, (1,)).item()
            image_tensor = torch.rot90(image_tensor, k, [1, 2])

            # Random Horizontal Flip
            if torch.rand(1).item() > 0.5:
                image_tensor = torch.flip(image_tensor, [2])

        sample = {"image": image_tensor, "inc_angle": inc_angle_tensor}

        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def get_dataloaders(load_cached_data=True, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached .npz files.
        batch_size (int): Batch size for loaders.
        debug (bool): If True, limits dataset size for debugging.
    """
    # 1. Get processed data (all in memory)
    t_imgs, t_incs, t_lbls, t_ids, test_imgs, test_incs, test_ids = (
        process_and_cache_data(load_cached_data)
    )

    # 2. Map IDs to indices for fast lookup
    # t_ids corresponds to the order in the processed arrays
    train_id_to_idx = {id_str: i for i, id_str in enumerate(t_ids)}
    test_id_to_idx = {id_str: i for i, id_str in enumerate(test_ids)}

    # 3. Load Metadata Splits
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # 4. Helper to subset data based on metadata
    def create_subset(df, source_imgs, source_incs, source_lbls, id_map, is_train):
        # Filter IDs that exist in the source data
        valid_ids = [uid for uid in df["id"].values if uid in id_map]
        indices = [id_map[uid] for uid in valid_ids]

        if debug:
            indices = indices[: Config.DEBUG_SUBSET_SIZE]
            valid_ids = valid_ids[: Config.DEBUG_SUBSET_SIZE]

        imgs = source_imgs[indices]
        incs = source_incs[indices]
        ids = np.array(valid_ids)

        lbls = None
        if source_lbls is not None:
            lbls = source_lbls[indices]

        return IcebergDataset(imgs, incs, lbls, ids, train_mode=is_train)

    # 5. Create Datasets
    train_dataset = create_subset(
        df_train, t_imgs, t_incs, t_lbls, train_id_to_idx, is_train=True
    )
    val_dataset = create_subset(
        df_val, t_imgs, t_incs, t_lbls, train_id_to_idx, is_train=False
    )

    # Test dataset
    test_ids_list = [uid for uid in df_test["id"].values if uid in test_id_to_idx]
    test_indices = [test_id_to_idx[uid] for uid in test_ids_list]

    if debug:
        test_indices = test_indices[: Config.DEBUG_SUBSET_SIZE]
        test_ids_list = test_ids_list[: Config.DEBUG_SUBSET_SIZE]

    test_dataset = IcebergDataset(
        test_imgs[test_indices],
        test_incs[test_indices],
        labels=None,
        ids=np.array(test_ids_list),
        train_mode=False,
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
