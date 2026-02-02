import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import Config


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the dataset.

    Args:
        mode (str): 'train' or 'test'. If 'train', applies random flips.

    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        return transforms.Compose([])


class IcebergDataset(Dataset):
    """
    PyTorch Dataset for the Iceberg/Ship classification task.
    """

    def __init__(self, images, angles, labels=None, ids=None, transform=None):
        """
        Args:
            images (np.ndarray): Array of images with shape (N, 3, 75, 75).
            angles (np.ndarray): Array of incidence angles with shape (N,).
            labels (np.ndarray, optional): Array of labels (0 or 1) with shape (N,).
            ids (np.ndarray, optional): Array of IDs (strings) with shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load data
        img = self.images[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to tensor
        # Input is float32, keeping it as is.
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return tuple based on availability of labels/ids
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, angle_tensor, label_tensor
        elif self.ids is not None:
            return img_tensor, angle_tensor, self.ids[idx]
        else:
            # Fallback (should not happen in this pipeline)
            return img_tensor, angle_tensor


def prepare_data(load_cached_data=True):
    """
    Prepares the data for training and inference.
    Handles loading from raw JSONs, processing images, imputing angles,
    splitting based on metadata, and caching the results.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' data dictionaries.
              Each inner dict contains 'X', 'y' (or 'ids'), and 'angle'.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache paths
    paths = {
        "X_train": Config.CACHE_X_TRAIN,
        "y_train": Config.CACHE_Y_TRAIN,
        "angle_train": Config.CACHE_ANGLE_TRAIN,
        "X_val": Config.CACHE_X_VAL,
        "y_val": Config.CACHE_Y_VAL,
        "angle_val": Config.CACHE_ANGLE_VAL,
        "X_test": Config.CACHE_X_TEST,
        "id_test": Config.CACHE_ID_TEST,
        "angle_test": Config.CACHE_ANGLE_TEST,
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in paths.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {
            "train": {
                "X": np.load(paths["X_train"]),
                "y": np.load(paths["y_train"]),
                "angle": np.load(paths["angle_train"]),
            },
            "val": {
                "X": np.load(paths["X_val"]),
                "y": np.load(paths["y_val"]),
                "angle": np.load(paths["angle_val"]),
            },
            "test": {
                "X": np.load(paths["X_test"]),
                "ids": np.load(paths["id_test"], allow_pickle=True),
                "angle": np.load(paths["angle_test"]),
            },
        }
        return data

    print("Processing data from scratch...")

    # 1. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)

    # 2. Impute Incidence Angles
    # Compute median from training set only
    train_angle_median = df_train_meta["inc_angle"].median()

    # Fill NaNs
    df_train_meta["inc_angle"] = df_train_meta["inc_angle"].fillna(train_angle_median)
    df_val_meta["inc_angle"] = df_val_meta["inc_angle"].fillna(train_angle_median)
    df_test_meta["inc_angle"] = df_test_meta["inc_angle"].fillna(train_angle_median)

    # 3. Load Raw JSON Data
    # We load into a dictionary for fast lookup by ID
    print("Loading raw JSON files...")

    # Helper to process raw dataframe into id -> bands map
    def load_bands_map(json_path):
        df = pd.read_json(json_path)
        # Create a dictionary: id -> (band_1, band_2)
        # We assume band_1 and band_2 are available
        return df.set_index("id")[["band_1", "band_2"]].to_dict("index")

    train_bands_map = load_bands_map(Config.TRAIN_JSON)
    test_bands_map = load_bands_map(Config.TEST_JSON)

    # Combine maps (train.json contains both train and val splits)
    full_bands_map = {**train_bands_map, **test_bands_map}

    # 4. Process Images and Construct Arrays
    def process_split(df_meta, is_test=False):
        num_samples = len(df_meta)

        # Pre-allocate arrays
        X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
        angles = df_meta["inc_angle"].values.astype(np.float32)

        if is_test:
            ids = df_meta["id"].values
            labels = None
        else:
            ids = None
            labels = df_meta["is_iceberg"].values.astype(np.float32)

        for i, row in enumerate(df_meta.itertuples()):
            img_id = row.id
            bands = full_bands_map[img_id]

            # Reshape bands
            b1 = np.array(bands["band_1"]).reshape(75, 75)
            b2 = np.array(bands["band_2"]).reshape(75, 75)

            # Create 3rd channel: Average
            b3 = (b1 + b2) / 2.0

            # Stack: (3, 75, 75)
            X[i, 0, :, :] = b1
            X[i, 1, :, :] = b2
            X[i, 2, :, :] = b3

        return X, angles, labels, ids

    print("Processing Train Split...")
    X_train, angle_train, y_train, _ = process_split(df_train_meta, is_test=False)

    print("Processing Val Split...")
    X_val, angle_val, y_val, _ = process_split(df_val_meta, is_test=False)

    print("Processing Test Split...")
    X_test, angle_test, _, ids_test = process_split(df_test_meta, is_test=True)

    # 5. Save to Cache
    print("Saving to cache...")
    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["angle_train"], angle_train)

    np.save(paths["X_val"], X_val)
    np.save(paths["y_val"], y_val)
    np.save(paths["angle_val"], angle_val)

    np.save(paths["X_test"], X_test)
    np.save(paths["id_test"], ids_test)
    np.save(paths["angle_test"], angle_test)

    data = {
        "train": {"X": X_train, "y": y_train, "angle": angle_train},
        "val": {"X": X_val, "y": y_val, "angle": angle_val},
        "test": {"X": X_test, "ids": ids_test, "angle": angle_test},
    }

    return data
