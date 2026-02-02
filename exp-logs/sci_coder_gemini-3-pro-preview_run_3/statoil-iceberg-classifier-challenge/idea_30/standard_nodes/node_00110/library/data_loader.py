import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed


class IcebergDataset(Dataset):
    def __init__(self, X, angles, ids, y=None, transform=None):
        """
        Custom Dataset for Iceberg/Ship classification.

        Args:
            X (np.ndarray): Image data of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            ids (np.ndarray): Image IDs.
            y (np.ndarray, optional): Labels of shape (N,). Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.angles = angles
        self.ids = ids
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve data
        img = self.X[idx]  # Shape: (3, 75, 75)
        angle = self.angles[idx]

        # Convert to Tensor
        img_tensor = torch.from_numpy(img).float()
        angle_tensor = torch.tensor(angle).float()

        # Apply transforms (e.g., augmentations)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return data
        if self.y is not None:
            label = torch.tensor(self.y[idx]).float()
            # Model expects: forward(x_img, x_angle) -> loss(pred, label)
            return (img_tensor, angle_tensor), label
        else:
            return (img_tensor, angle_tensor)


def process_split(metadata_path, raw_data_dict, median_angle, is_test=False):
    """
    Helper function to process raw data based on metadata indices.
    """
    df_meta = pd.read_csv(metadata_path)
    ids = df_meta["id"].values

    X_list = []
    angles_list = []
    y_list = []

    for img_id in ids:
        item = raw_data_dict.get(img_id)
        if item is None:
            raise ValueError(f"ID {img_id} found in metadata but not in raw JSON.")

        # Process Bands
        # Band 1: HH, Band 2: HV
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        # Band 3: Average of HH and HV
        b3 = (b1 + b2) / 2.0

        # Stack to create (3, 75, 75) image
        img = np.stack([b1, b2, b3], axis=0)
        X_list.append(img)

        # Process Incidence Angle
        ang = item["inc_angle"]
        if ang == "na" or pd.isna(ang):
            ang = median_angle
        else:
            ang = float(ang)
        angles_list.append(ang)

        # Process Label
        if not is_test:
            y_list.append(item["is_iceberg"])

    X = np.array(X_list, dtype=np.float32)
    angles = np.array(angles_list, dtype=np.float32)
    ids_arr = np.array(ids)

    if not is_test:
        y = np.array(y_list, dtype=np.float32)
        return X, angles, ids_arr, y
    else:
        return X, angles, ids_arr, None


def get_loaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.
    Handles caching, imputation, and augmentation.
    """
    set_seed(42)

    CACHE_DIR = "./working/idea_30/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_files = {
        "train": ["X_train.npy", "angles_train.npy", "y_train.npy", "ids_train.npy"],
        "val": ["X_val.npy", "angles_val.npy", "y_val.npy", "ids_val.npy"],
        "test": ["X_test.npy", "angles_test.npy", "ids_test.npy"],
    }

    # Check if cache exists
    all_cached = True
    if load_cached_data:
        for split, files in cache_files.items():
            for f in files:
                if not os.path.exists(os.path.join(CACHE_DIR, f)):
                    all_cached = False
                    break
    else:
        all_cached = False

    if all_cached:
        print(f"Loading cached data from {CACHE_DIR}...")
        # Load Train
        X_train = np.load(os.path.join(CACHE_DIR, "X_train.npy"))
        angles_train = np.load(os.path.join(CACHE_DIR, "angles_train.npy"))
        y_train = np.load(os.path.join(CACHE_DIR, "y_train.npy"))
        ids_train = np.load(os.path.join(CACHE_DIR, "ids_train.npy"), allow_pickle=True)

        # Load Val
        X_val = np.load(os.path.join(CACHE_DIR, "X_val.npy"))
        angles_val = np.load(os.path.join(CACHE_DIR, "angles_val.npy"))
        y_val = np.load(os.path.join(CACHE_DIR, "y_val.npy"))
        ids_val = np.load(os.path.join(CACHE_DIR, "ids_val.npy"), allow_pickle=True)

        # Load Test
        X_test = np.load(os.path.join(CACHE_DIR, "X_test.npy"))
        angles_test = np.load(os.path.join(CACHE_DIR, "angles_test.npy"))
        ids_test = np.load(os.path.join(CACHE_DIR, "ids_test.npy"), allow_pickle=True)

    else:
        print("Processing data from scratch...")

        # 1. Determine Median Angle from Training Metadata
        df_train_meta = pd.read_csv("./metadata/train.csv")
        train_angles_numeric = pd.to_numeric(
            df_train_meta["inc_angle"], errors="coerce"
        )
        median_angle = train_angles_numeric.median()

        # Handle edge case if median is NaN (though unlikely)
        if pd.isna(median_angle):
            median_angle = 0.0

        # 2. Load Raw JSON Data
        # Loading train.json (contains both train and val samples)
        with open("./input/train.json", "r") as f:
            train_json_data = json.load(f)
        # Create map for O(1) access
        train_data_map = {item["id"]: item for item in train_json_data}

        # Loading test.json
        with open("./input/test.json", "r") as f:
            test_json_data = json.load(f)
        test_data_map = {item["id"]: item for item in test_json_data}

        # 3. Process Splits
        print("Processing Train Split...")
        X_train, angles_train, ids_train, y_train = process_split(
            "./metadata/train.csv", train_data_map, median_angle, is_test=False
        )

        print("Processing Val Split...")
        X_val, angles_val, ids_val, y_val = process_split(
            "./metadata/val.csv", train_data_map, median_angle, is_test=False
        )

        print("Processing Test Split...")
        X_test, angles_test, ids_test, _ = process_split(
            "./metadata/test.csv", test_data_map, median_angle, is_test=True
        )

        # 4. Save to Cache
        print("Saving to cache...")
        np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(CACHE_DIR, "angles_train.npy"), angles_train)
        np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(CACHE_DIR, "ids_train.npy"), ids_train)

        np.save(os.path.join(CACHE_DIR, "X_val.npy"), X_val)
        np.save(os.path.join(CACHE_DIR, "angles_val.npy"), angles_val)
        np.save(os.path.join(CACHE_DIR, "y_val.npy"), y_val)
        np.save(os.path.join(CACHE_DIR, "ids_val.npy"), ids_val)

        np.save(os.path.join(CACHE_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(CACHE_DIR, "angles_test.npy"), angles_test)
        np.save(os.path.join(CACHE_DIR, "ids_test.npy"), ids_test)

    # Define Transforms
    # Augmentation only for training
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    )

    # Create Datasets
    train_dataset = IcebergDataset(
        X_train, angles_train, ids_train, y_train, transform=train_transform
    )
    val_dataset = IcebergDataset(X_val, angles_val, ids_val, y_val, transform=None)
    test_dataset = IcebergDataset(X_test, angles_test, ids_test, y=None, transform=None)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
