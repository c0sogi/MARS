import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import set_seed

# Define cache directory
CACHE_DIR = "./working/idea_55/"


class IcebergDataset(Dataset):
    def __init__(self, X, angles, ids, y=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            ids (np.ndarray): Image IDs of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
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
        # Convert to tensor
        # X is (3, 75, 75), float32
        image = torch.from_numpy(self.X[idx]).float()
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)
        img_id = self.ids[idx]

        if self.transform:
            image = self.transform(image)

        sample = {"image": image, "angle": angle, "id": img_id}

        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.float32)
            sample["label"] = label

        return sample


def get_transforms(mode="train"):
    """
    Returns transforms for data augmentation.
    """
    if mode == "train":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )
    else:
        return None


def _process_json_to_dict(json_path):
    """
    Reads a json file and returns a dictionary mapping id -> item.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def _build_arrays(meta_df, raw_data_map, is_test=False):
    """
    Constructs numpy arrays from metadata and raw data map.
    """
    count = len(meta_df)
    X = np.zeros((count, 3, 75, 75), dtype=np.float32)
    angles = np.full(count, np.nan, dtype=np.float32)
    ids = []
    y = np.zeros(count, dtype=np.float32) if not is_test else None

    for i, row in meta_df.iterrows():
        img_id = row["id"]
        ids.append(img_id)

        item = raw_data_map[img_id]

        # Process Bands
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        avg = (b1 + b2) / 2.0

        # Stack channels: (3, 75, 75)
        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = avg

        # Process Angle
        # Metadata already has inc_angle with NaNs, but let's take from raw if needed or trust metadata.
        # The prompt says raw has "na". Metadata generation script handled coercion.
        # We will use the raw data "na" check to be safe and consistent with "impute with train median" logic.
        ang_val = item["inc_angle"]
        if ang_val == "na":
            angles[i] = np.nan
        else:
            angles[i] = float(ang_val)

        # Process Label
        if not is_test:
            y[i] = row["is_iceberg"]

    return X, angles, np.array(ids), y


def load_and_process_data(load_cached_data=True):
    """
    Loads data from cache or processes from scratch.
    Implements median imputation for incidence angles.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define filenames
    files = {
        "train": ["X_train.npy", "angle_train.npy", "ids_train.npy", "y_train.npy"],
        "val": ["X_val.npy", "angle_val.npy", "ids_val.npy", "y_val.npy"],
        "test": ["X_test.npy", "angle_test.npy", "ids_test.npy"],
    }

    # Check if all files exist
    all_exist = True
    for split, flist in files.items():
        for fname in flist:
            if not os.path.exists(os.path.join(CACHE_DIR, fname)):
                all_exist = False
                break

    if load_cached_data and all_exist:
        print("Loading data from cache...")
        data = {}
        for split, flist in files.items():
            data[split] = {}
            data[split]["X"] = np.load(os.path.join(CACHE_DIR, flist[0]))
            data[split]["angles"] = np.load(os.path.join(CACHE_DIR, flist[1]))
            data[split]["ids"] = np.load(os.path.join(CACHE_DIR, flist[2]))
            if split != "test":
                data[split]["y"] = np.load(os.path.join(CACHE_DIR, flist[3]))
        return data["train"], data["val"], data["test"]

    print("Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv("./metadata/train.csv")
    val_meta = pd.read_csv("./metadata/val.csv")
    test_meta = pd.read_csv("./metadata/test.csv")

    # Load Raw Data
    # Note: train.json covers both train_meta and val_meta
    # test.json covers test_meta
    print("Loading raw JSON files...")
    train_json_map = _process_json_to_dict("./input/train.json")
    test_json_map = _process_json_to_dict("./input/test.json")

    # Build Arrays
    print("Building arrays...")
    X_train, ang_train, ids_train, y_train = _build_arrays(
        train_meta, train_json_map, is_test=False
    )
    X_val, ang_val, ids_val, y_val = _build_arrays(
        val_meta, train_json_map, is_test=False
    )
    X_test, ang_test, ids_test, _ = _build_arrays(
        test_meta, test_json_map, is_test=True
    )

    # Impute Angles
    # Calculate median from TRAIN set only
    train_median_angle = np.nanmedian(ang_train)
    print(f"Imputing missing angles with Train Median: {train_median_angle}")

    # Apply imputation
    ang_train = np.where(np.isnan(ang_train), train_median_angle, ang_train)
    ang_val = np.where(np.isnan(ang_val), train_median_angle, ang_val)
    ang_test = np.where(np.isnan(ang_test), train_median_angle, ang_test)

    # Save to Cache
    print("Saving to cache...")
    np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(CACHE_DIR, "angle_train.npy"), ang_train)
    np.save(os.path.join(CACHE_DIR, "ids_train.npy"), ids_train)
    np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(CACHE_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(CACHE_DIR, "angle_val.npy"), ang_val)
    np.save(os.path.join(CACHE_DIR, "ids_val.npy"), ids_val)
    np.save(os.path.join(CACHE_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(CACHE_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(CACHE_DIR, "angle_test.npy"), ang_test)
    np.save(os.path.join(CACHE_DIR, "ids_test.npy"), ids_test)

    train_data = {"X": X_train, "angles": ang_train, "ids": ids_train, "y": y_train}
    val_data = {"X": X_val, "angles": ang_val, "ids": ids_val, "y": y_val}
    test_data = {"X": X_test, "angles": ang_test, "ids": ids_test}

    return train_data, val_data, test_data


def make_loader(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    set_seed(42)

    train_data, val_data, test_data = load_and_process_data(load_cached_data)

    train_dataset = IcebergDataset(
        train_data["X"],
        train_data["angles"],
        train_data["ids"],
        train_data["y"],
        transform=get_transforms("train"),
    )

    val_dataset = IcebergDataset(
        val_data["X"],
        val_data["angles"],
        val_data["ids"],
        val_data["y"],
        transform=get_transforms("val"),
    )

    test_dataset = IcebergDataset(
        test_data["X"],
        test_data["angles"],
        test_data["ids"],
        y=None,
        transform=get_transforms("test"),
    )

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
