import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_train_median_angle():
    """Calculates the median incidence angle from the training metadata."""
    meta_path = "./metadata/train.csv"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df = pd.read_csv(meta_path)
    # Coerce errors to NaN, then drop NaNs
    angles = pd.to_numeric(df["inc_angle"], errors="coerce").dropna()
    return angles.median()


def process_json_data(json_path, metadata_df, median_angle):
    """
    Parses raw JSON and extracts images, angles, and labels based on metadata.
    """
    # Load raw JSON
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Create a dictionary for O(1) lookup
    data_map = {item["id"]: item for item in raw_data}

    # Lists to store processed data
    images = []
    angles = []
    labels = []
    ids = []

    # Iterate through metadata to preserve order and filter
    for _, row in metadata_df.iterrows():
        img_id = row["id"]
        item = data_map.get(img_id)

        if item is None:
            continue

        # Process Images
        # Band 1 (HH) and Band 2 (HV)
        b1 = np.array(item["band_1"]).reshape(75, 75)
        b2 = np.array(item["band_2"]).reshape(75, 75)
        # Band 3 (Average)
        b3 = (b1 + b2) / 2.0

        # Stack to (75, 75, 3)
        img = np.dstack((b1, b2, b3))
        images.append(img)

        # Process Angle
        ang = item["inc_angle"]
        if ang == "na" or pd.isna(ang):
            ang = median_angle
        else:
            ang = float(ang)
        angles.append(ang)

        # Process Label (if exists)
        if "is_iceberg" in row:
            labels.append(int(row["is_iceberg"]))
        else:
            labels.append(-1)  # Dummy label for test

        ids.append(img_id)

    return (
        np.array(images, dtype=np.float32),
        np.array(angles, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(ids),
    )


def load_data(mode="train", load_cached_data=True):
    """
    Loads data for a specific mode (train/val/test).
    Handles caching to ./working/idea_15/
    """
    CACHE_DIR = "./working/idea_15/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define cache filenames
    cache_X = os.path.join(CACHE_DIR, f"X_{mode}.npy")
    cache_a = os.path.join(CACHE_DIR, f"angles_{mode}.npy")
    cache_y = os.path.join(CACHE_DIR, f"y_{mode}.npy")
    cache_id = os.path.join(CACHE_DIR, f"ids_{mode}.npy")

    # Check if cache exists
    if load_cached_data:
        if (
            os.path.exists(cache_X)
            and os.path.exists(cache_a)
            and os.path.exists(cache_y)
        ):
            print(f"Loading cached {mode} data from {CACHE_DIR}...")
            X = np.load(cache_X)
            a = np.load(cache_a)
            y = np.load(cache_y)
            # IDs are optional for training but good for debugging/submission
            ids = np.load(cache_id) if os.path.exists(cache_id) else None
            return X, a, y, ids

    print(f"Processing {mode} data from scratch...")

    # Get median angle for imputation
    median_angle = get_train_median_angle()

    # Load Metadata
    meta_path = f"./metadata/{mode}.csv"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    df_meta = pd.read_csv(meta_path)

    # Determine source file (train.json or test.json)
    # We assume all rows in a split come from the same source file for simplicity,
    # or we can group. The metadata generation script puts train/val from train.json and test from test.json.
    source_file = df_meta["source_file"].iloc[0]
    json_path = os.path.join("./input", source_file)

    X, a, y, ids = process_json_data(json_path, df_meta, median_angle)

    # Save to cache
    np.save(cache_X, X)
    np.save(cache_a, a)
    np.save(cache_y, y)
    np.save(cache_id, ids)

    return X, a, y, ids


class IcebergDataset(Dataset):
    def __init__(self, images, angles, labels=None, transform=None):
        """
        Args:
            images (np.array): Shape (N, 75, 75, 3)
            angles (np.array): Shape (N,)
            labels (np.array): Shape (N,)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.angles = angles
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image: (75, 75, 3)
        image = self.images[idx]
        angle = self.angles[idx]

        # Convert to Tensor (H, W, C) -> (C, H, W)
        # Data is float32, so we just convert to tensor.
        # Note: ToTensor() scales [0, 255] -> [0, 1] if input is uint8.
        # Here input is float (dB), so we use torch.from_numpy directly.
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3, 75, 75)

        # Apply transforms (Augmentation)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        angle_tensor = torch.tensor(angle, dtype=torch.float32)

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image_tensor, angle_tensor, label_tensor
        else:
            return image_tensor, angle_tensor


def get_transforms(mode="train"):
    """
    Returns torchvision transforms.
    For 'train', applies random flips.
    For 'val'/'test', returns None (or identity).
    """
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        return None
