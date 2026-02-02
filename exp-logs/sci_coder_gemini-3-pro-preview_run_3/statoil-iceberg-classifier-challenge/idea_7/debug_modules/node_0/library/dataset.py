import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from library.config import (
    TRAIN_JSON,
    TEST_JSON,
    WORKING_DIR,
    IMAGE_SIZE,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
)


def get_transforms(mode="train"):
    """
    Returns the torchvision transformations for the given mode.

    Args:
        mode (str): 'train', 'val', or 'test'.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    if mode == "train":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )
    else:
        return transforms.Compose([])


def process_and_cache_data(data_type, load_cached_data=True):
    """
    Loads raw JSON data, processes it into numpy arrays, and caches it.

    Args:
        data_type (str): 'train' or 'test'.
        load_cached_data (bool): Whether to try loading from cache first.

    Returns:
        dict: Dictionary containing 'X', 'angle', 'ids', and optionally 'y'.
    """
    # Define cache paths
    cache_files = {
        "X": os.path.join(WORKING_DIR, f"X_{data_type}.npy"),
        "angle": os.path.join(WORKING_DIR, f"angle_{data_type}.npy"),
        "ids": os.path.join(WORKING_DIR, f"ids_{data_type}.npy"),
    }
    if data_type == "train":
        cache_files["y"] = os.path.join(WORKING_DIR, f"y_{data_type}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in cache_files.values())
        if all_exist:
            data = {}
            for k, v in cache_files.items():
                data[k] = np.load(v, allow_pickle=True)
            return data

    # 2. Process from scratch
    json_path = TRAIN_JSON if data_type == "train" else TEST_JSON

    # Ensure working directory exists (redundant with config but safe)
    os.makedirs(WORKING_DIR, exist_ok=True)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    num_samples = len(raw_data)

    # Extract IDs
    ids = np.array([item["id"] for item in raw_data])

    # Extract Bands and reshape to (N, 75, 75)
    # Band 1: HH, Band 2: HV
    band_1 = np.array([item["band_1"] for item in raw_data], dtype=np.float32).reshape(
        num_samples, 75, 75
    )
    band_2 = np.array([item["band_2"] for item in raw_data], dtype=np.float32).reshape(
        num_samples, 75, 75
    )

    # Create 3rd channel: Average of HH and HV
    band_3 = (band_1 + band_2) / 2.0

    # Stack to create (N, 75, 75, 3)
    X = np.stack([band_1, band_2, band_3], axis=-1)

    # Extract Incidence Angle
    # Handle 'na' by converting to NaN
    angles = []
    for item in raw_data:
        ang = item["inc_angle"]
        if ang == "na":
            angles.append(np.nan)
        else:
            angles.append(float(ang))
    angle_arr = np.array(angles, dtype=np.float32)

    # Extract Target if training data
    y_arr = None
    if data_type == "train":
        y_arr = np.array([item["is_iceberg"] for item in raw_data], dtype=np.float32)

    # 3. Save to cache
    np.save(cache_files["X"], X)
    np.save(cache_files["angle"], angle_arr)
    np.save(cache_files["ids"], ids)
    if y_arr is not None:
        np.save(cache_files["y"], y_arr)

    # Construct return dictionary
    data = {"X": X, "angle": angle_arr, "ids": ids}
    if y_arr is not None:
        data["y"] = y_arr

    return data


class IcebergDataset(Dataset):
    def __init__(
        self,
        metadata_df,
        full_data_dict,
        transform=None,
        angle_fill_value=None,
        mode="train",
    ):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata containing indices for this split.
            full_data_dict (dict): Dictionary containing full dataset arrays (X, angle, y, ids).
            transform (callable, optional): Transform to apply to the image.
            angle_fill_value (float, optional): Value to fill NaN incidence angles.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df
        self.transform = transform
        self.mode = mode

        # Select specific samples using original_index from metadata
        # Advanced indexing creates a copy, which is safe for subsequent modification (angle imputation)
        indices = self.metadata["original_index"].values

        self.X = full_data_dict["X"][indices]
        self.angle = full_data_dict["angle"][indices]
        self.ids = full_data_dict["ids"][indices]

        if mode in ["train", "val"]:
            self.y = full_data_dict["y"][indices]
        else:
            self.y = None

        # Handle missing incidence angles
        # We replace NaNs with the provided fill value (usually training set mean)
        if angle_fill_value is not None:
            mask = np.isnan(self.angle)
            if np.any(mask):
                self.angle[mask] = angle_fill_value

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve image: (75, 75, 3)
        img_np = self.X[idx]

        # Convert to Tensor and permute to (C, H, W) -> (3, 75, 75)
        # We do not normalize to 0-1 because data is in dB (float)
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1)

        # Apply transforms (augmentations)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Retrieve angle
        angle_val = self.angle[idx]
        angle_tensor = torch.tensor([angle_val], dtype=torch.float32)

        if self.mode in ["train", "val"]:
            # Return (Input, Angle, Target)
            target = torch.tensor([self.y[idx]], dtype=torch.float32)
            return img_tensor, angle_tensor, target
        else:
            # Return (Input, Angle, ID) for submission generation
            image_id = self.ids[idx]
            return img_tensor, angle_tensor, image_id
