import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.utils import seed_everything

# Constants
CACHE_DIR = "./working/idea_5/"
INPUT_DIR = "./input/"
METADATA_DIR = "./metadata/"


class IcebergDataset(Dataset):
    def __init__(self, X, angles, y=None, ids=None, transform=None):
        """
        Args:
            X (np.ndarray): Images of shape (N, 3, 75, 75).
            angles (np.ndarray): Incidence angles of shape (N,).
            y (np.ndarray, optional): Labels of shape (N,).
            ids (np.ndarray, optional): IDs of shape (N,).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = torch.from_numpy(X).float()
        self.angles = torch.from_numpy(angles).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is (C, H, W)
        img = self.X[idx]
        angle = self.angles[idx]

        if self.transform:
            img = self.transform(img)

        sample = {
            "image": img,
            "angle": angle,
        }

        if self.y is not None:
            sample["label"] = self.y[idx]

        if self.ids is not None:
            sample["id"] = self.ids[idx]

        return sample


def process_subset(
    mode, metadata_df, raw_data_dict, cache_dir, load_cached_data, angle_mean_fill
):
    """
    Process a specific subset (train/val/test), caching the results.

    Args:
        mode (str): 'train', 'val', or 'test'.
        metadata_df (pd.DataFrame): Metadata for this subset.
        raw_data_dict (dict): Dictionary mapping source_file to raw json data list.
                              Only needed if cache miss.
        cache_dir (str): Directory to save/load cache.
        load_cached_data (bool): Whether to attempt loading from cache.
        angle_mean_fill (float): Value to fill NaN angles with.

    Returns:
        tuple: (X, angles, y, ids)
    """
    os.makedirs(cache_dir, exist_ok=True)

    # File paths for cache
    f_X = os.path.join(cache_dir, f"X_{mode}.npy")
    f_angle = os.path.join(cache_dir, f"angle_{mode}.npy")
    f_y = os.path.join(cache_dir, f"y_{mode}.npy")
    f_ids = os.path.join(cache_dir, f"ids_{mode}.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if os.path.exists(f_X) and os.path.exists(f_angle) and os.path.exists(f_ids):
            # Check y existence only if not test
            if mode == "test" or os.path.exists(f_y):
                print(f"Loading {mode} data from cache...")
                X = np.load(f_X)
                angles = np.load(f_angle)
                ids = np.load(f_ids)
                y = np.load(f_y) if mode != "test" else None
                return X, angles, y, ids

    print(f"Processing {mode} data from scratch...")

    # 2. Process from raw data
    # We need to reconstruct the dataset based on metadata
    # Metadata contains: id, inc_angle, is_iceberg (optional), source_file, original_index

    num_samples = len(metadata_df)
    # Shape: (N, 3, 75, 75)
    X = np.zeros((num_samples, 3, 75, 75), dtype=np.float32)
    ids = []
    y = np.zeros(num_samples, dtype=np.float32) if mode != "test" else None

    # We will fill angles from metadata first, then impute
    raw_angles = metadata_df["inc_angle"].values.astype(np.float32)

    # Iterate and fill
    # Group by source file to minimize switching (though usually just 1 file per split)
    for source_file, group in metadata_df.groupby("source_file"):
        if source_file not in raw_data_dict:
            # Load raw json if not provided (should be provided in main flow usually)
            # But here we assume raw_data_dict is populated by caller if cache miss is expected
            raise ValueError(
                f"Raw data for {source_file} not provided in raw_data_dict"
            )

        raw_list = raw_data_dict[source_file]

        # Create a map for faster lookup if needed, but original_index is direct
        # raw_list is a list of dicts

        for i, (idx, row) in enumerate(group.iterrows()):
            # original_index points to the index in the raw json list
            orig_idx = row["original_index"]
            item = raw_list[orig_idx]

            # Verify ID match
            if item["id"] != row["id"]:
                raise ValueError(
                    f"ID mismatch at index {orig_idx}: meta={row['id']}, raw={item['id']}"
                )

            # Process Bands
            band_1 = np.array(item["band_1"]).reshape(75, 75)
            band_2 = np.array(item["band_2"]).reshape(75, 75)
            band_3 = (band_1 + band_2) / 2.0

            # Stack into (3, 75, 75)
            # Row index in the output array depends on the dataframe index (reset or not?)
            # We use `i` from enumerate, but we must map it to the correct position in X
            # Since we iterate by group, this is tricky.
            # Better approach: Iterate the dataframe directly and look up in raw_list.
            pass

    # Re-implement loop for correctness
    # Ensure raw_data_dict has the data
    for i in range(num_samples):
        row = metadata_df.iloc[i]
        src = row["source_file"]
        orig_idx = row["original_index"]

        item = raw_data_dict[src][orig_idx]

        # Process Bands
        b1 = np.array(item["band_1"], dtype=np.float32).reshape(75, 75)
        b2 = np.array(item["band_2"], dtype=np.float32).reshape(75, 75)
        b3 = (b1 + b2) / 2.0

        X[i, 0, :, :] = b1
        X[i, 1, :, :] = b2
        X[i, 2, :, :] = b3

        ids.append(row["id"])

        if mode != "test":
            y[i] = row["is_iceberg"]

    ids = np.array(ids)

    # Impute angles
    # raw_angles has NaNs where 'na' was present
    # Replace NaNs with angle_mean_fill
    angles = np.where(np.isnan(raw_angles), angle_mean_fill, raw_angles)

    # 3. Save to cache
    np.save(f_X, X)
    np.save(f_angle, angles)
    np.save(f_ids, ids)
    if mode != "test":
        np.save(f_y, y)

    return X, angles, y, ids


def get_dataloaders(batch_size=32, num_workers=2, load_cached_data=True):
    """
    Main function to prepare DataLoaders.
    """
    seed_everything(42)

    # 1. Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # 2. Calculate Angle Mean from Train Set (for imputation)
    # 'inc_angle' in metadata is numeric, NaNs are already np.nan
    angle_mean = train_meta["inc_angle"].mean()
    # If for some reason mean is NaN (all empty), fallback to 0 or some default
    if np.isnan(angle_mean):
        angle_mean = 0.0

    # 3. Check if we need to load raw JSONs
    # We only load raw JSONs if ANY of the cache files are missing or load_cached_data is False
    # Check cache existence
    modes = ["train", "val", "test"]
    need_raw = False
    if not load_cached_data:
        need_raw = True
    else:
        for m in modes:
            if not (
                os.path.exists(os.path.join(CACHE_DIR, f"X_{m}.npy"))
                and os.path.exists(os.path.join(CACHE_DIR, f"angle_{m}.npy"))
                and os.path.exists(os.path.join(CACHE_DIR, f"ids_{m}.npy"))
            ):
                need_raw = True
                break
            if m != "test" and not os.path.exists(
                os.path.join(CACHE_DIR, f"y_{m}.npy")
            ):
                need_raw = True
                break

    raw_data_dict = {}
    if need_raw:
        print("Loading raw JSON files for processing...")
        # Load train.json
        with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
            raw_data_dict["train.json"] = json.load(f)
        # Load test.json
        with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
            raw_data_dict["test.json"] = json.load(f)

    # 4. Process Subsets
    X_train, ang_train, y_train, ids_train = process_subset(
        "train", train_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    X_val, ang_val, y_val, ids_val = process_subset(
        "val", val_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    X_test, ang_test, y_test, ids_test = process_subset(
        "test", test_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    # 5. Define Transforms
    # Train: Horizontal and Vertical Flip
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )

    # Val/Test: No transforms (identity)
    val_transform = None

    # 6. Create Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, ids_train, transform=train_transform
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, ids_val, transform=val_transform
    )
    test_dataset = IcebergDataset(
        X_test, ang_test, None, ids_test, transform=val_transform
    )

    # 7. Create DataLoaders
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
