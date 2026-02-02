import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
import cv2
from library.utils import get_slice_number

# Constants
CACHE_DIR = "./working/idea_34/"
IMG_SIZE = 320
NUM_SLICES = 16
MODALITIES = ["flair", "t1w", "t1wce", "t2w"]


class GNHRDataset(Dataset):
    """
    Dataset class that handles the loading and processing of MRI volumes
    for the GNHR-Net architecture.
    """

    def __init__(self, metadata_df, root_dir="./input"):
        self.metadata_df = metadata_df
        self.root_dir = root_dir
        self.img_size = IMG_SIZE
        self.num_slices = NUM_SLICES
        self.modalities = MODALITIES

    def __len__(self):
        return len(self.metadata_df)

    def load_and_process_modality(self, rel_paths):
        """
        Loads, sorts, samples, resizes, and normalizes a single modality volume.
        Returns a numpy array of shape (16, 320, 320).
        """
        # 1. Check for empty paths
        if len(rel_paths) == 0:
            return np.zeros(
                (self.num_slices, self.img_size, self.img_size), dtype=np.float32
            )

        # 2. Validate and Sort paths
        valid_paths = []
        for p in rel_paths:
            full_path = os.path.join(self.root_dir, p)
            if os.path.exists(full_path):
                valid_paths.append(full_path)

        if len(valid_paths) == 0:
            return np.zeros(
                (self.num_slices, self.img_size, self.img_size), dtype=np.float32
            )

        # Sort using the external integer sorting utility
        valid_paths.sort(key=lambda x: get_slice_number(x))

        # 3. Uniform Sampling (10% - 90%)
        total_slices = len(valid_paths)

        # Define sampling range
        start = int(total_slices * 0.1)
        end = int(total_slices * 0.9)

        # Handle edge cases where volume is too small
        if end <= start:
            start = 0
            end = total_slices

        # Generate indices: uniformly distributed between start and end
        # endpoint=False ensures we stay strictly within the range
        indices = np.linspace(start, end - 1, self.num_slices).astype(int)

        # 4. Load and Resize
        slices = []
        for idx in indices:
            # Clamp index just in case
            idx = max(0, min(idx, total_slices - 1))
            path = valid_paths[idx]

            try:
                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)

                # Resize to target resolution (320x320)
                if img.shape != (self.img_size, self.img_size):
                    img = cv2.resize(
                        img,
                        (self.img_size, self.img_size),
                        interpolation=cv2.INTER_AREA,
                    )

                slices.append(img)
            except Exception:
                # Fallback for corrupt file: append zero slice
                slices.append(
                    np.zeros((self.img_size, self.img_size), dtype=np.float32)
                )

        volume = np.array(slices)  # Shape (16, 320, 320)

        # 5. View-Adaptive Per-Modality Normalization
        # Normalize based on the min/max of the *selected* slices only
        min_val = np.min(volume)
        max_val = np.max(volume)

        if max_val - min_val > 0:
            volume = (volume - min_val) / (max_val - min_val)
        else:
            # Avoid division by zero if volume is constant (e.g. all black)
            volume = np.zeros_like(volume)

        return volume

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        brats_id = row["BraTS21ID"]

        # Map column names in parquet to modalities
        path_cols = {
            "flair": "flair_paths",
            "t1w": "t1w_paths",
            "t1wce": "t1wce_paths",
            "t2w": "t2w_paths",
        }

        channels = []
        for mod in self.modalities:
            col_name = path_cols[mod]
            paths = row[col_name]

            # Handle potential None/NaN in dataframe
            if paths is None or not isinstance(paths, (list, np.ndarray)):
                paths = []

            vol = self.load_and_process_modality(paths)
            channels.append(vol)

        # Stack Modalities
        # Each vol is (16, 320, 320). We want (64, 320, 320).
        # Concatenate along the depth/channel axis (axis 0)
        X = np.concatenate(channels, axis=0)

        # Convert to Tensor
        X_tensor = torch.from_numpy(X).float()

        # Get Target
        if "MGMT_value" in row:
            y = torch.tensor(float(row["MGMT_value"]), dtype=torch.float32)
        else:
            y = torch.tensor(-1.0, dtype=torch.float32)  # Dummy for test set

        return X_tensor, y, brats_id


class CachedDataset(Dataset):
    """
    Simple wrapper for pre-loaded tensors in RAM.
    """

    def __init__(self, x_data, y_data, id_data):
        self.x = x_data
        self.y = y_data
        self.ids = id_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ids[idx]


def get_dataloaders(
    batch_size=8, num_workers=4, load_cached_data=True, limit_data=None
):
    """
    Creates DataLoaders for train, val, and test sets.
    Implements caching logic to save processed arrays to disk.
    """

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    splits = ["train", "val", "test"]
    dataloaders = {}

    for split in splits:
        meta_path = f"./metadata/{split}.parquet"
        if not os.path.exists(meta_path):
            continue

        # Define cache file paths
        cache_X = os.path.join(CACHE_DIR, f"cached_{split}_X.npy")
        cache_y = os.path.join(CACHE_DIR, f"cached_{split}_y.npy")
        cache_ids = os.path.join(CACHE_DIR, f"cached_{split}_ids.npy")

        loaded_from_cache = False

        # 1. Try Loading from Cache
        if (
            load_cached_data
            and os.path.exists(cache_X)
            and os.path.exists(cache_y)
            and os.path.exists(cache_ids)
        ):
            try:
                print(f"[{split.upper()}] Loading data from cache...")
                # Load into RAM (220GB available, so this is safe and fastest)
                X_np = np.load(cache_X)
                y_np = np.load(cache_y)
                ids_np = np.load(cache_ids, allow_pickle=True)
                loaded_from_cache = True
            except Exception as e:
                print(f"[{split.upper()}] Cache load failed: {e}. Reprocessing...")
                loaded_from_cache = False

        # 2. Process from Scratch if needed
        if not loaded_from_cache:
            print(f"[{split.upper()}] Processing data from scratch...")
            df = pd.read_parquet(meta_path)

            if limit_data:
                df = df.head(limit_data)

            # Use the GNHRDataset logic to process samples one by one
            dataset_processor = GNHRDataset(df)

            X_list = []
            y_list = []
            ids_list = []

            for i in range(len(dataset_processor)):
                xi, yi, idi = dataset_processor[i]
                X_list.append(xi.numpy())
                y_list.append(yi.item())
                ids_list.append(str(idi))

                if (i + 1) % 50 == 0:
                    print(f"Processed {i+1}/{len(dataset_processor)} samples")

            X_np = np.array(X_list, dtype=np.float32)
            y_np = np.array(y_list, dtype=np.float32)
            ids_np = np.array(ids_list)

            # Save to cache for future runs
            print(f"[{split.upper()}] Saving cache to {CACHE_DIR}...")
            np.save(cache_X, X_np)
            np.save(cache_y, y_np)
            np.save(cache_ids, ids_np)

        # 3. Create Final Dataset and DataLoader
        tensor_x = torch.from_numpy(X_np)
        tensor_y = torch.from_numpy(y_np)

        final_dataset = CachedDataset(tensor_x, tensor_y, ids_np)

        shuffle = split == "train"

        dataloaders[split] = DataLoader(
            final_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloaders
