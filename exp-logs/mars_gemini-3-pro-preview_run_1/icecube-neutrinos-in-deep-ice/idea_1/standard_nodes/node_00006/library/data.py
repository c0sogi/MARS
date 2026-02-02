import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
from library.config import Config
from library.utils import load_sensor_geometry, angles_to_vector


class NeutrinoDataset(IterableDataset):
    """
    PyTorch IterableDataset for Neutrino Direction Prediction.
    Reads data from Parquet batch files using a metadata index.
    Optimized for sequential file access to avoid memory thrashing.
    """

    def __init__(self, metadata, geometry, mode="train"):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing event_id, file_path, indices, etc.
            geometry (pd.DataFrame): Sensor geometry (index=sensor_id, cols=[x, y, z]).
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata
        self.mode = mode

        # Convert geometry to numpy array for fast O(1) lookup
        max_id = int(geometry.index.max())
        self.geo_array = np.zeros((max_id + 1, 3), dtype=np.float32)
        self.geo_array[geometry.index] = geometry[["x", "y", "z"]].values

        # Group metadata by file_path for efficient iteration
        # Cite debug_lesson_1: Align Data Sampling with Storage Shards
        self.file_groups = list(self.metadata.groupby("file_path"))

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            my_groups = self.file_groups
        else:
            # Shard files across workers to prevent redundant loading
            # Cite debug_lesson_2: Shard Datasets Across Workers
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            my_groups = self.file_groups[worker_id::num_workers]

        # Shuffle files if training
        if self.mode == "train":
            # Shuffle the order of files assigned to this worker
            indices = np.random.permutation(len(my_groups))
            my_groups = [my_groups[i] for i in indices]

        for file_path, meta_subset in my_groups:
            # Load the batch file once
            full_path = os.path.join(Config.INPUT_DIR, file_path)
            cols = ["time", "sensor_id", "charge", "auxiliary"]

            try:
                # Load entire batch file
                df_batch = pd.read_parquet(full_path, columns=cols, engine="pyarrow")
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue

            # Shuffle events within the file if training
            if self.mode == "train":
                meta_subset = meta_subset.sample(frac=1).reset_index(drop=True)

            # Iterate over events in this batch
            for row in meta_subset.itertuples():
                first_idx = int(row.first_pulse_index)
                last_idx = int(row.last_pulse_index)

                # Slice the specific event
                event_pulses = df_batch.iloc[first_idx : last_idx + 1]

                # Filter Auxiliary (Noise)
                mask = ~event_pulses["auxiliary"]
                event_pulses = event_pulses[mask]

                # Handle case where all pulses are auxiliary
                if len(event_pulses) == 0:
                    x = torch.zeros(
                        (Config.SEQ_LEN, Config.NUM_FEATURES), dtype=torch.float32
                    )
                    if self.mode == "test":
                        yield x, torch.tensor(row.event_id, dtype=torch.long)
                    else:
                        yield x, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
                    continue

                # Sort by time
                times = event_pulses["time"].values
                charges = event_pulses["charge"].values
                sensor_ids = event_pulses["sensor_id"].values.astype(int)

                sort_idx = np.argsort(times)
                times = times[sort_idx]
                charges = charges[sort_idx]
                sensor_ids = sensor_ids[sort_idx]

                # Feature Engineering
                coords = self.geo_array[sensor_ids]
                norm_time = (times - Config.NORM_TIME_MEAN) / Config.NORM_TIME_STD
                norm_coords = (coords - Config.NORM_COORD_MEAN) / Config.NORM_COORD_STD
                norm_charge = np.log1p(charges)

                features = np.column_stack(
                    [norm_coords, norm_time, norm_charge]
                ).astype(np.float32)

                # Pad or Truncate
                L = features.shape[0]
                target_len = Config.SEQ_LEN
                final_tensor = np.zeros(
                    (target_len, Config.NUM_FEATURES), dtype=np.float32
                )

                if L >= target_len:
                    final_tensor[:] = features[:target_len]
                else:
                    final_tensor[:L] = features

                x = torch.from_numpy(final_tensor)

                if self.mode == "test":
                    yield x, torch.tensor(row.event_id, dtype=torch.long)
                else:
                    az = row.azimuth
                    zen = row.zenith
                    vec = angles_to_vector(az, zen)
                    y = torch.from_numpy(vec).float()
                    yield x, y

            # Explicitly clear memory after processing the file
            del df_batch
            gc.collect()


def get_dataloaders():
    """
    Creates DataLoaders for train and validation sets.

    Returns:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
    """
    # 1. Load Metadata
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH)
    val_meta = pd.read_parquet(Config.VAL_META_PATH)

    # 2. Debug Subsetting
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling subset of data...")

        # Cite debug_lesson_1: Align Data Sampling with Storage Shards
        def filter_by_batch(df, target_size):
            unique_batches = df["batch_id"].unique()
            selected_batches = []
            current_count = 0
            for batch in unique_batches:
                batch_count = len(df[df["batch_id"] == batch])
                selected_batches.append(batch)
                current_count += batch_count
                if current_count >= target_size:
                    break
            return df[df["batch_id"].isin(selected_batches)].iloc[:target_size]

        train_meta = filter_by_batch(train_meta, Config.DEBUG_SUBSET_SIZE)
        val_meta = filter_by_batch(val_meta, Config.DEBUG_SUBSET_SIZE)

    # 3. Load Geometry
    # Using load_cached_data=True to use the cached parquet file if available
    geometry = load_sensor_geometry(load_cached_data=True)

    # 4. Create Datasets
    train_dataset = NeutrinoDataset(train_meta, geometry, mode="train")
    val_dataset = NeutrinoDataset(val_meta, geometry, mode="val")

    # 5. Create DataLoaders
    # persistent_workers=True keeps the workers (and their batch_cache) alive between epochs
    # Only enable persistent_workers if num_workers > 0
    use_persistent_workers = Config.NUM_WORKERS > 0

    # Cite debug_lesson_1: Align Data Sampling with Storage Shards
    # We switch to IterableDataset which handles shuffling internally.
    # DataLoader shuffle must be False for IterableDataset.
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=use_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates DataLoader for the test set.

    Returns:
        test_loader (DataLoader): Loader for test data.
    """
    # 1. Load Metadata
    test_meta = pd.read_parquet(Config.TEST_META_PATH)

    # 1.5 Debug Subsetting
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling subset of test data...")
        # Cite debug_lesson_1: Align Data Sampling with Storage Shards
        unique_batches = test_meta["batch_id"].unique()
        selected_batches = []
        current_count = 0
        for batch in unique_batches:
            batch_count = len(test_meta[test_meta["batch_id"] == batch])
            selected_batches.append(batch)
            current_count += batch_count
            if current_count >= Config.DEBUG_SUBSET_SIZE:
                break
        test_meta = test_meta[test_meta["batch_id"].isin(selected_batches)].iloc[
            : Config.DEBUG_SUBSET_SIZE
        ]

    # 2. Load Geometry
    geometry = load_sensor_geometry(load_cached_data=True)

    # 3. Create Dataset
    test_dataset = NeutrinoDataset(test_meta, geometry, mode="test")

    # 4. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    return test_loader
