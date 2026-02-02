import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config, set_seed
from library.utils import get_config_hash

# ==========================================
# Constants for Normalization
# ==========================================
# Approximate scale factors based on detector geometry and physics
POS_SCALE = 500.0
TIME_SCALE = 30000.0
CHARGE_SCALE = 1.0  # We use log1p, so raw scale is less critical


class BatchProcessor:
    """
    Handles the loading, preprocessing, and caching of raw data batches.
    """

    def __init__(self, geometry_path):
        self.geometry = pd.read_csv(geometry_path)
        self.sensor_map = self.geometry.set_index("sensor_id")[["x", "y", "z"]]

    def process_batch(self, batch_id, meta_df, mode="train", load_cached_data=True):
        """
        Processes a single batch of data.

        Args:
            batch_id (int): The ID of the batch.
            meta_df (pd.DataFrame): Metadata containing event_ids to process in this batch.
            mode (str): 'train' or 'test'. If 'test', targets are not computed.
            load_cached_data (bool): Whether to attempt loading from disk.

        Returns:
            dict: Dictionary containing 'seq', 'features', 'targets', 'event_ids'.
        """
        # 1. Generate Cache Path
        # Create a config dictionary relevant for data processing to hash
        process_config = {
            "seq_len": Config.SEQ_LEN,
            "seq_features": Config.SEQ_FEATURES,
            "manual_features": Config.MANUAL_FEATURES,
            "pos_scale": POS_SCALE,
            "time_scale": TIME_SCALE,
            "mode": mode,
        }
        config_hash = get_config_hash(process_config)
        cache_filename = f"batch_{batch_id}_{mode}_{config_hash}.npz"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 2. Try Load Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                # We return the path to the npz file to allow memory mapping later,
                # or load it here. For robustness, we load it here but return the data dict.
                # To save memory in the main process, we might want to return the path,
                # but the Dataset expects data or a way to access it.
                # Let's return the path to the cached file.
                return cache_path
            except Exception as e:
                print(f"Failed to load cache {cache_path}: {e}. Recomputing.")

        # 3. Compute from Scratch
        # Construct raw file path
        # Note: Input directory structure is train/batch_X.parquet or test/batch_X.parquet
        folder = "train" if mode != "test" else "test"
        raw_path = os.path.join(Config.INPUT_DIR, folder, f"batch_{batch_id}.parquet")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw batch file not found: {raw_path}")

        # Load raw data
        df = pd.read_parquet(raw_path)

        # Filter to events present in meta_df (which might be a subset due to MAX_SAMPLES)
        # However, for efficiency, we usually process the whole batch and filter later,
        # but to save space we can filter now.
        valid_events = meta_df["event_id"].unique()
        df = df[df.index.isin(valid_events)]  # event_id is index in parquet

        # Reset index to make event_id a column
        df = df.reset_index()

        # Merge Geometry
        df = df.merge(self.geometry, on="sensor_id", how="left")

        # Filter Auxiliary
        df = df[~df["auxiliary"]]

        # --- Feature Engineering (Global) ---
        # We compute these before truncating the sequence

        # Weighted Centroids
        df["wx"] = df["x"] * df["charge"]
        df["wy"] = df["y"] * df["charge"]
        df["wz"] = df["z"] * df["charge"]

        # Groupby for aggregation
        # We sort by event_id to ensure alignment
        df = df.sort_values(by=["event_id", "time"])

        grp = df.groupby("event_id")

        # Aggregations
        # Note: This can be slow. Optimized approach:
        agg_df = grp.agg(
            {
                "charge": ["sum", "count"],
                "wx": "sum",
                "wy": "sum",
                "wz": "sum",
                "time": [lambda x: x.quantile(0.1), lambda x: x.quantile(0.5)],
            }
        )

        # Flatten columns
        agg_df.columns = [
            "total_charge",
            "n_pulses",
            "sum_wx",
            "sum_wy",
            "sum_wz",
            "time_q10",
            "time_q50",
        ]

        # Compute Manual Features
        # Avoid div by zero
        total_q = agg_df["total_charge"].replace(0, 1.0)

        manual_feats = pd.DataFrame(index=agg_df.index)
        manual_feats["center_x"] = agg_df["sum_wx"] / total_q / POS_SCALE
        manual_feats["center_y"] = agg_df["sum_wy"] / total_q / POS_SCALE
        manual_feats["center_z"] = agg_df["sum_wz"] / total_q / POS_SCALE
        manual_feats["time_q10"] = (
            agg_df["time_q10"] - df["time"].min()
        ) / TIME_SCALE  # Relative to global min? Or batch min?
        # Better: relative to 0 since time is relative in event.
        # Actually time is relative within event window, but values are large (e.g. 10000).
        # We normalize simply by scale.
        manual_feats["time_q10"] = agg_df["time_q10"] / TIME_SCALE
        manual_feats["time_q50"] = agg_df["time_q50"] / TIME_SCALE
        manual_feats["total_charge"] = np.log1p(agg_df["total_charge"])

        # Ensure column order matches Config
        manual_feats = manual_feats[Config.MANUAL_FEATURES].astype(np.float32)

        # --- Sequence Generation ---
        # We need top L pulses per event
        # df is already sorted by event_id, time

        # Add a counter for pulses within event
        df["pulse_idx"] = grp.cumcount()

        # Filter top L
        df_seq = df[df["pulse_idx"] < Config.SEQ_LEN].copy()

        # Normalize Sequence Features
        df_seq["x"] = df_seq["x"] / POS_SCALE
        df_seq["y"] = df_seq["y"] / POS_SCALE
        df_seq["z"] = df_seq["z"] / POS_SCALE
        df_seq["time"] = df_seq["time"] / TIME_SCALE
        df_seq["charge"] = np.log1p(df_seq["charge"])

        # Pivot to create tensor structure (N_events, L, N_features)
        # We use pivot_table.
        # Features: x, y, z, time, charge
        features = Config.SEQ_FEATURES

        # Create a multi-index pivot
        # Index: event_id, Columns: pulse_idx
        # This results in columns like (x, 0), (x, 1)...
        pivot = df_seq.pivot(index="event_id", columns="pulse_idx", values=features)

        # Reindex to ensure all event_ids from meta are present (if some were filtered out completely by aux)
        # and ensure all pulse indices 0..L-1 exist
        # However, reindexing is expensive.
        # We only keep events that survived the aux filter.

        # The pivot table has MultiIndex columns: Level 0 = feature, Level 1 = pulse_idx
        # We need to stack this into (N, L, F)

        # Get unique event ids from pivot (sorted)
        event_ids_processed = pivot.index.values

        # Initialize output tensor
        n_events = len(event_ids_processed)
        seq_tensor = np.zeros(
            (n_events, Config.SEQ_LEN, len(features)), dtype=np.float32
        )

        for i, feat in enumerate(features):
            # Extract sub-dataframe for this feature
            if feat in pivot:
                feat_df = pivot[feat]
                # feat_df has columns 0, 1, 2... up to max pulses found
                # We need to map these to 0..SEQ_LEN
                # Fill missing with 0
                # Get values as numpy
                # We need to ensure we have columns 0..SEQ_LEN-1
                # Reindex columns
                feat_df = feat_df.reindex(columns=range(Config.SEQ_LEN), fill_value=0.0)
                seq_tensor[:, :, i] = feat_df.values.astype(np.float32)

        # --- Align Targets ---
        # manual_feats and seq_tensor are aligned by event_ids_processed
        # We need to get targets for these events

        if mode != "test":
            # meta_df has targets. Set index to event_id to align
            target_df = meta_df.set_index("event_id").loc[event_ids_processed]
            targets = target_df[["azimuth", "zenith"]].values.astype(np.float32)
        else:
            targets = np.zeros((n_events, 2), dtype=np.float32)  # Dummy

        # Align manual features
        manual_feats = manual_feats.loc[event_ids_processed].values.astype(np.float32)

        # Save to disk
        np.savez(
            cache_path,
            seq=seq_tensor,
            features=manual_feats,
            targets=targets,
            event_ids=event_ids_processed,
        )

        # Clean up
        del df, df_seq, pivot, grp, agg_df
        gc.collect()

        return cache_path


class IceCubeDataset(Dataset):
    def __init__(self, meta_df, mode="train", load_cached_data=True):
        """
        Args:
            meta_df (pd.DataFrame): Metadata for the dataset.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Use caching.
        """
        self.mode = mode
        self.meta_df = meta_df.copy()

        # Sort metadata to ensure deterministic order
        self.meta_df = self.meta_df.sort_values(["batch_id", "event_id"]).reset_index(
            drop=True
        )

        # Initialize processor
        self.processor = BatchProcessor(Config.SENSOR_GEOMETRY_PATH)

        # Pre-process / Ensure cache exists for all batches
        unique_batches = self.meta_df["batch_id"].unique()
        self.batch_files = {}

        print(f"Preparing {len(unique_batches)} batches for {mode} set...")

        for batch_id in unique_batches:
            # Get subset of meta for this batch
            batch_meta = self.meta_df[self.meta_df["batch_id"] == batch_id]

            # Process (or get cache path)
            cache_path = self.processor.process_batch(
                batch_id, batch_meta, mode=mode, load_cached_data=load_cached_data
            )
            self.batch_files[batch_id] = cache_path

        # Build Index Mapping
        # We need to map global index -> (batch_id, local_index)
        # Since we processed batches and they might have filtered events (e.g. all aux),
        # we need to know exactly which events are in the processed files.

        # To do this efficiently without loading all files:
        # We assume the process_batch saves ALL events requested in meta_df that have valid data.
        # If an event has NO non-aux pulses, it might be dropped.
        # For this implementation, we will load the 'event_ids' array from each cache file
        # to build the precise map. This is fast (small read).

        self.global_index = []  # List of (batch_id, local_idx)
        self.valid_event_ids = []

        for batch_id in unique_batches:
            path = self.batch_files[batch_id]
            # Load only event_ids
            with np.load(path) as data:
                e_ids = data["event_ids"]

            # Create indices
            for i, eid in enumerate(e_ids):
                self.global_index.append((batch_id, i))
                self.valid_event_ids.append(eid)

        self.length = len(self.global_index)
        print(f"Dataset {mode} initialized. Total events: {self.length}")

        # Open files in mmap mode for random access
        self.open_files = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_id, local_idx = self.global_index[idx]

        # Lazy load / mmap file
        if batch_id not in self.open_files:
            # We use mmap_mode='r' to avoid loading full array to RAM
            # We keep the file handle open
            self.open_files[batch_id] = np.load(
                self.batch_files[batch_id], mmap_mode="r"
            )

        data = self.open_files[batch_id]

        # Retrieve data
        # Copy to ensure it's a writable tensor and not a read-only mmap slice
        seq = torch.tensor(data["seq"][local_idx], dtype=torch.float32)
        features = torch.tensor(data["features"][local_idx], dtype=torch.float32)

        if self.mode != "test":
            targets = torch.tensor(data["targets"][local_idx], dtype=torch.float32)
            return seq, features, targets
        else:
            # For test, we might need event_id to map back
            event_id = self.valid_event_ids[idx]
            return seq, features, torch.tensor(event_id, dtype=torch.long)


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use disk caching.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load Metadata
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH)
    val_meta = pd.read_parquet(Config.VAL_META_PATH)
    test_meta = pd.read_parquet(Config.TEST_META_PATH)

    # Debugging / Subsampling
    if Config.MAX_SAMPLES:
        print(f"DEBUG: Subsampling dataset to {Config.MAX_SAMPLES} samples.")
        train_meta = train_meta.iloc[: Config.MAX_SAMPLES]
        val_meta = val_meta.iloc[: Config.MAX_SAMPLES]
        test_meta = test_meta.iloc[: Config.MAX_SAMPLES]

    # Create Datasets
    train_dataset = IceCubeDataset(
        train_meta, mode="train", load_cached_data=load_cached_data
    )
    val_dataset = IceCubeDataset(
        val_meta, mode="val", load_cached_data=load_cached_data
    )
    test_dataset = IceCubeDataset(
        test_meta, mode="test", load_cached_data=load_cached_data
    )

    # Create DataLoaders
    # num_workers > 0 works best with file paths or simple gets,
    # but with mmap objects inside dataset, we need to be careful.
    # Usually mmap works fine with fork.

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
