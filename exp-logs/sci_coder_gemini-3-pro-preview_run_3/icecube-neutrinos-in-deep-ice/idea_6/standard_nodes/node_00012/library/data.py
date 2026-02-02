import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library import utils

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def load_sensor_geometry(load_cached_data=True):
    """
    Loads the sensor geometry mapping sensor_id -> (x, y, z).
    Implements caching as required.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "sensor_geometry_map.npy")

    if load_cached_data and os.path.exists(cache_path):
        return np.load(cache_path)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Load from CSV
    df = pd.read_csv(Config.SENSOR_GEOMETRY)

    # Create a dense array for fast lookup
    # Assuming sensor_ids are contiguous or close to it. Max ID is ~5160.
    max_id = df["sensor_id"].max()
    geo_map = np.zeros((max_id + 1, 3), dtype=np.float32)

    # Fill map
    geo_map[df["sensor_id"].values] = df[["x", "y", "z"]].values.astype(np.float32)

    # Save to cache
    np.save(cache_path, geo_map)

    return geo_map


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------


class IceCubeDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform
        self.geometry_map = load_sensor_geometry(load_cached_data=True)

        # Load Metadata
        if mode == "train":
            self.meta_path = Config.TRAIN_META
        elif mode == "val":
            self.meta_path = Config.VAL_META
        elif mode == "test":
            self.meta_path = Config.TEST_META
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.metadata = pd.read_parquet(self.meta_path)

        # Debugging: Reduce dataset size
        if Config.DEBUG:
            # Cite debug_lesson_1: Preserve Data Locality When Subsampling Large Datasets
            # Filter to a single batch to avoid I/O thrashing due to shuffled metadata
            if not self.metadata.empty:
                target_batch = self.metadata["batch_id"].iloc[0]
                self.metadata = self.metadata[self.metadata["batch_id"] == target_batch]
            self.metadata = self.metadata.iloc[: Config.DEBUG_SIZE].copy()

        # Batch Caching
        self.current_batch_id = None
        self.current_batch_df = None

        # Pre-compute file path mapping for faster lookup
        # metadata contains 'batch_id' and 'file_path' (relative)
        # We create a map: batch_id -> full_path
        unique_batches = self.metadata[["batch_id", "file_path"]].drop_duplicates()
        self.batch_paths = {
            row.batch_id: os.path.join(Config.INPUT_DIR, row.file_path)
            for row in unique_batches.itertuples(index=False)
        }

    def __len__(self):
        return len(self.metadata)

    def _get_batch_df(self, batch_id):
        """
        Loads the batch dataframe, using a simple 1-element cache.
        """
        if self.current_batch_id != batch_id:
            path = self.batch_paths[batch_id]
            # Load the batch file
            # The batch files have event_id as index.
            self.current_batch_df = pd.read_parquet(path)
            self.current_batch_id = batch_id
        return self.current_batch_df

    def __getitem__(self, idx):
        # 1. Get Event Metadata
        row = self.metadata.iloc[idx]
        event_id = row.event_id
        batch_id = row.batch_id

        # 2. Load Pulse Data
        batch_df = self._get_batch_df(batch_id)

        # Retrieve pulses for this event.
        # Using loc[event_id] is efficient since event_id is the index in parquet
        try:
            event_pulses = batch_df.loc[event_id]
            # If only one pulse, it returns a Series, convert to DataFrame
            if isinstance(event_pulses, pd.Series):
                event_pulses = event_pulses.to_frame().T
        except KeyError:
            # Fallback for missing events (should not happen in clean data)
            event_pulses = pd.DataFrame(
                columns=["sensor_id", "time", "charge", "auxiliary"]
            )

        # 3. Merge Geometry
        sensor_ids = event_pulses["sensor_id"].values.astype(int)
        pos = self.geometry_map[sensor_ids]  # (N, 3)

        time = event_pulses["time"].values.astype(np.float32)
        charge = event_pulses["charge"].values.astype(np.float32)
        auxiliary = event_pulses["auxiliary"].values.astype(np.float32)

        # 4. Hybrid Sampling
        # Select N_PULSES based on charge and time
        n_pulses = len(time)
        target_n = Config.N_PULSES

        if n_pulses > target_n:
            # Indices sorted by charge (descending) and time (ascending)
            # We want to keep high charge and early time

            # Top k charge
            idx_charge = np.argsort(charge)[-target_n:]

            # Top k early time
            idx_time = np.argsort(time)[:target_n]

            # Union
            selected_indices = np.union1d(idx_charge, idx_time)

            # If still too many, prioritize charge
            if len(selected_indices) > target_n:
                # Get charges of selected
                sub_charges = charge[selected_indices]
                # Pick top N from these
                sub_top_k = np.argsort(sub_charges)[-target_n:]
                selected_indices = selected_indices[sub_top_k]

            # Apply selection
            pos = pos[selected_indices]
            time = time[selected_indices]
            charge = charge[selected_indices]
            auxiliary = auxiliary[selected_indices]

        # 5. Canonical Transformation
        # Compute rotation matrix based on the sampled pulses
        R, center = utils.compute_canonical_rotation(pos, time, charge)

        # Apply rotation to positions
        pos_rot = utils.apply_rotation(pos, R, center)

        # 6. Feature Engineering
        # Normalize time: relative to start, scaled
        if len(time) > 0:
            t_min = time.min()
            time_rel = (time - t_min) / 1000.0  # Scale to roughly unit variance
        else:
            time_rel = time

        # Log scale charge
        charge_log = np.log10(charge + 1.0)

        # Construct Node Features
        # Features: [x_rot, y_rot, z_rot, time_rel, charge_log, auxiliary]
        # Shape: (N, 6)
        x_features = np.stack(
            [
                pos_rot[:, 0],
                pos_rot[:, 1],
                pos_rot[:, 2],
                time_rel,
                charge_log,
                auxiliary,
            ],
            axis=1,
        )

        # 7. Target Processing
        target_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if self.mode in ["train", "val"]:
            # Get azimuth/zenith from metadata
            az = row.azimuth
            ze = row.zenith

            # Convert to vector
            tx, ty, tz = utils.spherical_to_cartesian(az, ze)
            true_vec = np.array([tx, ty, tz], dtype=np.float32)

            # Rotate target vector into canonical frame
            # Vectors are not translated, only rotated
            target_vec = utils.apply_rotation(true_vec, R, inverse=False)
            # Ensure shape is (3,)
            target_vec = target_vec.flatten()

        # 8. Padding
        # We need fixed size for batching
        curr_n = x_features.shape[0]
        pad_size = target_n - curr_n

        mask = np.ones(target_n, dtype=bool)

        if pad_size > 0:
            # Pad with zeros
            padding = np.zeros((pad_size, x_features.shape[1]), dtype=np.float32)
            x_features = np.concatenate([x_features, padding], axis=0)

            # Pad positions
            pos_padding = np.zeros((pad_size, 3), dtype=np.float32)
            pos_rot = np.concatenate([pos_rot, pos_padding], axis=0)

            # Update mask
            mask[curr_n:] = False

        elif pad_size < 0:
            # Should not happen due to sampling logic, but for safety
            x_features = x_features[:target_n]
            pos_rot = pos_rot[:target_n]
            mask = mask[:target_n]  # Should be all true

        # Convert to tensors
        return {
            "x": torch.from_numpy(x_features).float(),  # (N, 6)
            "pos": torch.from_numpy(pos_rot).float(),  # (N, 3)
            "mask": torch.from_numpy(mask).bool(),  # (N,)
            "target": torch.from_numpy(target_vec).float(),  # (3,)
            "rotation": torch.from_numpy(R).float(),  # (3, 3)
            "event_id": torch.tensor(event_id, dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------------


def get_dataloaders():
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Train Loader
    train_dataset = IceCubeDataset(mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    # Validation Loader
    val_dataset = IceCubeDataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    # Test Loader
    test_dataset = IceCubeDataset(mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
    )

    return train_loader, val_loader, test_loader
