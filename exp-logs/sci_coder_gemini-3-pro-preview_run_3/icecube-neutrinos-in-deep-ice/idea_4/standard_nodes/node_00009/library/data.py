import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Data, Batch
import pyarrow.parquet as pq

import library.config as config
import library.utils as utils


# -----------------------------------------------------------------------------
# Custom Sampler for Efficient Batch Reading
# -----------------------------------------------------------------------------
class GroupedBatchSampler(Sampler):
    """
    Sampler that yields indices grouped by batch_id to minimize file I/O.
    For training, it shuffles the order of batches and indices within batches.
    """

    def __init__(self, metadata, batch_size, shuffle=True, seed=42):
        self.metadata = metadata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # Group indices by batch_id
        # metadata is a pandas DataFrame with 'batch_id'
        # We assume the index of the dataframe corresponds to the dataset index
        self.batch_groups = self.metadata.groupby(
            "batch_id"
        ).indices  # dict: batch_id -> array of indices
        self.batch_ids = list(self.batch_groups.keys())

    def __iter__(self):
        # Shuffle batch order
        batch_order = self.batch_ids[:]
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(batch_order)

        final_indices = []
        for bid in batch_order:
            indices = self.batch_groups[bid]
            if self.shuffle:
                # Different seed per batch to ensure randomness across epochs if seed changes
                # But here we keep it deterministic per run based on init seed
                rng_batch = np.random.default_rng(self.seed + bid)
                rng_batch.shuffle(indices)
            final_indices.extend(indices)

        return iter(final_indices)

    def __len__(self):
        return len(self.metadata)


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class IceCubeGraphDataset(Dataset):
    def __init__(self, metadata_path, mode="train", transform=None):
        """
        Args:
            metadata_path (str): Path to the metadata parquet file.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        # Load Metadata
        try:
            self.meta = pd.read_parquet(metadata_path)
        except FileNotFoundError:
            print(f"Metadata file not found at {metadata_path}")
            raise

        # Debugging: Sample subset if configured
        if config.DEBUG_SAMPLE_SIZE is not None:
            # Sort by batch_id to ensure we only load a few files (Cite debug_lesson_1)
            self.meta = (
                self.meta.sort_values("batch_id")
                .iloc[: config.DEBUG_SAMPLE_SIZE]
                .reset_index(drop=True)
            )

        # Load Sensor Geometry
        self.sensor_geometry = utils.load_sensor_geometry()

        # Caching mechanism for batch files
        # We keep the current batch dataframe in memory to avoid reloading for sequential access
        self.current_batch_id = -1
        self.current_batch_df = None

    def __len__(self):
        return len(self.meta)

    def _load_batch_data(self, batch_id, file_path):
        """
        Loads the batch data from parquet.
        Implements a simple 1-element LRU cache.
        """
        if batch_id != self.current_batch_id:
            full_path = os.path.join(config.INPUT_DIR, file_path)
            try:
                # Read the parquet file using pyarrow engine
                self.current_batch_df = pd.read_parquet(full_path, engine="pyarrow")
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                raise e

            self.current_batch_id = batch_id

        return self.current_batch_df

    def __getitem__(self, idx):
        # 1. Get Event Metadata
        row = self.meta.iloc[idx]
        event_id = int(row["event_id"])
        batch_id = int(row["batch_id"])

        # 2. Load Pulse Data
        # Relies on GroupedBatchSampler to make this efficient (sequential access to same batch)
        df_batch = self._load_batch_data(batch_id, row["file_path"])

        # Extract pulses for this event
        start_idx = int(row["first_pulse_index"])
        end_idx = int(row["last_pulse_index"])

        # Slice the dataframe (end_idx is inclusive in metadata, exclusive in iloc)
        event_pulses = df_batch.iloc[start_idx : end_idx + 1]

        # 3. Preprocessing
        # Filter auxiliary pulses (noise) if we have enough data
        mask = event_pulses["auxiliary"] == False
        if mask.sum() >= config.MIN_PULSES:
            event_pulses = event_pulses[mask]

        # Get features
        sensor_ids = event_pulses["sensor_id"].values.astype(np.int64)
        time = event_pulses["time"].values.astype(np.float32)
        charge = event_pulses["charge"].values.astype(np.float32)

        # Map geometry
        # Clip sensor_ids to be safe, though they should be valid
        sensor_ids = np.clip(sensor_ids, 0, len(self.sensor_geometry) - 1)
        pos = self.sensor_geometry[sensor_ids]  # Shape: (N, 3)

        # 4. Sampling
        num_pulses = len(charge)
        if num_pulses > config.NUM_POINTS:
            # Strategy: Prioritize high charge
            sort_idx = np.argsort(charge)[::-1]
            top_idx = sort_idx[: config.NUM_POINTS]

            pos = pos[top_idx]
            time = time[top_idx]
            charge = charge[top_idx]

        # 5. Normalization
        time_norm = (time - config.MEAN_TIME) / config.STD_TIME
        charge_norm = (charge - config.MEAN_CHARGE) / config.STD_CHARGE
        pos_norm = pos / config.COORD_SCALE

        # 6. Global Eigen Features
        # Compute on the selected pulses using raw values for physical accuracy
        eigen_features = utils.compute_eigen_characteristics(pos, charge)

        # 7. Construct Node Features
        # Feature vector: [x_norm, y_norm, z_norm, time_norm, charge_norm]
        x_feat = np.column_stack([pos_norm, time_norm, charge_norm]).astype(np.float32)

        # 8. Prepare Target
        if self.mode != "test":
            azimuth = row["azimuth"]
            zenith = row["zenith"]
            # Convert to Cartesian unit vector
            nx, ny, nz = utils.spherical_to_cartesian(azimuth, zenith)
            y = torch.tensor([nx, ny, nz], dtype=torch.float32)
        else:
            # Dummy target for test set
            y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # 9. Create PyG Data Object
        data = Data(
            x=torch.from_numpy(x_feat),
            pos=torch.from_numpy(pos_norm).float(),
            global_features=torch.from_numpy(eigen_features).unsqueeze(0),  # (1, 12)
            y=y.unsqueeze(0),  # (1, 3)
            event_id=torch.tensor([event_id], dtype=torch.long),
            num_nodes=x_feat.shape[0],
        )

        return data


# -----------------------------------------------------------------------------
# Collate Function
# -----------------------------------------------------------------------------
def pyg_collate_fn(data_list):
    """
    Collates a list of PyG Data objects into a Batch object.
    """
    return Batch.from_data_list(data_list)


# -----------------------------------------------------------------------------
# Data Loader Factory
# -----------------------------------------------------------------------------
def get_dataloaders(load_cached_data=False):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Placeholder for caching logic.
                                 The current implementation uses in-memory LRU caching
                                 optimized by the GroupedBatchSampler.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 1. Train Loader
    train_dataset = IceCubeGraphDataset(config.TRAIN_META_PATH, mode="train")

    # Use GroupedBatchSampler to optimize IO by grouping requests by batch file
    train_sampler = GroupedBatchSampler(
        train_dataset.meta, batch_size=config.BATCH_SIZE, shuffle=True, seed=config.SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        collate_fn=pyg_collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    # 2. Validation Loader
    val_dataset = IceCubeGraphDataset(config.VAL_META_PATH, mode="val")

    val_sampler = GroupedBatchSampler(
        val_dataset.meta,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No shuffle for validation
        seed=config.SEED,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        collate_fn=pyg_collate_fn,
        pin_memory=True,
    )

    # 3. Test Loader
    test_dataset = IceCubeGraphDataset(config.TEST_META_PATH, mode="test")

    test_sampler = GroupedBatchSampler(
        test_dataset.meta, batch_size=config.BATCH_SIZE, shuffle=False, seed=config.SEED
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=config.NUM_WORKERS,
        collate_fn=pyg_collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
