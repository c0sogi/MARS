import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import get_config_hash, load_sensor_geometry


class IceCubeDataset(Dataset):
    def __init__(self, mode="train", subset_size=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            subset_size (int, optional): Number of events to load. Defaults to Config limits.
            load_cached_data (bool): Whether to use cached pre-processed data.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Set default subset sizes from Config if not provided
        if subset_size is None:
            if mode == "train":
                self.subset_size = Config.TRAIN_SUBSET_SIZE
            elif mode == "val":
                self.subset_size = Config.VAL_SUBSET_SIZE
            else:
                self.subset_size = (
                    None  # Test usually processes all or is batched externally
                )
        else:
            self.subset_size = subset_size

        # Setup cache directory
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load Geometry and create fast lookup array
        # Sensor IDs range from 0 to ~5160
        geometry = load_sensor_geometry()
        max_sensor_id = geometry["sensor_id"].max()
        self.geo_arr = np.zeros((max_sensor_id + 1, 3), dtype=np.float32)
        self.geo_arr[geometry["sensor_id"]] = geometry[["x", "y", "z"]].values

        # Load Metadata
        if mode == "train":
            self.meta = pd.read_parquet(Config.TRAIN_META)
            self.split_name = "train"
        elif mode == "val":
            self.meta = pd.read_parquet(Config.VAL_META)
            # Validation data comes from the 'train' batch files
            self.split_name = "train"
        elif mode == "test":
            self.meta = pd.read_parquet(Config.TEST_META)
            self.split_name = "test"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply Subsampling
        if self.subset_size is not None and self.subset_size < len(self.meta):
            if mode == "train":
                # Shuffle for training to get a random subset of the huge dataset
                self.meta = self.meta.sample(
                    n=self.subset_size, random_state=Config.SEED
                ).reset_index(drop=True)
            else:
                # Deterministic slice for val/test
                self.meta = self.meta.iloc[: self.subset_size].reset_index(drop=True)

        # Containers for in-memory data
        self.features = []
        self.targets = []
        self.event_ids = []

        # Load and Process Data
        self._load_data()

    def _load_data(self):
        """
        Iterates over batches in metadata, loads/processes them, and aggregates into memory.
        """
        # Group metadata by batch_id to optimize IO
        batch_groups = self.meta.groupby("batch_id")

        print(
            f"[{self.mode.upper()}] Loading {len(self.meta)} events from {len(batch_groups)} batches..."
        )

        config_hash = get_config_hash()

        for batch_id, group in batch_groups:
            # Construct cache filename
            # We use split_name because train/val share 'train' source files, test uses 'test'
            cache_filename = f"batch_{batch_id}_{self.split_name}_{config_hash}.npz"
            cache_path = os.path.join(self.cache_dir, cache_filename)

            # 1. Try to load from cache
            loaded = False
            if self.load_cached_data and os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    batch_features = data["features"]
                    batch_event_ids = data["event_ids"]
                    loaded = True
                except Exception as e:
                    print(
                        f"Failed to load cache for batch {batch_id}: {e}. Reprocessing."
                    )

            # 2. Process from scratch if needed
            if not loaded:
                batch_features, _, batch_event_ids = self._process_batch(
                    batch_id, self.split_name
                )
                # Save to cache
                np.savez_compressed(
                    cache_path, features=batch_features, event_ids=batch_event_ids
                )

            # 3. Filter and Align
            # We only want the events present in our current metadata subset (`group`)
            # `batch_features` contains ALL events in the batch file.

            needed_ids = group["event_id"].values

            # Find indices of needed_ids in the loaded batch_event_ids
            # batch_event_ids is typically sorted, so we use searchsorted for speed
            idx = np.searchsorted(batch_event_ids, needed_ids)
            idx = np.clip(idx, 0, len(batch_event_ids) - 1)

            # Verify matches (searchsorted returns insertion point, need to check equality)
            found_mask = batch_event_ids[idx] == needed_ids
            valid_batch_indices = idx[found_mask]

            if len(valid_batch_indices) > 0:
                # Append Features
                self.features.append(batch_features[valid_batch_indices])
                self.event_ids.append(batch_event_ids[valid_batch_indices])

                # Append Targets
                # Targets are in the metadata `group`, not the batch file (for efficiency)
                if self.mode != "test":
                    # Extract targets corresponding to the found events
                    # Note: `group` might not be sorted by event_id if we shuffled metadata.
                    # We need to ensure targets align with the features we just appended.
                    # The features we appended are in the order of `needed_ids[found_mask]`.
                    # So we take the targets from `group` and apply `found_mask`.

                    batch_targets = group[["azimuth", "zenith"]].values
                    self.targets.append(batch_targets[found_mask])

        # Concatenate all batches
        if len(self.features) > 0:
            self.features = np.concatenate(self.features, axis=0)
            self.event_ids = np.concatenate(self.event_ids, axis=0)
            if self.mode != "test":
                self.targets = np.concatenate(self.targets, axis=0)
            else:
                self.targets = None
        else:
            # Handle empty case
            self.features = np.empty((0, Config.MAX_PULSES, 6), dtype=np.float32)
            self.event_ids = np.empty((0,), dtype=np.int64)
            self.targets = None

    def _process_batch(self, batch_id, split_name):
        """
        Loads a raw parquet batch, applies physics-informed sampling and normalization.
        Returns (features, targets=None, event_ids).
        """
        file_path = os.path.join(
            Config.INPUT_DIR, f"{split_name}/batch_{batch_id}.parquet"
        )

        # Load raw data
        df = pd.read_parquet(file_path)
        # Ensure event_id is a column
        if "event_id" not in df.columns:
            df = df.reset_index()

        # Extract columns as numpy arrays for speed
        raw_event_ids = df["event_id"].values
        times = df["time"].values.astype(np.float32)
        charges = df["charge"].values.astype(np.float32)
        auxs = df["auxiliary"].values.astype(np.float32)
        sensor_ids = df["sensor_id"].values

        # Map geometry
        coords = self.geo_arr[sensor_ids]
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

        # Identify event boundaries
        # Assuming sorted by event_id, find unique events and their counts
        unique_events, start_indices, counts = np.unique(
            raw_event_ids, return_index=True, return_counts=True
        )
        num_events = len(unique_events)

        N = Config.MAX_PULSES
        K = Config.EARLY_PULSES

        # Output container: (Num_Events, N, 6)
        # Channels: [x, y, z, time, charge, aux]
        features = np.zeros((num_events, N, 6), dtype=np.float32)

        # Iterate over events
        # Note: Vectorization across variable-length events is hard, explicit loop with optimized numpy ops is standard here.
        for i in range(num_events):
            start = start_indices[i]
            count = counts[i]
            end = start + count

            # Slice event data
            e_time = times[start:end]
            e_charge = charges[start:end]
            e_aux = auxs[start:end]
            e_x = xs[start:end]
            e_y = ys[start:end]
            e_z = zs[start:end]

            # --- Physics-Informed Sampling ---

            # 1. Sort by time to identify early pulses
            # Note: Data is relative time, we sort to be sure.
            t_sort_idx = np.argsort(e_time)

            # 2. Select indices
            if count <= N:
                # Take all, pad later
                selected_indices = t_sort_idx
            else:
                # Take first K
                keep_indices = t_sort_idx[:K]

                # Sample remaining N-K based on charge
                remainder_indices = t_sort_idx[K:]
                needed = N - K

                rem_charges = e_charge[remainder_indices]
                total_q = rem_charges.sum()

                if total_q > 0:
                    probs = rem_charges / total_q
                    # Normalize to ensure sum is exactly 1.0
                    probs = probs / probs.sum()
                    chosen_local = np.random.choice(
                        len(remainder_indices), size=needed, p=probs, replace=True
                    )
                    chosen_indices = remainder_indices[chosen_local]
                else:
                    # Fallback to uniform if charge is 0
                    chosen_indices = np.random.choice(
                        remainder_indices, size=needed, replace=True
                    )

                selected_indices = np.concatenate([keep_indices, chosen_indices])

            # Extract sampled features
            s_time = e_time[selected_indices]
            s_charge = e_charge[selected_indices]
            s_aux = e_aux[selected_indices]
            s_x = e_x[selected_indices]
            s_y = e_y[selected_indices]
            s_z = e_z[selected_indices]

            # --- Normalization ---

            # Time: Relative to the earliest pulse in the sample (or event min)
            # We use sample min to keep it self-contained
            t_ref = s_time.min()
            s_time = (s_time - t_ref) / Config.TIME_SCALE

            # Coordinates
            s_x = s_x / Config.COORD_SCALE
            s_y = s_y / Config.COORD_SCALE
            s_z = s_z / Config.COORD_SCALE

            # Charge: Log transform
            s_charge = np.log1p(s_charge)

            # Stack
            event_feats = np.stack([s_x, s_y, s_z, s_time, s_charge, s_aux], axis=1)

            # Fill output array (handles padding automatically since initialized to 0)
            actual_len = len(event_feats)
            features[i, :actual_len, :] = event_feats

        return features, None, unique_events

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return tensors
        x = torch.tensor(self.features[idx], dtype=torch.float32)

        if self.mode != "test" and self.targets is not None:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x, y
        else:
            # Return dummy target for test set
            return x, torch.zeros(2, dtype=torch.float32)
