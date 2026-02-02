import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_sensor_geometry, get_config_hash


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for IceCube Neutrino Direction Prediction.
    Implements Stratified Causal-Signal Sampling and robust caching.
    """

    def __init__(self, mode="train", transform=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.transform = transform

        # 1. Load Metadata
        if mode == "train":
            self.meta_path = Config.TRAIN_META_PATH
        elif mode == "val":
            self.meta_path = Config.VAL_META_PATH
        elif mode == "test":
            self.meta_path = Config.TEST_META_PATH
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.meta = pd.read_parquet(self.meta_path)

        # 2. Load Sensor Geometry
        # Returns dataframe indexed by sensor_id with columns x, y, z
        self.geo_df = load_sensor_geometry(Config.SENSOR_GEO_PATH)

        # 3. Setup Caching
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.config_hash = Config.get_config_hash()

        # 4. Runtime Batch Cache (Simple Memory Buffer)
        # Keeps the data of the currently accessed batch in memory to minimize disk I/O
        # when accessing sequential events.
        self.last_batch_id = -1
        self.last_batch_features = {}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Get event info from metadata
        row = self.meta.iloc[idx]
        batch_id = int(row["batch_id"])
        event_id = int(row["event_id"])

        # Check memory cache
        if batch_id != self.last_batch_id:
            self.last_batch_features = self._load_batch(batch_id)
            self.last_batch_id = batch_id

        # Retrieve features
        # Shape: (N_PULSES, IN_CHANNELS)
        features = self.last_batch_features.get(event_id)

        if features is None:
            # Fallback: This might happen if the cache file exists but is missing this specific event
            # (e.g. due to split logic changes). Force reprocess.
            self.last_batch_features = self._load_batch(batch_id, force_process=True)
            features = self.last_batch_features.get(event_id)
            if features is None:
                raise KeyError(
                    f"Event {event_id} not found in batch {batch_id} after processing."
                )

        # Prepare Tensors
        x = torch.tensor(features, dtype=torch.float32)

        # Prepare Target
        if self.mode in ["train", "val"]:
            y = torch.tensor([row["azimuth"], row["zenith"]], dtype=torch.float32)
        else:
            # Dummy target for test set
            y = torch.tensor([0.0, 0.0], dtype=torch.float32)

        sample = {"x": x, "y": y, "event_id": event_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_batch(self, batch_id, force_process=False):
        """
        Loads a batch from disk cache. If missing or forced, processes it from raw parquet.
        Returns a dictionary mapping event_id -> features array.
        """
        # Define cache filename
        # We include mode in filename because train/val split logic might separate events
        filename = f"batch_{batch_id}_{self.mode}_{self.config_hash}.npz"
        filepath = os.path.join(self.cache_dir, filename)

        if not force_process and os.path.exists(filepath):
            try:
                data = np.load(filepath)
                # Reconstruct dict: event_id -> features
                ids = data["ids"]
                feats = data["features"]
                return dict(zip(ids, feats))
            except Exception as e:
                print(f"Error loading cache {filepath}: {e}. Reprocessing.")

        return self._process_batch(batch_id, filepath)

    def _process_batch(self, batch_id, save_path):
        """
        Reads raw parquet, applies Stratified Causal-Signal Sampling, and saves to disk.
        """
        # 1. Identify raw file path from metadata
        subset = self.meta[self.meta["batch_id"] == batch_id]
        if subset.empty:
            raise ValueError(f"Batch {batch_id} not found in metadata.")

        rel_path = subset.iloc[0]["file_path"]
        raw_path = os.path.join(Config.INPUT_DIR, rel_path)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw batch file not found: {raw_path}")

        # 2. Load Raw Data
        # Columns: event_id (index), time, sensor_id, charge, auxiliary
        df = pd.read_parquet(raw_path)

        # Ensure event_id is a column
        if df.index.name == "event_id":
            df = df.reset_index()
        elif "event_id" not in df.columns:
            # Attempt to infer if index is event_id
            df["event_id"] = df.index
            df = df.reset_index(drop=True)

        # Filter to only events in this split (train/val/test)
        valid_ids = set(subset["event_id"].unique())
        df = df[df["event_id"].isin(valid_ids)]

        # 3. Merge Geometry
        # self.geo_df has index sensor_id, cols x, y, z
        df = df.merge(self.geo_df, on="sensor_id", how="left")

        # 4. Group and Process
        grouped = df.groupby("event_id")

        processed_ids = []
        processed_feats = []

        # Pre-compute constants
        N = Config.N_PULSES
        K = Config.K_CAUSAL
        M = Config.M_SIGNAL

        for eid, group in grouped:
            # Deterministic RNG per event for context sampling
            rng = np.random.default_rng(Config.SEED + eid)

            # Sort by time
            group = group.sort_values("time")

            # Indices
            indices = np.arange(len(group))

            # --- Stratified Sampling ---

            # 1. Causal Set (First K by time)
            k_actual = min(len(indices), K)
            idx_causal = indices[:k_actual]
            rem_indices = indices[k_actual:]

            # 2. Signal Set (Top M by charge from remainder)
            if len(rem_indices) > 0:
                rem_charges = group.iloc[rem_indices]["charge"].values

                if len(rem_indices) > M:
                    # argpartition is faster than argsort for top-k
                    # We want indices of the largest M charges
                    top_m_local = np.argpartition(rem_charges, -M)[-M:]
                    idx_signal = rem_indices[top_m_local]

                    # Remove signal from pool for context
                    mask = np.ones(len(rem_indices), dtype=bool)
                    mask[top_m_local] = False
                    pool_indices = rem_indices[mask]
                else:
                    idx_signal = rem_indices
                    pool_indices = np.array([], dtype=int)
            else:
                idx_signal = np.array([], dtype=int)
                pool_indices = np.array([], dtype=int)

            # 3. Context Set (Random sample from remainder)
            n_current = len(idx_causal) + len(idx_signal)
            n_needed = N - n_current

            if n_needed > 0 and len(pool_indices) > 0:
                if len(pool_indices) <= n_needed:
                    idx_context = pool_indices
                else:
                    idx_context = rng.choice(pool_indices, size=n_needed, replace=False)
            else:
                idx_context = np.array([], dtype=int)

            # Combine and re-sort by time
            final_idx = np.concatenate([idx_causal, idx_signal, idx_context])
            final_idx.sort()

            # Extract Data
            sel = group.iloc[final_idx]

            # --- Feature Engineering & Normalization ---

            # Coordinates: Scale by 500m (approx detector radius)
            x = sel["x"].values / 500.0
            y = sel["y"].values / 500.0
            z = sel["z"].values / 500.0

            # Time: Relative to event start, scaled to microseconds
            t_raw = sel["time"].values
            t_min = group["time"].min()
            t = (t_raw - t_min) / 1000.0

            # Charge: Log10 transform
            q = sel["charge"].values
            log_q = np.log10(np.maximum(q, 0.01))  # Clip to avoid log(0)

            # Auxiliary: Cast to float
            aux = sel["auxiliary"].values.astype(np.float32)

            # Stack features: [x, y, z, time, log_charge, auxiliary]
            feat = np.stack([x, y, z, t, log_q, aux], axis=1).astype(np.float32)

            # Padding if fewer than N pulses
            if len(feat) < N:
                pad_size = N - len(feat)
                # Pad with zeros
                padding = np.zeros((pad_size, Config.IN_CHANNELS), dtype=np.float32)
                # Set log_charge to -5.0 to indicate silence/no-pulse
                padding[:, 4] = -5.0
                feat = np.concatenate([feat, padding], axis=0)

            processed_ids.append(eid)
            processed_feats.append(feat)

        # Convert to arrays
        processed_ids = np.array(processed_ids)
        processed_feats = np.array(processed_feats)  # Shape: (B, N, 6)

        # Save compressed to disk
        np.savez_compressed(save_path, ids=processed_ids, features=processed_feats)

        return dict(zip(processed_ids, processed_feats))


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Stacks input features and targets.
    """
    x = torch.stack([item["x"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    event_ids = [item["event_id"] for item in batch]

    return {"x": x, "y": y, "event_ids": event_ids}
