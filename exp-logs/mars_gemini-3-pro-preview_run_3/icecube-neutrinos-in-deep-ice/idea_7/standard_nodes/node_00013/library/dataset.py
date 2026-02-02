import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from torch_geometric.data import Data, Dataset
from library.config import Config
from library.utils import load_sensor_geometry, compute_canonical_frame


class IceCubeDataset(Dataset):
    def __init__(self, mode="train", batch_ids=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            batch_ids (list): List of batch IDs to include. If None, uses all available in metadata.
        """
        # Initialize without root, transform, pre_transform
        super().__init__(root=None, transform=None, pre_transform=None)

        self.mode = mode
        self.cache_dir = Config.WORKING_DIR / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Load Metadata
        if mode == "train":
            meta_path = Config.TRAIN_METADATA_PATH
        elif mode == "val":
            meta_path = Config.VAL_METADATA_PATH
        elif mode == "test":
            meta_path = Config.TEST_METADATA_PATH
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.metadata = pd.read_parquet(meta_path)

        # Filter by batch_ids if provided
        if batch_ids is not None:
            self.metadata = self.metadata[
                self.metadata["batch_id"].isin(batch_ids)
            ].copy()

        # If DEBUG, limit size to a small subset
        if Config.DEBUG:
            unique_batches = self.metadata["batch_id"].unique()
            # Take first 2 batches max
            limit_batches = unique_batches[:2]
            self.metadata = self.metadata[
                self.metadata["batch_id"].isin(limit_batches)
            ].copy()
            if len(self.metadata) > Config.DEBUG_SUBSET_SIZE:
                self.metadata = self.metadata.iloc[: Config.DEBUG_SUBSET_SIZE].copy()

        # Get unique batches to process
        self.batch_ids = sorted(self.metadata["batch_id"].unique())

        # Load Geometry
        self.sensor_geometry = load_sensor_geometry()

        # 2. Ensure Cache Exists for all required batches
        # We process them now to ensure __getitem__ is fast and thread-safe
        for bid in self.batch_ids:
            self._process_and_cache_batch(bid)

        # 3. Build Index Mapping
        # Sort metadata to ensure alignment with our batch lookups
        self.metadata = self.metadata.sort_values(["batch_id", "event_id"]).reset_index(
            drop=True
        )

        # Map batch_id to its start index in the metadata
        # This allows us to calculate local_idx = global_idx - batch_start_idx
        self.batch_starts = {}
        counts = self.metadata["batch_id"].value_counts().sort_index()

        current_idx = 0
        # Iterate over sorted batch_ids
        for bid in self.batch_ids:
            count = counts.get(bid, 0)
            self.batch_starts[bid] = current_idx
            current_idx += count

        self.total_events = len(self.metadata)

        # Cache for memory-mapped arrays
        # key: batch_id, value: (X_mmap, I_mmap, Y_mmap)
        self.mem_maps = {}

    def _process_and_cache_batch(self, batch_id):
        """
        Processes a single batch and saves it to disk as .npy files.
        """
        # Define filenames
        f_X = self.cache_dir / f"{self.mode}_batch_{batch_id}_X.npy"
        f_I = self.cache_dir / f"{self.mode}_batch_{batch_id}_ids.npy"
        f_Y = self.cache_dir / f"{self.mode}_batch_{batch_id}_y.npy"

        # Check if exists
        if f_X.exists() and f_I.exists() and (self.mode == "test" or f_Y.exists()):
            return

        print(f"Processing batch {batch_id}...")

        # Load Raw Data
        dir_name = "train" if self.mode == "val" else self.mode
        batch_file = Config.INPUT_DIR / dir_name / f"batch_{batch_id}.parquet"
        df_batch = pd.read_parquet(batch_file)

        if "event_id" not in df_batch.columns:
            df_batch = df_batch.reset_index()

        unique_events = sorted(df_batch["event_id"].unique())

        # Get targets for this batch from metadata if available
        if self.mode != "test":
            batch_meta = self.metadata[self.metadata["batch_id"] == batch_id].set_index(
                "event_id"
            )
            valid_events = set(batch_meta.index)
        else:
            valid_events = set(unique_events)

        # Pre-allocate lists
        all_features = []
        all_indices = []
        all_targets = []

        grouped = df_batch.groupby("event_id")
        current_offset = 0

        for eid in unique_events:
            if eid not in valid_events:
                continue

            group = grouped.get_group(eid)

            # Extract Raw Data
            s_idx = group["sensor_id"].values
            time = group["time"].values.astype(np.float32)
            charge = group["charge"].values.astype(np.float32)
            aux = group["auxiliary"].values.astype(np.float32)

            # Geometry lookup
            pos = self.sensor_geometry[s_idx]  # (N, 3)
            x_raw, y_raw, z_raw = pos[:, 0], pos[:, 1], pos[:, 2]

            # Hybrid Sampling
            n_pulses = len(time)
            if n_pulses > Config.MAX_PULSES:
                k_charge = Config.MAX_PULSES // 2
                k_time = Config.MAX_PULSES - k_charge

                # Top Charge
                idx_charge = np.argpartition(-charge, k_charge)[:k_charge]
                # Early Time
                idx_time = np.argpartition(time, k_time)[:k_time]

                keep_idx = np.unique(np.concatenate([idx_charge, idx_time]))

                if len(keep_idx) > Config.MAX_PULSES:
                    sub_charge = charge[keep_idx]
                    sub_top = np.argpartition(-sub_charge, Config.MAX_PULSES)[
                        : Config.MAX_PULSES
                    ]
                    keep_idx = keep_idx[sub_top]

                x_raw = x_raw[keep_idx]
                y_raw = y_raw[keep_idx]
                z_raw = z_raw[keep_idx]
                time = time[keep_idx]
                charge = charge[keep_idx]
                aux = aux[keep_idx]

            # Canonical Frame
            R = compute_canonical_frame(x_raw, y_raw, z_raw, time, charge)

            # Apply Rotation
            raw_pos_stack = np.stack([x_raw, y_raw, z_raw], axis=1)
            can_pos = raw_pos_stack @ R.T

            # Normalization
            x_norm = x_raw / Config.POS_SCALE
            y_norm = y_raw / Config.POS_SCALE
            z_norm = z_raw / Config.POS_SCALE

            t_min = np.min(time)
            t_norm = (time - t_min) / Config.TIME_SCALE

            cx_norm = can_pos[:, 0] / Config.POS_SCALE
            cy_norm = can_pos[:, 1] / Config.POS_SCALE
            cz_norm = can_pos[:, 2] / Config.POS_SCALE

            q_norm = np.log10(charge + 0.1) / Config.CHARGE_SCALE

            # Stack Features: [x, y, z, t, cx, cy, cz, q, aux]
            features = np.stack(
                [
                    x_norm,
                    y_norm,
                    z_norm,
                    t_norm,
                    cx_norm,
                    cy_norm,
                    cz_norm,
                    q_norm,
                    aux,
                ],
                axis=1,
            ).astype(np.float32)

            all_features.append(features)

            count = len(features)
            all_indices.append([current_offset, count])
            current_offset += count

            if self.mode != "test":
                row = batch_meta.loc[eid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                all_targets.append([row["azimuth"], row["zenith"]])

        # Save to disk
        if not all_features:
            X_arr = np.zeros((0, Config.IN_CHANNELS), dtype=np.float32)
            I_arr = np.zeros((0, 2), dtype=np.int32)
            Y_arr = np.zeros((0, 2), dtype=np.float32)
        else:
            X_arr = np.concatenate(all_features, axis=0)
            I_arr = np.array(all_indices, dtype=np.int32)
            if self.mode != "test":
                Y_arr = np.array(all_targets, dtype=np.float32)

        np.save(f_X, X_arr)
        np.save(f_I, I_arr)
        if self.mode != "test":
            np.save(f_Y, Y_arr)

    def _get_batch_mmap(self, batch_id):
        if batch_id not in self.mem_maps:
            f_X = self.cache_dir / f"{self.mode}_batch_{batch_id}_X.npy"
            f_I = self.cache_dir / f"{self.mode}_batch_{batch_id}_ids.npy"

            X_mmap = np.load(f_X, mmap_mode="r")
            I_mmap = np.load(f_I, mmap_mode="r")

            Y_mmap = None
            if self.mode != "test":
                f_Y = self.cache_dir / f"{self.mode}_batch_{batch_id}_y.npy"
                Y_mmap = np.load(f_Y, mmap_mode="r")

            self.mem_maps[batch_id] = (X_mmap, I_mmap, Y_mmap)

        return self.mem_maps[batch_id]

    def len(self):
        return self.total_events

    def get(self, idx):
        # Retrieve metadata row
        row = self.metadata.iloc[idx]
        batch_id = int(row["batch_id"])

        # Calculate local index
        batch_start = self.batch_starts[batch_id]
        local_idx = idx - batch_start

        # Retrieve Data
        X_mmap, I_mmap, Y_mmap = self._get_batch_mmap(batch_id)

        start, count = I_mmap[local_idx]

        # Load features into memory
        x = torch.from_numpy(np.array(X_mmap[start : start + count]))

        if self.mode != "test":
            y = torch.from_numpy(np.array(Y_mmap[local_idx])).unsqueeze(0)
        else:
            y = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)

        return Data(x=x, y=y)
