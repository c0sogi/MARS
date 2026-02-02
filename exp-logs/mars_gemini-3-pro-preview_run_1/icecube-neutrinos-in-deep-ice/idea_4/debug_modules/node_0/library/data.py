import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from library.config import Config
from library.utils import azimuth_zenith_to_vector


class IceCubeGraphDataset(IterableDataset):
    """
    Iterable Dataset for Neutrino Direction Prediction.

    Processes raw IceCube pulses into graph structures suitable for GNNs.
    Handles caching of processed batches to disk to speed up training.
    """

    def __init__(self, mode="train", batch_ids=None):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            batch_ids (list, optional): List of batch IDs to load. If None, loads all for the mode.
        """
        super().__init__()
        self.mode = mode
        self.config_hash = Config.get_config_hash()

        # 1. Load Metadata
        if mode == "train":
            self.meta_path = Config.TRAIN_META_PATH
        elif mode == "val":
            self.meta_path = Config.VAL_META_PATH
        elif mode == "test":
            self.meta_path = Config.TEST_META_PATH
        else:
            raise ValueError("Mode must be train, val, or test")

        self.meta_df = pd.read_parquet(self.meta_path)

        # Filter specific batches if requested (e.g. for debugging)
        if batch_ids is not None:
            self.meta_df = self.meta_df[self.meta_df["batch_id"].isin(batch_ids)]

        self.batch_ids = self.meta_df["batch_id"].unique()

        # 2. Load Geometry
        self.geo_df = pd.read_csv(Config.SENSOR_GEO_PATH)
        self.sensor_map = self.geo_df.set_index("sensor_id")[["x", "y", "z"]]

        # 3. Setup Cache
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def process_batch(self, batch_id):
        """
        Loads a raw parquet batch, processes it into graph features, and caches it as .npz.

        Args:
            batch_id (int): The ID of the batch to process.

        Returns:
            dict-like: NpzFile object or dict containing processed arrays.
        """
        # Define unique cache filename based on config hash and mode
        cache_filename = f"batch_{batch_id}_{self.mode}_{self.config_hash}.npz"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        # Check cache
        if os.path.exists(cache_path):
            try:
                # Load with allow_pickle=True is default, but we only stored arrays so it's safe/compliant
                return np.load(cache_path)
            except Exception:
                pass  # If file is corrupt, re-process

        # --- Processing Logic ---

        # 1. Load Raw Data
        batch_file = f"batch_{batch_id}.parquet"
        # Val data is physically located in the 'train' folder
        folder = "test" if self.mode == "test" else "train"
        file_path = os.path.join(Config.INPUT_DIR, folder, batch_file)

        if not os.path.exists(file_path):
            return None

        df_batch = pd.read_parquet(file_path)

        # 2. Filter Auxiliary Pulses
        if Config.FILTER_AUXILIARY:
            df_batch = df_batch[~df_batch["auxiliary"]]

        # 3. Merge Geometry
        # Ensure event_id is a column
        if "event_id" not in df_batch.columns:
            df_batch = df_batch.reset_index()

        df_batch = df_batch.merge(self.sensor_map, on="sensor_id", how="left")

        # 4. Filter Events for this Split
        # Get target info for this batch from metadata
        batch_meta = self.meta_df[self.meta_df["batch_id"] == batch_id]
        valid_events = set(batch_meta["event_id"])

        # Keep only events belonging to this split (train vs val)
        df_batch = df_batch[df_batch["event_id"].isin(valid_events)]

        if len(df_batch) == 0:
            return None

        # 5. Prepare Target Map
        if self.mode != "test":
            target_map = batch_meta.set_index("event_id")[
                ["azimuth", "zenith"]
            ].to_dict("index")

        # 6. Efficient Grouping & Processing
        # Sort by event_id to enable sequential slicing
        df_batch = df_batch.sort_values(["event_id"])

        # Extract columns to numpy for speed
        evt_ids_col = df_batch["event_id"].values
        charges_col = df_batch["charge"].values
        times_col = df_batch["time"].values
        xs_col = df_batch["x"].values
        ys_col = df_batch["y"].values
        zs_col = df_batch["z"].values

        # Identify event boundaries
        unique_evts, indices = np.unique(evt_ids_col, return_index=True)
        starts = indices
        ends = np.append(starts[1:], len(evt_ids_col))

        # Collectors
        feature_list = []
        count_list = []
        target_list = []
        event_id_list = []

        for i, event_id in enumerate(unique_evts):
            start, end = starts[i], ends[i]

            # Slice event data
            e_charge = charges_col[start:end]
            e_time = times_col[start:end]
            e_x = xs_col[start:end]
            e_y = ys_col[start:end]
            e_z = zs_col[start:end]

            num_pulses = end - start
            if num_pulses == 0:
                continue

            # Sampling: Prioritize high charge
            if num_pulses > Config.MAX_PULSES:
                # Get indices of top k charges
                # Note: argpartition puts top k at the end, need to slice carefully
                # We use -e_charge to get largest values
                top_k_idx = np.argpartition(-e_charge, Config.MAX_PULSES - 1)[
                    : Config.MAX_PULSES
                ]

                e_charge = e_charge[top_k_idx]
                e_time = e_time[top_k_idx]
                e_x = e_x[top_k_idx]
                e_y = e_y[top_k_idx]
                e_z = e_z[top_k_idx]

            # Normalization
            # Time: Relative to earliest pulse in selection
            t_min = e_time.min()
            norm_time = (e_time - t_min) / Config.NORM_TIME_SCALE

            # Position: Scaled
            norm_x = e_x / Config.NORM_POS_SCALE
            norm_y = e_y / Config.NORM_POS_SCALE
            norm_z = e_z / Config.NORM_POS_SCALE

            # Charge: Log10
            # Clip to avoid log(0) issues, though min charge is usually ~0.025
            e_charge = np.maximum(e_charge, 1e-6)
            norm_charge = np.log10(e_charge)

            # Stack features: [x, y, z, t, logq]
            features = np.stack(
                [norm_x, norm_y, norm_z, norm_time, norm_charge], axis=1
            )

            feature_list.append(features)
            count_list.append(len(features))
            event_id_list.append(event_id)

            # Targets
            if self.mode != "test":
                t = target_map[event_id]
                target_list.append([t["azimuth"], t["zenith"]])
            else:
                target_list.append([0.0, 0.0])  # Dummy

        # Concatenate all
        if not feature_list:
            return None

        all_features = np.concatenate(feature_list, axis=0).astype(np.float32)
        counts = np.array(count_list, dtype=np.int32)
        targets = np.array(target_list, dtype=np.float32)
        event_ids_arr = np.array(event_id_list, dtype=np.int64)

        # Save to cache (No Pickle used for objects, just arrays)
        np.savez(
            cache_path,
            features=all_features,
            counts=counts,
            targets=targets,
            event_ids=event_ids_arr,
        )

        return np.load(cache_path)

    def __iter__(self):
        """
        Iterates over batches, loads processed data, and yields Data objects.
        Handles multi-worker splitting.
        """
        worker_info = get_worker_info()
        batch_ids = self.batch_ids.copy()

        # Split workload among workers
        if worker_info is not None:
            per_worker = int(np.ceil(len(batch_ids) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(batch_ids))
            batch_ids = batch_ids[iter_start:iter_end]

        # Shuffle batch order for training
        if self.mode == "train":
            np.random.shuffle(batch_ids)

        for batch_id in batch_ids:
            # Load or process batch
            data_npz = self.process_batch(batch_id)
            if data_npz is None:
                continue

            # Unpack arrays
            features = data_npz["features"]
            counts = data_npz["counts"]
            targets = data_npz["targets"]
            event_ids = data_npz["event_ids"]

            cursor = 0
            for i in range(len(counts)):
                n = counts[i]

                # Extract event features
                x = torch.from_numpy(features[cursor : cursor + n])

                # Prepare target
                az, zen = targets[i][0], targets[i][1]

                if self.mode != "test":
                    # Convert spherical to 3D Cartesian vector for loss
                    y_vec = azimuth_zenith_to_vector(az, zen)  # returns np array
                    y = torch.from_numpy(y_vec).float().unsqueeze(0)  # [1, 3]
                else:
                    y = torch.zeros((1, 3), dtype=torch.float32)

                # Create PyG Data object
                # event_id is stored for submission generation
                data = Data(
                    x=x, y=y, event_id=torch.tensor([event_ids[i]], dtype=torch.long)
                )

                yield data

                cursor += n

    def __len__(self):
        """Approximate length (number of events)."""
        return len(self.meta_df)
