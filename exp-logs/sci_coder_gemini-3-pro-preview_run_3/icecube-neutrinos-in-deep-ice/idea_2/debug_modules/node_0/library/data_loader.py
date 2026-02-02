import pandas as pd
import numpy as np
import os
from pathlib import Path
from library.config import (
    INPUT_DIR,
    CACHE_DIR,
    SENSOR_GEOMETRY_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    FEATURE_NAMES,
    DEBUG_SAMPLE_SIZE,
)
from library.utils import spherical_to_cartesian


class NeutrinoDataLoader:
    def __init__(self):
        self.geometry = self._load_geometry()

    def _load_geometry(self):
        """
        Load sensor geometry and prepare it for merging.
        """
        if not SENSOR_GEOMETRY_PATH.exists():
            raise FileNotFoundError(
                f"Sensor geometry file not found at {SENSOR_GEOMETRY_PATH}"
            )

        df = pd.read_csv(SENSOR_GEOMETRY_PATH)

        # If sensor_id is a column, set it as index.
        # Otherwise assume row index corresponds to sensor_id (0 to 5159).
        if "sensor_id" in df.columns:
            df = df.set_index("sensor_id")

        return df[["x", "y", "z"]].astype(np.float32)

    def _process_batch(self, batch_rel_path, meta_batch_df):
        """
        Load a raw batch file, merge geometry, and calculate aggregate features.
        """
        full_path = INPUT_DIR / batch_rel_path
        if not full_path.exists():
            print(f"Warning: {full_path} not found. Skipping.")
            return pd.DataFrame()

        # Load raw pulses
        pulses = pd.read_parquet(full_path)

        # Filter to events present in the metadata (important for debug/subsets)
        valid_events = meta_batch_df["event_id"].unique()
        pulses = pulses[pulses["event_id"].isin(valid_events)]

        if pulses.empty:
            return pd.DataFrame()

        # Merge geometry (x, y, z)
        # Using left_on='sensor_id' and right_index=True for efficiency
        pulses = pulses.merge(
            self.geometry, left_on="sensor_id", right_index=True, how="left"
        )

        # --- Feature Engineering ---
        # Pre-compute terms for weighted statistics
        pulses["wx"] = pulses["x"] * pulses["charge"]
        pulses["wy"] = pulses["y"] * pulses["charge"]
        pulses["wz"] = pulses["z"] * pulses["charge"]

        pulses["wx2"] = pulses["x"] ** 2 * pulses["charge"]
        pulses["wy2"] = pulses["y"] ** 2 * pulses["charge"]
        pulses["wz2"] = pulses["z"] ** 2 * pulses["charge"]

        # Define aggregations
        # Must match logic required to generate FEATURE_NAMES
        aggs = {
            "charge": ["count", "sum", "mean", "std", "max"],
            "auxiliary": ["mean"],
            "time": ["min", "max", "mean", "std"],
            "x": ["mean", "std"],
            "y": ["mean", "std"],
            "z": ["mean", "std"],
            "wx": ["sum"],
            "wy": ["sum"],
            "wz": ["sum"],
            "wx2": ["sum"],
            "wy2": ["sum"],
            "wz2": ["sum"],
        }

        # Group by event_id
        df_agg = pulses.groupby("event_id").agg(aggs)

        # Flatten MultiIndex columns (e.g., ('charge', 'sum') -> 'charge_sum')
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]

        # Rename to match FEATURE_NAMES
        rename_map = {
            "charge_count": "n_pulses",
            "auxiliary_mean": "aux_ratio",
        }
        df_agg = df_agg.rename(columns=rename_map)

        # Derived Features
        # 1. Time Duration
        df_agg["time_duration"] = df_agg["time_max"] - df_agg["time_min"]

        # 2. Weighted Means (Center of Mass)
        # Avoid division by zero by replacing 0 charge_sum with 1 (though charge_sum should be > 0)
        safe_charge_sum = df_agg["charge_sum"].replace(0, 1.0)

        df_agg["x_w_mean"] = df_agg["wx_sum"] / safe_charge_sum
        df_agg["y_w_mean"] = df_agg["wy_sum"] / safe_charge_sum
        df_agg["z_w_mean"] = df_agg["wz_sum"] / safe_charge_sum

        # 3. Weighted Standard Deviations
        # Var = E[x^2] - (E[x])^2
        var_x = (df_agg["wx2_sum"] / safe_charge_sum) - df_agg["x_w_mean"] ** 2
        var_y = (df_agg["wy2_sum"] / safe_charge_sum) - df_agg["y_w_mean"] ** 2
        var_z = (df_agg["wz2_sum"] / safe_charge_sum) - df_agg["z_w_mean"] ** 2

        # Clip negative variance (floating point errors)
        df_agg["x_w_std"] = np.sqrt(var_x.clip(lower=0))
        df_agg["y_w_std"] = np.sqrt(var_y.clip(lower=0))
        df_agg["z_w_std"] = np.sqrt(var_z.clip(lower=0))

        # Ensure all requested features exist and fill NaNs
        for col in FEATURE_NAMES:
            if col not in df_agg.columns:
                df_agg[col] = 0.0

        # Select final columns in correct order, fill NaNs (e.g. std of 1 pulse)
        df_agg = df_agg[FEATURE_NAMES].fillna(0).astype(np.float32)

        return df_agg

    def load_split(self, split, load_cached_data=True):
        """
        Load dataset for a specific split (train, val, test).
        Returns X (features), y (targets), ids (event_ids).
        """
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Define cache file paths
        cache_X_path = CACHE_DIR / f"{split}_X.npy"
        cache_y_path = CACHE_DIR / f"{split}_y.npy"
        cache_ids_path = CACHE_DIR / f"{split}_ids.npy"

        # 1. Try loading monolithic cache
        if load_cached_data and cache_X_path.exists() and cache_ids_path.exists():
            # For test set, y might not exist
            if split == "test" or cache_y_path.exists():
                print(f"Loading {split} set from cache...")
                X = np.load(cache_X_path)
                ids = np.load(cache_ids_path)
                y = np.load(cache_y_path) if split != "test" else None
                return X, y, ids

        print(f"Generating {split} set from scratch...")

        # 2. Load Metadata
        if split == "train":
            meta_path = TRAIN_META_PATH
        elif split == "val":
            meta_path = VAL_META_PATH
        else:
            meta_path = TEST_META_PATH

        meta_df = pd.read_parquet(meta_path)

        # Apply Debug Sampling
        if DEBUG_SAMPLE_SIZE is not None:
            print(f"Debug Mode: Sampling first {DEBUG_SAMPLE_SIZE} events.")
            meta_df = meta_df.iloc[:DEBUG_SAMPLE_SIZE]

        # 3. Process Batches
        batch_ids = sorted(meta_df["batch_id"].unique())

        X_list = []
        y_list = []
        ids_list = []

        for batch_id in batch_ids:
            # Define batch-level cache
            batch_cache_path = CACHE_DIR / f"{split}_batch_{batch_id}_features.parquet"

            # Subset metadata for this batch
            batch_meta = meta_df[meta_df["batch_id"] == batch_id]

            # Check if batch is cached
            if load_cached_data and batch_cache_path.exists():
                batch_features = pd.read_parquet(batch_cache_path)
                # Filter to ensure it matches current metadata subset
                batch_features = batch_features.loc[
                    batch_features.index.isin(batch_meta["event_id"])
                ]
            else:
                # Process raw batch
                rel_path = batch_meta.iloc[0]["file_path"]
                batch_features = self._process_batch(rel_path, batch_meta)

                # Save batch cache
                batch_features.to_parquet(batch_cache_path)

            # Align features with metadata
            # Reindex ensures we have rows for all events in metadata, in correct order
            batch_features = batch_features.reindex(batch_meta["event_id"]).fillna(0)

            X_list.append(batch_features.values)
            ids_list.append(batch_meta["event_id"].values)

            # Compute Targets for Train/Val
            if split != "test":
                azimuth = batch_meta["azimuth"].values
                zenith = batch_meta["zenith"].values
                tx, ty, tz = spherical_to_cartesian(azimuth, zenith)
                targets = np.stack([tx, ty, tz], axis=1)
                y_list.append(targets)

        # 4. Concatenate and Save Monolithic Cache
        if not X_list:
            raise ValueError(f"No data found for split {split}")

        X = np.vstack(X_list).astype(np.float32)
        ids = np.concatenate(ids_list).astype(np.int64)

        np.save(cache_X_path, X)
        np.save(cache_ids_path, ids)

        if split != "test":
            y = np.vstack(y_list).astype(np.float32)
            np.save(cache_y_path, y)
        else:
            y = None

        print(f"Finished processing {split} set. Shape: {X.shape}")
        return X, y, ids
