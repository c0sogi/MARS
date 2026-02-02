import os
import gc
import pandas as pd
import numpy as np
from library.config import (
    INPUT_DIR,
    SENSOR_GEO_PATH,
    FEATURE_NAMES,
    TARGET_COLS_VECTOR,
    TARGET_COLS_ANGLES,
    DEBUG_SAMPLE_SIZE,
    SEED,
)
from library.utils import setup_logger, spherical_to_cartesian


class IceCubeFeatureGenerator:
    def __init__(self):
        self.logger = setup_logger("FeatureGenerator")
        self.geo_df = self._load_geometry()

    def _load_geometry(self):
        """Loads sensor geometry and prepares it for merging."""
        self.logger.info(f"Loading sensor geometry from {SENSOR_GEO_PATH}")
        geo_df = pd.read_csv(SENSOR_GEO_PATH)
        # Ensure sensor_id is available for merge.
        # Based on description, sensor_id is likely a column or we assume it matches sensor_id in pulses.
        # If sensor_id is not in columns, reset index assuming index is sensor_id.
        if "sensor_id" not in geo_df.columns:
            geo_df["sensor_id"] = geo_df.index

        return geo_df[["sensor_id", "x", "y", "z"]]

    def _compute_features_for_batch(self, batch_df):
        """
        Computes event-level features for a single batch of pulses using vectorized operations.

        Args:
            batch_df: DataFrame containing pulse data (event_id, time, charge, sensor_id, auxiliary).

        Returns:
            DataFrame with one row per event and engineered features.
        """
        # Merge with geometry
        # Note: batch_df usually has event_id as index or column. We ensure it's a column for grouping.
        if "event_id" not in batch_df.columns:
            batch_df = batch_df.reset_index()

        # Merge geometry
        batch_df = batch_df.merge(self.geo_df, on="sensor_id", how="left")

        # Pre-calculate weighted coordinates
        batch_df["wx"] = batch_df["x"] * batch_df["charge"]
        batch_df["wy"] = batch_df["y"] * batch_df["charge"]
        batch_df["wz"] = batch_df["z"] * batch_df["charge"]

        # ---------------------------------------------------------
        # 1. Basic Aggregations & Center of Gravity (CoG)
        # ---------------------------------------------------------
        # Group by event_id
        # We use sum for weighted coords and charge to calculate CoG
        aggs = {
            "charge": ["sum", "count"],
            "wx": "sum",
            "wy": "sum",
            "wz": "sum",
            "time": ["min", "max"],  # Duration
        }

        # Perform initial aggregation
        event_feats = batch_df.groupby("event_id").agg(aggs)

        # Flatten columns
        event_feats.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col
            for col in event_feats.columns.values
        ]

        # Rename for clarity
        event_feats = event_feats.rename(
            columns={
                "charge_sum": "total_charge",
                "charge_count": "n_pulses",
                "wx_sum": "sum_wx",
                "wy_sum": "sum_wy",
                "wz_sum": "sum_wz",
                "time_min": "t_min",
                "time_max": "t_max",
            }
        )

        # Calculate CoG
        # Avoid division by zero
        mask = event_feats["total_charge"] > 0
        event_feats["center_x"] = 0.0
        event_feats["center_y"] = 0.0
        event_feats["center_z"] = 0.0

        event_feats.loc[mask, "center_x"] = (
            event_feats.loc[mask, "sum_wx"] / event_feats.loc[mask, "total_charge"]
        )
        event_feats.loc[mask, "center_y"] = (
            event_feats.loc[mask, "sum_wy"] / event_feats.loc[mask, "total_charge"]
        )
        event_feats.loc[mask, "center_z"] = (
            event_feats.loc[mask, "sum_wz"] / event_feats.loc[mask, "total_charge"]
        )

        # Additional simple features
        event_feats["log_n_pulses"] = np.log1p(event_feats["n_pulses"])
        event_feats["time_duration"] = event_feats["t_max"] - event_feats["t_min"]

        # ---------------------------------------------------------
        # 2. Spread and Covariance (Vectorized)
        # ---------------------------------------------------------
        # We need to compute weighted variance/covariance.
        # Var(X) = (Sum w*(x - mu_x)^2) / Sum w
        # Cov(X,Y) = (Sum w*(x - mu_x)*(y - mu_y)) / Sum w

        # Map event-level means back to pulse level
        # batch_df is large, so we do this carefully

        # Create a mapping series
        # Note: 'event_id' is the index of event_feats
        mu_x = event_feats["center_x"]
        mu_y = event_feats["center_y"]
        mu_z = event_feats["center_z"]

        # Map to original dataframe
        batch_df = batch_df.set_index("event_id")  # Ensure index alignment
        batch_df["mu_x"] = mu_x
        batch_df["mu_y"] = mu_y
        batch_df["mu_z"] = mu_z
        batch_df = batch_df.reset_index()  # Restore event_id column

        # Calculate residuals
        dx = batch_df["x"] - batch_df["mu_x"]
        dy = batch_df["y"] - batch_df["mu_y"]
        dz = batch_df["z"] - batch_df["mu_z"]
        q = batch_df["charge"]

        # Weighted products
        # We only need upper triangle for covariance matrix
        batch_df["w_dx_dx"] = q * dx * dx
        batch_df["w_dy_dy"] = q * dy * dy
        batch_df["w_dz_dz"] = q * dz * dz
        batch_df["w_dx_dy"] = q * dx * dy
        batch_df["w_dx_dz"] = q * dx * dz
        batch_df["w_dy_dz"] = q * dy * dz

        # Aggregate residuals
        cov_aggs = {
            "w_dx_dx": "sum",
            "w_dy_dy": "sum",
            "w_dz_dz": "sum",
            "w_dx_dy": "sum",
            "w_dx_dz": "sum",
            "w_dy_dz": "sum",
        }
        cov_sums = batch_df.groupby("event_id").agg(cov_aggs)

        # Normalize by total charge to get Covariance
        # Join with total_charge
        cov_sums = cov_sums.join(event_feats["total_charge"])

        mask_cov = cov_sums["total_charge"] > 0
        for col in ["w_dx_dx", "w_dy_dy", "w_dz_dz", "w_dx_dy", "w_dx_dz", "w_dy_dz"]:
            res_col = col.replace("w_", "cov_").replace(
                "_d", ""
            )  # e.g. w_dx_dy -> cov_xy
            # Handle diagonal (variance) -> spread (std)
            if col in ["w_dx_dx", "w_dy_dy", "w_dz_dz"]:
                var_col = col.replace("w_d", "cov_").replace(
                    "_d", ""
                )  # w_dx_dx -> cov_xx
                event_feats[var_col] = 0.0
                event_feats.loc[mask_cov, var_col] = (
                    cov_sums.loc[mask_cov, col] / cov_sums.loc[mask_cov, "total_charge"]
                )

                # Spread is sqrt(variance)
                spread_col = var_col.replace("cov_", "spread_").replace(
                    var_col[-1], ""
                )  # cov_xx -> spread_x
                event_feats[spread_col] = np.sqrt(event_feats[var_col])
            else:
                cov_col = col.replace("w_d", "cov_").replace(
                    "_d", ""
                )  # w_dx_dy -> cov_xy
                event_feats[cov_col] = 0.0
                event_feats.loc[mask_cov, cov_col] = (
                    cov_sums.loc[mask_cov, col] / cov_sums.loc[mask_cov, "total_charge"]
                )

        # ---------------------------------------------------------
        # 3. Temporal Percentiles (Optimized)
        # ---------------------------------------------------------
        # Sorting approach is O(N log N) but N is batch size (~2M pulses), so it's fast enough.

        # Sort by event_id and time
        batch_sorted = batch_df[["event_id", "time"]].sort_values(["event_id", "time"])

        # Get counts per event to calculate indices
        counts = batch_sorted.groupby("event_id").size()
        # Ensure alignment with event_feats index
        counts = counts.reindex(event_feats.index).fillna(0).astype(int)

        # Calculate cumulative starts
        # We need to handle the fact that groupby might change order, but reindex fixed it.
        # However, we need the cumulative sum relative to the sorted dataframe.
        # Let's rely on the sorted dataframe's structure.

        # Re-calculate counts from the sorted frame to be safe about order
        counts_sorted = batch_sorted.groupby("event_id", sort=False).size()
        cumulative = counts_sorted.cumsum()
        starts = cumulative - counts_sorted

        # Calculate global indices for percentiles
        # 10th, 50th, 90th
        # We use floor/ceil logic or simple int casting.
        # idx = start + (count * percentile)

        # We need to map these back to the event_feats index.
        # Let's create a temporary DF for time feats
        time_feats = pd.DataFrame(index=counts_sorted.index)

        for p_name, p_val in [
            ("time_10th", 0.10),
            ("time_50th", 0.50),
            ("time_90th", 0.90),
        ]:
            # Calculate offset
            offsets = (counts_sorted * p_val).astype(int)
            # Clip offsets to be within [0, count-1]
            offsets = np.clip(offsets, 0, counts_sorted - 1)

            global_indices = starts + offsets

            # Extract times
            # iloc uses integer position
            values = batch_sorted["time"].iloc[global_indices].values
            time_feats[p_name] = values

        # Join time features
        event_feats = event_feats.join(time_feats)

        # Normalize time features relative to t_min (optional, but good for trees)
        # Or keep absolute? The prompt implies "relative to other pulses".
        # But "time" is already relative in the batch.
        # However, t_min varies. Let's make them relative to t_min of the event.
        for col in ["time_10th", "time_50th", "time_90th"]:
            event_feats[col] = event_feats[col] - event_feats["t_min"]

        # ---------------------------------------------------------
        # 4. Cleanup and Return
        # ---------------------------------------------------------
        # Select only requested features
        # Ensure all columns exist (fill 0 if missing due to empty events, though unlikely)
        for col in FEATURE_NAMES:
            if col not in event_feats.columns:
                event_feats[col] = 0.0

        return event_feats[FEATURE_NAMES]

    def process_split(self, meta_path, output_path, load_cached_data=True):
        """
        Main driver to process a dataset split (train/val/test).
        Handles caching, batch iteration, and target merging.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(output_path):
            self.logger.info(f"Loading cached features from {output_path}")
            return pd.read_parquet(output_path)

        self.logger.info(f"Processing data from metadata: {meta_path}")

        # 2. Load Metadata
        meta_df = pd.read_parquet(meta_path)

        # Debugging: Sample if requested
        if DEBUG_SAMPLE_SIZE is not None and len(meta_df) > DEBUG_SAMPLE_SIZE:
            self.logger.info(f"DEBUG: Sampling {DEBUG_SAMPLE_SIZE} events.")
            meta_df = meta_df.sample(n=DEBUG_SAMPLE_SIZE, random_state=SEED).copy()

        # 3. Iterate over batches
        batch_ids = meta_df["batch_id"].unique()
        feature_dfs = []

        # Pre-filter meta to speed up lookups
        meta_df = meta_df.set_index("event_id")

        self.logger.info(f"Found {len(batch_ids)} batches to process.")

        for batch_id in batch_ids:
            # Construct path (assuming standard structure relative to INPUT_DIR)
            # Metadata contains 'batch_file_path' usually, let's check
            # The provided metadata generation script adds 'batch_file_path'.
            # We can grab it from the first entry for this batch_id

            # Get subset of meta for this batch
            batch_meta = meta_df[meta_df["batch_id"] == batch_id]
            if batch_meta.empty:
                continue

            # Get file path from the first row of this batch in metadata
            rel_path = batch_meta["batch_file_path"].iloc[0]
            full_path = os.path.join(INPUT_DIR, rel_path)

            if not os.path.exists(full_path):
                self.logger.warning(f"Batch file not found: {full_path}. Skipping.")
                continue

            # Load batch pulses
            try:
                batch_pulses = pd.read_parquet(full_path)
            except Exception as e:
                self.logger.error(f"Error reading {full_path}: {e}")
                continue

            # Filter pulses to only those events in our (possibly sampled) metadata
            # This is crucial for Debug mode and to avoid processing events not in this split
            valid_events = batch_meta.index
            batch_pulses = batch_pulses[batch_pulses.index.isin(valid_events)]

            if batch_pulses.empty:
                continue

            # Compute Features
            batch_features = self._compute_features_for_batch(batch_pulses)

            # Append
            feature_dfs.append(batch_features)

            # Explicit GC
            del batch_pulses, batch_features
            gc.collect()

        # 4. Concatenate all features
        if not feature_dfs:
            raise ValueError("No features generated. Check input paths and metadata.")

        full_features = pd.concat(feature_dfs)

        # 5. Merge Targets (if available in metadata)
        # meta_df has 'azimuth' and 'zenith' for train/val
        # We join on index (event_id)

        # Columns to keep from metadata
        meta_cols = []
        if "azimuth" in meta_df.columns and "zenith" in meta_df.columns:
            meta_cols.extend(["azimuth", "zenith"])

        if meta_cols:
            full_features = full_features.join(meta_df[meta_cols], how="inner")

            # Generate Vector Targets
            tx, ty, tz = spherical_to_cartesian(
                full_features["azimuth"].values, full_features["zenith"].values
            )
            full_features["target_x"] = tx
            full_features["target_y"] = ty
            full_features["target_z"] = tz
        else:
            # For test set, ensure we keep the index (event_id) and order matches metadata if needed
            # But join is on index, so it's safe.
            pass

        # 6. Save to Cache
        self.logger.info(f"Saving generated features to {output_path}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_features.to_parquet(output_path)

        return full_features
