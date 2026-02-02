import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import WORKING_DIR, SMOOTHING_PARAMS
from library.data_processing import process_data


class SpatialTargetEncoder:
    """
    Implements Spatially-Smoothed Target Encoding.
    Discretizes coordinates into a grid and calculates historical average fares
    for each pickup->dropoff route.
    """

    def __init__(self, smoothing_params=SMOOTHING_PARAMS):
        self.k_folds = smoothing_params.get("k_folds", 5)
        self.smoothing = smoothing_params.get("smoothing", 20)
        self.min_samples = smoothing_params.get("min_samples_leaf", 5)
        self.global_mean = None
        self.stats_df = None
        # Paths for persisting the learned state (without pickle)
        self.stats_path = os.path.join(WORKING_DIR, "encoder_stats.parquet")
        self.global_mean_path = os.path.join(WORKING_DIR, "global_mean.npy")

    def _discretize(self, df):
        """
        Discretizes coordinates to create a spatial grid (approx 1.1km resolution).
        Uses integer rounding for memory efficiency and speed.
        Precision: 2 decimal places (x * 100).
        """
        # Create copies to avoid modifying original dataframe
        # Using int32 to save memory on 55M rows
        p_lat = (df["pickup_latitude"] * 100).round().astype(np.int32)
        p_lon = (df["pickup_longitude"] * 100).round().astype(np.int32)
        d_lat = (df["dropoff_latitude"] * 100).round().astype(np.int32)
        d_lon = (df["dropoff_longitude"] * 100).round().astype(np.int32)

        return p_lat, p_lon, d_lat, d_lon

    def fit(self, df, target_col="fare_amount"):
        """
        Fits the encoder on the full dataset.
        Calculates sum and count of target for each route.
        Saves the stats to disk for use during inference.
        """
        print("Fitting SpatialTargetEncoder on full dataset...")
        self.global_mean = df[target_col].mean()

        p_lat, p_lon, d_lat, d_lon = self._discretize(df)

        # Create a lightweight temporary dataframe for grouping
        temp_df = pd.DataFrame(
            {
                "p_lat": p_lat,
                "p_lon": p_lon,
                "d_lat": d_lat,
                "d_lon": d_lon,
                "target": df[target_col],
            }
        )

        # Group by route and calculate stats
        stats = (
            temp_df.groupby(["p_lat", "p_lon", "d_lat", "d_lon"])["target"]
            .agg(["sum", "count"])
            .reset_index()
        )

        self.stats_df = stats

        # Save state
        print(f"Saving encoder stats to {self.stats_path}...")
        self.stats_df.to_parquet(self.stats_path, index=False)
        np.save(self.global_mean_path, np.array([self.global_mean]))

    def load_state(self):
        """
        Loads the encoder state (stats and global mean) from disk.
        Returns True if successful, False otherwise.
        """
        if os.path.exists(self.stats_path) and os.path.exists(self.global_mean_path):
            print(f"Loading encoder stats from {self.stats_path}...")
            self.stats_df = pd.read_parquet(self.stats_path)
            self.global_mean = float(np.load(self.global_mean_path)[0])
            return True
        return False

    def transform(self, df):
        """
        Applies the learned encoding to a dataset (Validation/Test).
        Uses smoothed averages based on the loaded stats.
        """
        print("Transforming data with SpatialTargetEncoder...")
        if self.stats_df is None:
            if not self.load_state():
                raise ValueError(
                    "Encoder must be fitted or state must exist before transform."
                )

        p_lat, p_lon, d_lat, d_lon = self._discretize(df)

        # Prepare keys for merge
        df_keys = pd.DataFrame(
            {"p_lat": p_lat, "p_lon": p_lon, "d_lat": d_lat, "d_lon": d_lon}
        )
        # Preserve original order
        df_keys["original_index"] = np.arange(len(df))

        # Merge with stats
        # how='left' ensures we keep all rows from input df
        merged = df_keys.merge(
            self.stats_df, on=["p_lat", "p_lon", "d_lat", "d_lon"], how="left"
        )

        # Fill NaNs (unknown routes) with 0 for stats
        merged["sum"] = merged["sum"].fillna(0)
        merged["count"] = merged["count"].fillna(0)

        # Apply Bayesian Smoothing
        # Formula: (sum + m * global_mean) / (count + m)
        m = self.smoothing
        smoothed_fare = (merged["sum"] + m * self.global_mean) / (merged["count"] + m)

        # Sort back to original order
        merged["smoothed_fare"] = smoothed_fare
        merged = merged.sort_values("original_index")

        return merged["smoothed_fare"].values

    def fit_transform(self, df, target_col="fare_amount"):
        """
        Applies K-Fold Mean Encoding to the training set.
        Generates features for each fold using stats from the other folds
        to prevent data leakage.
        """
        print("Running K-Fold Target Encoding on Training Data...")
        self.global_mean = df[target_col].mean()

        # Initialize result array
        result = np.zeros(len(df), dtype=np.float32)

        # Fixed random state for reproducibility
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        p_lat, p_lon, d_lat, d_lon = self._discretize(df)

        # Lightweight dataframe for K-Fold operations
        work_df = pd.DataFrame(
            {
                "p_lat": p_lat,
                "p_lon": p_lon,
                "d_lat": d_lat,
                "d_lon": d_lon,
                "target": df[target_col],
            }
        )

        for fold, (train_idx, val_idx) in enumerate(kf.split(work_df)):
            print(f"Processing Fold {fold + 1}/{self.k_folds}...")

            # Train set for this fold (used to calculate stats)
            X_tr = work_df.iloc[train_idx]
            # Validation set for this fold (where we assign values)
            X_val = work_df.iloc[val_idx]

            # Compute stats on X_tr
            stats = (
                X_tr.groupby(["p_lat", "p_lon", "d_lat", "d_lon"])["target"]
                .agg(["sum", "count"])
                .reset_index()
            )

            # Map stats to X_val
            X_val_keys = X_val[["p_lat", "p_lon", "d_lat", "d_lon"]].copy()
            # Track indices to assign back to result array correctly
            X_val_keys["idx_in_result"] = val_idx

            merged = X_val_keys.merge(
                stats, on=["p_lat", "p_lon", "d_lat", "d_lon"], how="left"
            )
            merged["sum"] = merged["sum"].fillna(0)
            merged["count"] = merged["count"].fillna(0)

            m = self.smoothing
            smoothed = (merged["sum"] + m * self.global_mean) / (merged["count"] + m)

            # Assign values
            result[merged["idx_in_result"].values] = smoothed.values

        # After generating features for the training set via K-Fold,
        # we must fit on the FULL dataset to save the state for validation/test sets.
        self.fit(df, target_col)

        return result


def get_target_encoded_data(split_name, load_cached_data=True):
    """
    Orchestrates the feature engineering pipeline.
    Loads processed data, applies Spatial Target Encoding, and caches the result.
    """
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Path for the final file with engineered features
    final_path = os.path.join(WORKING_DIR, f"{split_name}_encoded.parquet")

    # 1. Try Load from Cache
    if load_cached_data and os.path.exists(final_path):
        print(f"Loading target-encoded data for {split_name} from {final_path}...")
        return pd.read_parquet(final_path)

    # 2. Compute
    print(f"Computing target-encoded data for {split_name}...")

    # Load base processed data (cleaned + physical features)
    df = process_data(split_name, load_cached_data=load_cached_data)

    encoder = SpatialTargetEncoder()

    if split_name == "train":
        # For training, we use fit_transform (K-Fold) to prevent leakage
        if "fare_amount" not in df.columns:
            raise ValueError("Training data missing 'fare_amount'")

        encoded_values = encoder.fit_transform(df, target_col="fare_amount")
        df["route_avg_fare"] = encoded_values

    else:
        # For val/test, we use transform (using stats learned from train)
        # If stats file doesn't exist, we must fit on train first.
        if not encoder.load_state():
            print("Encoder state not found. Fitting on Train data first...")
            # Load train data just to fit (we don't need the result, just the side effect of saving stats)
            train_df = process_data("train", load_cached_data=True)
            encoder.fit(train_df, target_col="fare_amount")
            del train_df

        encoded_values = encoder.transform(df)
        df["route_avg_fare"] = encoded_values

    # 3. Save to Cache
    print(f"Saving target-encoded data for {split_name} to {final_path}...")
    df.to_parquet(final_path, index=False)

    return df
