import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from library import config


class GlobalRouteEncoder:
    """
    Implements Global-Prior Augmented feature engineering.
    Aggregates statistics from the full dataset and injects them into subsamples
    using Vectorized Subtraction to prevent leakage.
    """

    def __init__(self):
        self.global_stats = None
        self.global_mean = None
        self.coord_cols = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ]
        # Cache paths
        self.stats_path = os.path.join(config.WORKING_DIR, "global_route_stats.parquet")
        self.mean_path = os.path.join(config.WORKING_DIR, "global_mean.npy")

    def fit(self, full_train_df, load_cached_data=True):
        """
        Computes global sum and count of fare_amount for every unique route
        in the full dataset.

        Args:
            full_train_df (pd.DataFrame): The complete 55M row dataset (preprocessed).
            load_cached_data (bool): Whether to attempt loading from disk.
        """
        os.makedirs(config.WORKING_DIR, exist_ok=True)

        # 1. Attempt to load from cache
        if (
            load_cached_data
            and os.path.exists(self.stats_path)
            and os.path.exists(self.mean_path)
        ):
            try:
                self.global_stats = pd.read_parquet(self.stats_path)
                self.global_mean = float(np.load(self.mean_path))
                print(f"Loaded global stats from {self.stats_path}")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print("Computing global route statistics on full dataset...")

        # Compute global scalar mean (fallback value)
        self.global_mean = full_train_df[config.TARGET_COL].mean()
        np.save(self.mean_path, np.array(self.global_mean))

        # Group by coordinates to get Sum and Count
        # Using a list for aggregation to ensure column names are flat and predictable
        stats = (
            full_train_df.groupby(self.coord_cols)[config.TARGET_COL]
            .agg(["sum", "count"])
            .reset_index()
        )
        stats.rename(
            columns={"sum": "global_sum", "count": "global_count"}, inplace=True
        )

        # Optimize data types to save memory
        stats["global_sum"] = stats["global_sum"].astype(np.float32)
        stats["global_count"] = stats["global_count"].astype(np.int32)

        self.global_stats = stats

        # 3. Save to cache
        self.global_stats.to_parquet(self.stats_path, index=False)
        print("Global stats computed and cached.")

    def transform_train_vectorized(
        self, train_subsample_df, num_folds=config.NUM_FOLDS
    ):
        """
        Applies Background-Augmented K-Fold encoding to the training subsample.
        Calculates feature = (Global_Sum - Fold_Sum) / (Global_Count - Fold_Count)

        Args:
            train_subsample_df (pd.DataFrame): The stable training subsample.
            num_folds (int): Number of folds for the subtraction logic.

        Returns:
            pd.DataFrame: The dataframe with the new 'route_avg_fare' feature.
        """
        if self.global_stats is None:
            raise ValueError(
                "GlobalRouteEncoder must be fitted before calling transform."
            )

        print(
            "Applying vectorized background-augmented encoding to training subsample..."
        )

        # Work on a copy
        df = train_subsample_df.copy()

        # 1. Assign Folds
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=config.RANDOM_SEED)
        df["fold_id"] = -1
        # Assign fold indices
        for fold_idx, (_, val_idx) in enumerate(kf.split(df)):
            df.iloc[val_idx, df.columns.get_loc("fold_id")] = fold_idx

        # 2. Merge Global Stats (Left Join)
        df = df.merge(self.global_stats, on=self.coord_cols, how="left")

        # Fill NaNs for global stats (if route not in global, which implies mismatch or debug subsetting)
        df["global_sum"] = df["global_sum"].fillna(0)
        df["global_count"] = df["global_count"].fillna(0)

        # 3. Compute Fold-Level Stats (Vectorized)
        # Group by [Coords, Fold_ID] within the subsample
        fold_stats = (
            df.groupby(self.coord_cols + ["fold_id"])[config.TARGET_COL]
            .agg(["sum", "count"])
            .reset_index()
        )
        fold_stats.rename(
            columns={"sum": "fold_sum", "count": "fold_count"}, inplace=True
        )

        # 4. Merge Fold Stats back
        df = df.merge(fold_stats, on=self.coord_cols + ["fold_id"], how="left")
        df["fold_sum"] = df["fold_sum"].fillna(0)
        df["fold_count"] = df["fold_count"].fillna(0)

        # 5. Calculate "Rest of World" Stats (Global - Fold)
        rest_sum = df["global_sum"] - df["fold_sum"]
        rest_count = df["global_count"] - df["fold_count"]

        # 6. Compute Feature
        # Initialize with global mean (fallback)
        df["route_avg_fare"] = self.global_mean

        # Avoid division by zero.
        # rest_count == 0 implies the route only exists in the current fold of the subsample
        # (and nowhere else in the global set, or global set == subsample).
        mask_valid = rest_count > 0
        df.loc[mask_valid, "route_avg_fare"] = (
            rest_sum[mask_valid] / rest_count[mask_valid]
        )

        # 7. Cleanup
        drop_cols = ["fold_id", "global_sum", "global_count", "fold_sum", "fold_count"]
        df.drop(columns=drop_cols, inplace=True)

        return df

    def transform_inference(self, df):
        """
        Applies global stats directly to validation or test sets.
        Feature = Global_Sum / Global_Count

        Args:
            df (pd.DataFrame): Validation or Test dataframe.

        Returns:
            pd.DataFrame: Dataframe with 'route_avg_fare'.
        """
        if self.global_stats is None:
            raise ValueError(
                "GlobalRouteEncoder must be fitted before calling transform."
            )

        print("Applying global route encoding to inference set...")

        df_out = df.copy()

        # 1. Merge global stats
        df_out = df_out.merge(self.global_stats, on=self.coord_cols, how="left")

        # 2. Compute Feature
        # Initialize with global mean
        df_out["route_avg_fare"] = self.global_mean

        # Calculate average where data exists
        mask_valid = (df_out["global_count"] > 0) & (df_out["global_count"].notna())
        df_out.loc[mask_valid, "route_avg_fare"] = (
            df_out.loc[mask_valid, "global_sum"]
            / df_out.loc[mask_valid, "global_count"]
        )

        # 3. Cleanup
        df_out.drop(
            columns=["global_sum", "global_count"], inplace=True, errors="ignore"
        )

        return df_out
