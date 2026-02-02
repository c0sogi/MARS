import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from library.config import SMOOTHING_ALPHA


class GlobalRouteEncoder:
    """
    Implements Global-Scale Out-of-Fold (OOF) Target Encoding.

    This encoder discretizes spatial coordinates into grid cells and computes
    mean target values for each route (pickup_cell -> dropoff_cell).

    For the training set, it uses K-Fold OOF encoding to prevent data leakage.
    For validation/test sets, it applies the global mean computed from the full training set.
    """

    def __init__(
        self, grid_precision, n_splits=5, random_state=42, target_col="fare_amount"
    ):
        """
        Args:
            grid_precision (int): Decimal places for coordinate rounding.
            n_splits (int): Number of folds for OOF encoding.
            random_state (int): Seed for reproducibility.
            target_col (str): Name of the target variable.
        """
        self.grid_precision = grid_precision
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_col = target_col
        self.global_map = None
        self.global_mean = None
        self.group_cols = ["p_lat_r", "p_lon_r", "d_lat_r", "d_lon_r"]

    def _discretize(self, df):
        """
        Rounds coordinates to create grid cells.
        Returns a copy of the dataframe with discretized columns.
        """
        df = df.copy()
        df["p_lat_r"] = np.round(df["pickup_latitude"], self.grid_precision)
        df["p_lon_r"] = np.round(df["pickup_longitude"], self.grid_precision)
        df["d_lat_r"] = np.round(df["dropoff_latitude"], self.grid_precision)
        df["d_lon_r"] = np.round(df["dropoff_longitude"], self.grid_precision)
        return df

    def fit_transform_oof(self, df):
        """
        Performs K-Fold OOF Target Encoding on the training set.

        1. Discretizes coordinates.
        2. Splits data into K folds.
        3. For each fold, computes mean target of routes using out-of-fold data.
        4. Fills NaNs with global mean.
        5. Stores global statistics for later use in transform_global.

        Args:
            df (pd.DataFrame): Training data containing coordinates and target.

        Returns:
            pd.DataFrame: Dataframe with added 'oof_fare' column.
        """
        # Discretize coordinates
        df = self._discretize(df)

        # Initialize OOF column
        df["oof_fare"] = np.nan

        # Setup K-Fold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # Cite solution_lesson_node_00038: Spatially-Smoothed Target Encoding

        # Iterate through folds
        for train_idx, val_idx in kf.split(df):
            # Extract training fold data
            train_fold = df.iloc[train_idx][self.group_cols + [self.target_col]]

            # Compute route stats (count and mean)
            route_stats = (
                train_fold.groupby(self.group_cols)[self.target_col]
                .agg(["mean", "count"])
                .reset_index()
            )

            # Compute global mean for this fold
            fold_global_mean = train_fold[self.target_col].mean()

            # Apply Bayesian Smoothing
            # smoothed = (mean * count + global * alpha) / (count + alpha)
            route_stats["mean_fare"] = (
                route_stats["mean"] * route_stats["count"]
                + fold_global_mean * SMOOTHING_ALPHA
            ) / (route_stats["count"] + SMOOTHING_ALPHA)

            # Prepare validation fold for merging
            val_fold_features = df.iloc[val_idx][self.group_cols].reset_index()

            # Merge stats onto validation fold
            merged = val_fold_features.merge(
                route_stats[self.group_cols + ["mean_fare"]],
                on=self.group_cols,
                how="left",
            )

            # Assign predicted means back to the main dataframe
            update_series = pd.Series(
                merged["mean_fare"].values, index=merged["index"].values
            )
            df.loc[update_series.index, "oof_fare"] = update_series

        # Compute Global Stats (on full dataset) for inference
        self.global_mean = df[self.target_col].mean()

        # Fill NaNs in OOF column with global mean
        df["oof_fare"] = df["oof_fare"].fillna(self.global_mean)

        # Create and store the Global Route Map with Smoothing
        self.global_map = (
            df.groupby(self.group_cols)[self.target_col]
            .agg(["mean", "count"])
            .reset_index()
        )

        self.global_map["global_avg_fare"] = (
            self.global_map["mean"] * self.global_map["count"]
            + self.global_mean * SMOOTHING_ALPHA
        ) / (self.global_map["count"] + SMOOTHING_ALPHA)

        # Keep only necessary columns
        self.global_map = self.global_map[self.group_cols + ["global_avg_fare"]]

        return df

    def transform_global(self, df):
        """
        Applies Global Target Encoding to a new dataset (Test/Val).
        Uses the map computed during fit_transform_oof.

        Args:
            df (pd.DataFrame): Data containing coordinates.

        Returns:
            pd.DataFrame: Dataframe with added 'oof_fare' column.
        """
        if self.global_map is None:
            raise ValueError(
                "Encoder must be fitted using fit_transform_oof before calling transform_global."
            )

        # Discretize
        df = self._discretize(df)

        # Merge with global map
        # We use left merge to preserve the input dataframe structure
        df = df.merge(self.global_map, on=self.group_cols, how="left")

        # Fill missing routes with the global mean
        df["oof_fare"] = df["global_avg_fare"].fillna(self.global_mean)

        # Drop the intermediate column from the merge
        df = df.drop(columns=["global_avg_fare"])

        return df
