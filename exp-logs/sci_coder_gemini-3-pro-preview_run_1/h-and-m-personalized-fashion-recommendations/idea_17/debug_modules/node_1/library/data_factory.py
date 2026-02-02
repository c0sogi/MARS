import os
import pandas as pd
import numpy as np
from datetime import timedelta
from library.utils import Timer, memory_cleanup


class DataManager:
    def __init__(
        self,
        input_dir="./input",
        metadata_dir="./metadata",
        cache_dir="./working/idea_17",
    ):
        """
        Initialize the DataManager with directory paths.

        Args:
            input_dir (str): Path to raw input files.
            metadata_dir (str): Path to metadata files.
            cache_dir (str): Path to store cached parquet files.
        """
        self.input_dir = input_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_data(self, load_cached_data=True):
        """
        Loads the transaction data.
        If load_cached_data is True, attempts to load from parquet cache.
        Otherwise, loads from raw CSV, processes types, and caches.

        Returns:
            pd.DataFrame: The full transactions dataframe.
        """
        cache_path = os.path.join(self.cache_dir, "transactions_full.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached transactions from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

        with Timer("Loading Raw Data"):
            # We load the full transactions file to enable time-based splitting
            # as required by the TMVC architecture.
            csv_path = os.path.join(self.input_dir, "transactions_train.csv")

            # Using optimized types to save memory
            df = pd.read_csv(
                csv_path,
                dtype={
                    "article_id": "int32",
                    "price": "float32",
                    "sales_channel_id": "int8",
                },
            )

            # Convert date column to datetime objects
            df["t_dat"] = pd.to_datetime(df["t_dat"])

            # Save to cache for future runs
            print(f"Caching transactions to {cache_path}")
            df.to_parquet(cache_path, index=False)

        return df

    def get_time_split(self, df, days=7):
        """
        Splits the dataframe into training and validation sets based on the last 'days' days.
        This implements the Time-Based Split strategy (Train on T-1, Validate on T).

        Args:
            df (pd.DataFrame): The full transaction dataframe.
            days (int): Number of days for the validation period.

        Returns:
            tuple: (df_train, df_val)
        """
        max_date = df["t_dat"].max()
        split_date = max_date - timedelta(days=days)

        print(
            f"Splitting data. Train end: {split_date}. Validation start: {split_date + timedelta(days=1)}"
        )

        # Validation is strictly after the split date
        df_train = df[df["t_dat"] <= split_date].copy()
        df_val = df[df["t_dat"] > split_date].copy()

        return df_train, df_val

    def get_windowed_subsets(self, df_train, structure_weeks=16, velocity_weeks=1):
        """
        Extracts temporal subsets required for the TMVC architecture.

        Args:
            df_train (pd.DataFrame): The training split (history).
            structure_weeks (int): Window size for structure learning (Similarity Matrix).
            velocity_weeks (int): Window size for velocity modulation (Trend).

        Returns:
            tuple: (df_structure, df_velocity, df_full_history)
        """
        max_date = df_train["t_dat"].max()

        structure_start = max_date - timedelta(weeks=structure_weeks)
        velocity_start = max_date - timedelta(weeks=velocity_weeks)

        print(f"Generating Windowed Subsets (Max Date: {max_date})")
        print(f"Structure Window: {structure_weeks} weeks (Start: {structure_start})")
        print(f"Velocity Window: {velocity_weeks} weeks (Start: {velocity_start})")

        # Filter data for the structure window (e.g., last 16 weeks)
        df_structure = df_train[df_train["t_dat"] > structure_start].copy()

        # Filter data for the velocity window (e.g., last 1 week)
        df_velocity = df_train[df_train["t_dat"] > velocity_start].copy()

        # Full history is used for the Habit Stratum (Stratum 1)
        df_full_history = df_train

        return df_structure, df_velocity, df_full_history

    def load_test_customers(self):
        """
        Loads the list of customers requiring predictions for the submission.

        Returns:
            pd.DataFrame: DataFrame containing 'customer_id' column.
        """
        # Prefer metadata/test.csv as it is pre-generated
        test_path = os.path.join(self.metadata_dir, "test.csv")
        if os.path.exists(test_path):
            return pd.read_csv(test_path)
        else:
            # Fallback to sample_submission in input
            return pd.read_csv(os.path.join(self.input_dir, "sample_submission.csv"))
