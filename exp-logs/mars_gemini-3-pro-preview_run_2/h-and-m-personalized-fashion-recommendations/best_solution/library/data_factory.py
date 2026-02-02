import pandas as pd
import numpy as np
from datetime import timedelta
from library.config import Config


class DataFactory:
    """
    Handles loading, merging, and time-slicing of transaction data for the
    Hybrid Multi-Source Retrieval pipeline.
    """

    @staticmethod
    def load_full_data(load_cached_data: bool = True) -> pd.DataFrame:
        """
        Loads the full transaction history by combining the user-split metadata files.
        Ensures proper datetime conversion and sorting.
        """
        # Define cache path
        cache_path = Config.get_cache_path("full_transactions.parquet")

        # 1. Try to load from cache
        if load_cached_data and cache_path.exists():
            print(f"Loading cached full transactions from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Compute from scratch
        print("Loading and merging metadata files (train + val)...")
        train_path = Config.METADATA_DIR / "train.parquet"
        val_path = Config.METADATA_DIR / "val.parquet"

        # Load partitions
        # Note: These are user-based splits from the metadata step
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)

        # Concatenate to get global history
        full_df = pd.concat([df_train, df_val], axis=0, ignore_index=True)

        # Preprocessing
        # Convert t_dat to datetime
        full_df["t_dat"] = pd.to_datetime(full_df["t_dat"])

        # Ensure article_id is string and padded (metadata should already be, but enforce consistency)
        full_df["article_id"] = full_df["article_id"].astype(str)

        # Sort by date for time-slicing logic
        full_df = full_df.sort_values("t_dat").reset_index(drop=True)

        print(
            f"Full dataset loaded: {len(full_df)} rows. Date range: {full_df['t_dat'].min()} to {full_df['t_dat'].max()}"
        )

        # 3. Save to cache
        print(f"Caching full transactions to {cache_path}")
        full_df.to_parquet(cache_path, index=False)

        return full_df

    @staticmethod
    def get_time_split(df: pd.DataFrame, load_cached_data: bool = True):
        """
        Splits the provided dataframe into Training History and Validation Ground Truth
        based on the global VAL_SIZE_DAYS.

        Args:
            df: The full transaction dataframe.
            load_cached_data: Whether to use cached splits.

        Returns:
            train_history (pd.DataFrame): Data available for training/candidate generation.
            val_ground_truth (pd.DataFrame): Data for the validation period (target).
        """
        # Determine split cutoff
        max_date = df["t_dat"].max()
        split_date = max_date - timedelta(days=Config.VAL_SIZE_DAYS)

        # Define cache paths with parameters
        params = {
            "split_date": str(split_date.date()),
            "val_days": Config.VAL_SIZE_DAYS,
        }
        train_cache = Config.get_cache_path("split_train_history.parquet", params)
        val_cache = Config.get_cache_path("split_val_ground_truth.parquet", params)

        # 1. Try to load from cache
        if load_cached_data and train_cache.exists() and val_cache.exists():
            print(f"Loading cached time split from {train_cache} and {val_cache}")
            return pd.read_parquet(train_cache), pd.read_parquet(val_cache)

        # 2. Compute from scratch
        print(f"Performing time split. Cutoff Date: {split_date.date()}")
        print(
            f"Validation Period: {(split_date + timedelta(days=1)).date()} to {max_date.date()}"
        )

        # Split logic
        # Train: <= split_date
        # Val: > split_date
        train_history = df[df["t_dat"] <= split_date].copy()
        val_ground_truth = df[df["t_dat"] > split_date].copy()

        print(f"Train History: {len(train_history)} rows")
        print(f"Val Ground Truth: {len(val_ground_truth)} rows")

        # 3. Save to cache
        train_history.to_parquet(train_cache, index=False)
        val_ground_truth.to_parquet(val_cache, index=False)

        return train_history, val_ground_truth

    @staticmethod
    def get_windowed_data(
        df: pd.DataFrame,
        end_date: pd.Timestamp,
        weeks: int,
        suffix: str,
        load_cached_data: bool = True,
    ) -> pd.DataFrame:
        """
        Extracts a specific time window of data ending at `end_date` (inclusive).
        Used for creating Co-occurrence (4 weeks) and Embedding (10 weeks) datasets.

        Args:
            df: Source dataframe (typically train_history).
            end_date: The cutoff date for the window.
            weeks: Duration of the window in weeks.
            suffix: Identifier for the dataset type (e.g., 'cooc', 'embed').
            load_cached_data: Whether to use cache.
        """
        start_date = end_date - timedelta(weeks=weeks)

        # Define cache path
        params = {"end_date": str(end_date.date()), "weeks": weeks, "type": suffix}
        cache_path = Config.get_cache_path(f"windowed_{suffix}.parquet", params)

        # 1. Try to load from cache
        if load_cached_data and cache_path.exists():
            print(f"Loading cached windowed data ({suffix}) from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Compute from scratch
        print(
            f"Slicing windowed data ({suffix}): > {start_date.date()} and <= {end_date.date()}"
        )

        mask = (df["t_dat"] > start_date) & (df["t_dat"] <= end_date)
        windowed_df = df.loc[mask].copy()

        print(f"Windowed Data ({suffix}): {len(windowed_df)} rows")

        # 3. Save to cache
        windowed_df.to_parquet(cache_path, index=False)

        return windowed_df

    @staticmethod
    def load_test_customers() -> pd.DataFrame:
        """
        Loads the test customer IDs from the metadata.
        """
        return pd.read_parquet(Config.METADATA_DIR / "test.parquet")
