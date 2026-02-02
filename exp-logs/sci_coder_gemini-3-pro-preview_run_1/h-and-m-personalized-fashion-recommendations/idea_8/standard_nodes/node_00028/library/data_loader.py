import pandas as pd
import numpy as np
import os
import datetime
from library.utils import reduce_mem_usage, Timer


class TransactionLoader:
    """
    Handles loading, preprocessing, and time-based splitting of transaction data
    for the Decay-Weighted Behavioral Cascade model.
    """

    def __init__(
        self,
        data_dir="./input",
        metadata_dir="./metadata",
        cache_dir="./working/idea_8",
    ):
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def compute_decay_weights(self, df, reference_date):
        """
        Computes temporal decay weights: w(t) = 1 / sqrt(days_elapsed + 1).

        Args:
            df (pd.DataFrame): DataFrame containing a 't_dat' column.
            reference_date (datetime): The reference date (start of prediction window)
                                       to calculate elapsed days from.

        Returns:
            np.array: Array of float32 weights.
        """
        # Ensure t_dat is datetime
        if not np.issubdtype(df["t_dat"].dtype, np.datetime64):
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        # Calculate days elapsed: (Ref - Transaction_Date).days
        # Note: If transaction is on Ref-1, days=1. If on Ref, days=0.
        days_elapsed = (reference_date - df["t_dat"]).dt.days

        # Formula: 1 / sqrt(d + 1)
        # We use absolute value for safety, though days_elapsed should be >= 0 in valid splits
        weights = 1.0 / np.sqrt(np.abs(days_elapsed) + 1.0)

        return weights.astype(np.float32)

    def load_transactions(
        self, load_cached_data=True, train_weeks=8, val_days=7, validation=True
    ):
        """
        Loads transactions, performs time-based split, and computes decay weights.

        Args:
            load_cached_data (bool): Whether to load from cache if available.
            train_weeks (int): Number of weeks of history to use for training.
            val_days (int): Number of days for validation window (usually 7).
            validation (bool): If True, creates a hold-out validation set from the last `val_days`.
                               If False, uses the latest data for training (for submission).

        Returns:
            train_df (pd.DataFrame): Weighted training transactions.
            val_df (pd.DataFrame or None): Validation transactions (ground truth) if validation=True.
            test_customers (pd.DataFrame): DataFrame containing 'customer_id' for the submission.
        """
        # Construct cache paths based on parameters
        cache_key = f"tr_w{train_weeks}_v{val_days}_val{str(validation).lower()}"
        train_cache_path = os.path.join(self.cache_dir, f"{cache_key}_train.parquet")
        val_cache_path = os.path.join(self.cache_dir, f"{cache_key}_val.parquet")
        test_cache_path = os.path.join(self.cache_dir, "test_customers.parquet")

        # 1. Try Loading from Cache
        if load_cached_data:
            cache_exists = os.path.exists(train_cache_path) and os.path.exists(
                test_cache_path
            )
            if validation:
                cache_exists = cache_exists and os.path.exists(val_cache_path)

            if cache_exists:
                print(
                    f"[TransactionLoader] Loading cached data from {self.cache_dir}..."
                )
                train_df = pd.read_parquet(train_cache_path)
                test_customers = pd.read_parquet(test_cache_path)
                val_df = pd.read_parquet(val_cache_path) if validation else None
                return train_df, val_df, test_customers

        # 2. Load Raw Data
        print(
            "[TransactionLoader] Cache miss or reload requested. Processing raw data..."
        )
        with Timer("Load Raw CSVs"):
            # We load the metadata splits (which are user-based) and concat them
            # to get the full timeline for time-based splitting.
            cols = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
            dtypes = {
                "article_id": "int32",
                "price": "float32",
                "sales_channel_id": "int8",
            }

            df_train_meta = pd.read_csv(
                os.path.join(self.metadata_dir, "train.csv"), usecols=cols, dtype=dtypes
            )
            df_val_meta = pd.read_csv(
                os.path.join(self.metadata_dir, "val.csv"), usecols=cols, dtype=dtypes
            )

            # Concatenate to reconstruct full history
            df = pd.concat([df_train_meta, df_val_meta], axis=0, ignore_index=True)

            # Load test customers (submission template)
            test_customers = pd.read_csv(os.path.join(self.metadata_dir, "test.csv"))

            # Convert date
            df["t_dat"] = pd.to_datetime(df["t_dat"])

        # 3. Perform Time-Based Split
        print(
            f"[TransactionLoader] Performing Time-Based Split (Validation={validation})..."
        )
        max_date = df["t_dat"].max()

        if validation:
            # Validation Mode:
            # Validation Set = Last `val_days` (e.g., 7 days)
            # Training Set = `train_weeks` prior to Validation Start

            # Calculate start of validation period
            # If max_date is 2020-09-22 and val_days=7, val starts 2020-09-16
            val_start_date = max_date - datetime.timedelta(days=val_days - 1)

            # Calculate start of training period
            # Train ends where Val starts (exclusive of val_start_date for train, inclusive for val)
            train_end_date = val_start_date
            train_start_date = train_end_date - datetime.timedelta(weeks=train_weeks)

            # Create masks
            val_mask = df["t_dat"] >= val_start_date
            train_mask = (df["t_dat"] < val_start_date) & (
                df["t_dat"] >= train_start_date
            )

            val_df = df.loc[val_mask].copy()
            train_df = df.loc[train_mask].copy()

            # Reference date for decay is the start of validation (the "prediction time")
            ref_date = val_start_date

            print(
                f"  Train Period: {train_start_date.date()} to {train_end_date.date()} (Exclusive)"
            )
            print(f"  Val Period:   {val_start_date.date()} to {max_date.date()}")

        else:
            # Submission Mode:
            # Training Set = Last `train_weeks` of available data
            # No Validation Set

            train_end_date = max_date  # Inclusive
            train_start_date = max_date - datetime.timedelta(weeks=train_weeks)

            train_mask = df["t_dat"] >= train_start_date
            train_df = df.loc[train_mask].copy()
            val_df = None

            # Reference date is the day AFTER max_date (start of test period)
            ref_date = max_date + datetime.timedelta(days=1)

            print(f"  Train Period: {train_start_date.date()} to {max_date.date()}")

        # 4. Compute Decay Weights
        print("[TransactionLoader] Computing decay weights...")
        train_df["decay_weight"] = self.compute_decay_weights(train_df, ref_date)

        # 5. Optimize Memory
        train_df = reduce_mem_usage(train_df, verbose=False)
        test_customers = reduce_mem_usage(test_customers, verbose=False)
        if val_df is not None:
            val_df = reduce_mem_usage(val_df, verbose=False)

        # 6. Cache Data
        print(f"[TransactionLoader] Saving to cache: {self.cache_dir}")
        train_df.to_parquet(train_cache_path, index=False)
        test_customers.to_parquet(test_cache_path, index=False)
        if val_df is not None:
            val_df.to_parquet(val_cache_path, index=False)

        return train_df, val_df, test_customers

    def load_articles(self):
        """Loads articles metadata."""
        return pd.read_csv(
            os.path.join(self.data_dir, "articles.csv"), dtype={"article_id": "int32"}
        )

    def load_customers(self):
        """Loads customers metadata."""
        return pd.read_csv(os.path.join(self.data_dir, "customers.csv"))
