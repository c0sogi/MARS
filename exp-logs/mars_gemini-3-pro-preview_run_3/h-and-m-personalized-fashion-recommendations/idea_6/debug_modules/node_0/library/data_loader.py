import os
import gc
import pandas as pd
import numpy as np
from datetime import timedelta
from library.config import Config
from library.utils import Timer, print_memory_usage


class DataLoader:
    """
    Handles loading, preprocessing, and splitting of dataset files.
    Implements caching to speed up iterative development.
    """

    def __init__(self):
        self.article_map = None
        self.customer_map = None
        self.inverse_article_map = None
        self.inverse_customer_map = None

    def load_data(self, load_cached_data: bool = True):
        """
        Main entry point to load all necessary datasets.

        Args:
            load_cached_data (bool): If True, attempts to load processed files from cache.
                                     If False or cache missing, re-processes raw data.

        Returns:
            tuple: (train_df, val_df, test_df, articles_df, customers_df)
                   All DataFrames have IDs mapped to integers.
        """
        # Define cache paths
        cache_train = Config.WORKING_DIR / "train_processed.parquet"
        cache_val = Config.WORKING_DIR / "val_processed.parquet"
        cache_test = Config.WORKING_DIR / "test_processed.parquet"
        cache_articles = Config.WORKING_DIR / "articles_processed.parquet"
        cache_customers = Config.WORKING_DIR / "customers_processed.parquet"

        # Check if cache exists
        cache_exists = (
            cache_train.exists()
            and cache_val.exists()
            and cache_test.exists()
            and cache_articles.exists()
            and cache_customers.exists()
            and Config.CACHE_ARTICLE_MAP.exists()
            and Config.CACHE_CUSTOMER_MAP.exists()
        )

        if load_cached_data and cache_exists:
            print("Loading data from cache...")
            with Timer("Load Cached Data"):
                train_df = pd.read_parquet(cache_train)
                val_df = pd.read_parquet(cache_val)
                test_df = pd.read_parquet(cache_test)
                articles_df = pd.read_parquet(cache_articles)
                customers_df = pd.read_parquet(cache_customers)

                # Load maps
                self.article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)
                self.customer_map = np.load(
                    Config.CACHE_CUSTOMER_MAP, allow_pickle=True
                )

                # Reconstruct inverse maps if needed (optional, but good for debugging)
                # self.inverse_article_map = {i: aid for i, aid in enumerate(self.article_map)}

                if Config.DEBUG:
                    print("Debug mode: Subsampling cached data...")
                    train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
                    val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

            print_memory_usage("After Cache Load")
            return train_df, val_df, test_df, articles_df, customers_df

        # If no cache or forced reload
        print("Cache not found or reload requested. Processing raw data...")
        return self._process_raw_data()

    def _process_raw_data(self):
        """
        Internal method to load raw files, generate mappings, and save to cache.
        """
        with Timer("Load Raw Files"):
            # Load Metadata Parquets (Transactions)
            train_df = pd.read_parquet(Config.TRAIN_METADATA)
            val_df = pd.read_parquet(Config.VAL_METADATA)
            test_df = pd.read_parquet(
                Config.TEST_METADATA
            )  # This is sample submission list

            # Load Entities
            articles_df = pd.read_csv(Config.ARTICLES_CSV)
            customers_df = pd.read_csv(Config.CUSTOMERS_CSV)

        # --- 1. Generate Mappings ---
        print("Generating ID Mappings...")

        # Article Map: Use all articles from articles.csv
        # Ensure article_id is int64
        articles_df["article_id"] = articles_df["article_id"].astype("int64")
        unique_articles = articles_df["article_id"].unique()
        # Create array where index is the mapped ID, value is original ID
        self.article_map = unique_articles
        # Dictionary for fast lookup during mapping
        article_id_to_idx = {aid: i for i, aid in enumerate(unique_articles)}

        # Customer Map: Union of all customers in Train, Val, Test, and Customers.csv
        # Note: customers.csv should contain all, but we ensure coverage
        unique_customers = customers_df["customer_id"].unique()
        self.customer_map = unique_customers
        customer_id_to_idx = {cid: i for i, cid in enumerate(unique_customers)}

        # --- 2. Map DataFrames ---
        print("Mapping DataFrames to Integer IDs...")

        def map_ids(df, cust_map, art_map=None):
            # Map Customer IDs
            if "customer_id" in df.columns:
                # Use map to handle potential missing keys safely (though shouldn't happen with full entity files)
                # Using pandas map is faster than apply
                df["customer_id"] = (
                    df["customer_id"].map(cust_map).fillna(-1).astype("int32")
                )

            # Map Article IDs
            if "article_id" in df.columns and art_map is not None:
                df["article_id"] = (
                    df["article_id"]
                    .astype("int64")
                    .map(art_map)
                    .fillna(-1)
                    .astype("int32")
                )

            return df

        train_df = map_ids(train_df, customer_id_to_idx, article_id_to_idx)
        val_df = map_ids(val_df, customer_id_to_idx, article_id_to_idx)
        test_df = map_ids(
            test_df, customer_id_to_idx
        )  # No article_id in test_metadata (it's submission file)

        articles_df["article_id"] = (
            articles_df["article_id"].map(article_id_to_idx).astype("int32")
        )
        customers_df["customer_id"] = (
            customers_df["customer_id"].map(customer_id_to_idx).astype("int32")
        )

        # --- 3. Date Conversion ---
        print("Converting Dates...")
        train_df["t_dat"] = pd.to_datetime(train_df["t_dat"])
        val_df["t_dat"] = pd.to_datetime(val_df["t_dat"])

        # --- 4. Debug Subsampling ---
        if Config.DEBUG:
            print(f"Debug Mode: Subsampling to {Config.DEBUG_SAMPLE_SIZE} rows...")
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()
            # We don't subsample entities aggressively to maintain referential integrity in maps,
            # but for strict debug speed we could. Here we keep entities to avoid key errors.

        # --- 5. Save to Cache ---
        print("Saving processed data to cache...")
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        train_df.to_parquet(Config.WORKING_DIR / "train_processed.parquet", index=False)
        val_df.to_parquet(Config.WORKING_DIR / "val_processed.parquet", index=False)
        test_df.to_parquet(Config.WORKING_DIR / "test_processed.parquet", index=False)
        articles_df.to_parquet(
            Config.WORKING_DIR / "articles_processed.parquet", index=False
        )
        customers_df.to_parquet(
            Config.WORKING_DIR / "customers_processed.parquet", index=False
        )

        np.save(Config.CACHE_ARTICLE_MAP, self.article_map)
        np.save(Config.CACHE_CUSTOMER_MAP, self.customer_map)

        print_memory_usage("After Processing")
        return train_df, val_df, test_df, articles_df, customers_df

    def get_sliding_windows(self, transactions_df):
        """
        Generates temporal splits for the Sliding Window training strategy.

        Args:
            transactions_df (pd.DataFrame): The full transaction history (usually train_df).

        Yields:
            dict: Contains 'train_history', 'train_target', 'window_id'
        """
        # Determine the global maximum date in the dataset
        max_date = transactions_df["t_dat"].max()

        print(f"Generating sliding windows. Max Date: {max_date}")

        for i in range(Config.NUM_SLIDING_WINDOWS):
            # Calculate the target week range for this window
            # Window 0: [Max - 7, Max] (Validation-like)
            # Window 1: [Max - 14, Max - 7]
            # ...
            days_offset = i * Config.SLIDING_WINDOW_STEP

            target_end = max_date - timedelta(days=days_offset)
            target_start = target_end - timedelta(days=Config.SLIDING_WINDOW_SIZE)

            # History is everything before target_start
            # We can further split history into Short-Term and Long-Term inside the pipeline
            # Here we just provide the cut-off point
            history_cutoff = target_start

            print(f"Window {i}: Target [{target_start} to {target_end}]")

            # Slice Data
            # Note: We use strictly less than for history to avoid leakage
            mask_history = transactions_df["t_dat"] < history_cutoff
            mask_target = (transactions_df["t_dat"] >= history_cutoff) & (
                transactions_df["t_dat"] <= target_end
            )

            df_history = transactions_df.loc[mask_history].copy()
            df_target = transactions_df.loc[mask_target].copy()

            if df_target.empty:
                print(f"Warning: Window {i} has empty target set. Skipping.")
                continue

            yield {
                "window_id": i,
                "history_df": df_history,
                "target_df": df_target,
                "target_start_date": target_start,
                "target_end_date": target_end,
            }

            # Explicit garbage collection to manage memory between yields
            del df_history, df_target, mask_history, mask_target
            gc.collect()

    def get_original_article_ids(self, mapped_ids):
        """
        Converts mapped integer IDs back to original article IDs (strings with leading zeros).
        """
        if self.article_map is None:
            self.article_map = np.load(Config.CACHE_ARTICLE_MAP, allow_pickle=True)

        # Handle single int or array/list
        if isinstance(mapped_ids, (int, np.integer)):
            original_id = self.article_map[mapped_ids]
            return f"{original_id:010d}"

        # Vectorized lookup
        original_ids = self.article_map[mapped_ids]
        # Convert to formatted strings
        return [f"{oid:010d}" for oid in original_ids]

    def get_original_customer_ids(self, mapped_ids):
        """
        Converts mapped integer IDs back to original customer IDs (hex strings).
        """
        if self.customer_map is None:
            self.customer_map = np.load(Config.CACHE_CUSTOMER_MAP, allow_pickle=True)

        if isinstance(mapped_ids, (int, np.integer)):
            return self.customer_map[mapped_ids]

        return self.customer_map[mapped_ids]
