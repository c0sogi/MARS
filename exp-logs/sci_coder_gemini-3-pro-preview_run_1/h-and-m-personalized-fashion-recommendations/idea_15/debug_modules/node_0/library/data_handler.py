import os
import pandas as pd
import numpy as np
from library import config


class DataHandler:
    """
    Handles data loading, preprocessing, mapping, and splitting for the ADIPC model.
    Manages the transition between raw CSV data and optimized structures for
    sparse matrix operations.
    """

    def __init__(self):
        self.item_map = None
        self.user_map = None

    def create_mappings(self, load_cached_data=True):
        """
        Creates or loads mappings for customer_id and article_id to integers.
        Ensures all customers and articles in the ecosystem are mapped to a dense integer range.

        Args:
            load_cached_data (bool): If True, attempts to load from disk cache.

        Returns:
            tuple: (item_map DataFrame, user_map DataFrame)
        """
        item_map_path = os.path.join(config.WORKING_DIR, config.CACHE_ITEM_MAP)
        user_map_path = os.path.join(config.WORKING_DIR, config.CACHE_USER_MAP)

        if (
            load_cached_data
            and os.path.exists(item_map_path)
            and os.path.exists(user_map_path)
        ):
            print(f"Loading ID mappings from cache: {config.WORKING_DIR}...")
            self.item_map = pd.read_parquet(item_map_path)
            self.user_map = pd.read_parquet(user_map_path)
        else:
            print("Generating new ID mappings from raw metadata...")

            # Load all unique articles from the master file
            articles = pd.read_csv(config.ARTICLES_CSV, usecols=["article_id"])
            unique_articles = articles["article_id"].unique()
            self.item_map = pd.DataFrame(
                {
                    "article_id": unique_articles,
                    "article_idx": np.arange(len(unique_articles), dtype="int32"),
                }
            )

            # Load all unique customers from the master file
            customers = pd.read_csv(config.CUSTOMERS_CSV, usecols=["customer_id"])
            unique_customers = customers["customer_id"].unique()
            self.user_map = pd.DataFrame(
                {
                    "customer_id": unique_customers,
                    "user_idx": np.arange(len(unique_customers), dtype="int32"),
                }
            )

            # Save to cache
            print(f"Saving mappings to {config.WORKING_DIR}...")
            self.item_map.to_parquet(item_map_path, index=False)
            self.user_map.to_parquet(user_map_path, index=False)

        return self.item_map, self.user_map

    def load_dataset(self, mode="validation", debug_size=None, load_cached_data=True):
        """
        Loads the transaction data, applies mappings, and splits based on the requested mode.

        Args:
            mode (str): 'validation' or 'submission'.
                        - 'validation': Holds out last 7 days for scoring.
                        - 'submission': Uses full dataset for training.
            debug_size (int): If set, limits the number of transaction rows (taking the most recent).
            load_cached_data (bool): Whether to use cached processed transactions.

        Returns:
            dict: Contains:
                - 'history_df': DataFrame for model input (graph & query).
                - 'future_df': DataFrame for ground truth (validation only).
                - 'target_users': Array of user_indices to predict for.
                - 'cutoff_date': Timestamp used for the split.
                - 'item_map': The article mapping DataFrame.
                - 'user_map': The customer mapping DataFrame.
        """
        # 1. Ensure mappings exist
        if self.item_map is None or self.user_map is None:
            self.create_mappings(load_cached_data=load_cached_data)

        # 2. Load and Process Transactions
        cache_path = os.path.join(config.WORKING_DIR, "transactions_processed.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print("Loading processed transactions from cache...")
            df = pd.read_parquet(cache_path)
        else:
            print("Processing raw transactions from metadata...")
            # Load train and val partitions from metadata to ensure we use the clean dataset
            # We concatenate them to reconstruct the full timeline
            df_train = pd.read_csv(
                config.TRAIN_META_PATH, usecols=["t_dat", "customer_id", "article_id"]
            )
            df_val = pd.read_csv(
                config.VAL_META_PATH, usecols=["t_dat", "customer_id", "article_id"]
            )

            df = pd.concat([df_train, df_val], ignore_index=True)

            # Convert date to datetime for temporal operations
            df["t_dat"] = pd.to_datetime(df["t_dat"])

            # Map IDs to integers
            print("Mapping IDs...")
            df = df.merge(self.user_map, on="customer_id", how="left")
            df = df.merge(self.item_map, on="article_id", how="left")

            # Drop original string columns to save memory
            df = df.drop(columns=["customer_id", "article_id"])

            # Drop unmapped rows (should be zero if mappings are from master files)
            original_len = len(df)
            df = df.dropna(subset=["user_idx", "article_idx"])
            if len(df) < original_len:
                print(
                    f"Warning: Dropped {original_len - len(df)} rows due to missing ID mappings."
                )

            # Cast indices to int32 for memory efficiency
            df["user_idx"] = df["user_idx"].astype("int32")
            df["article_idx"] = df["article_idx"].astype("int32")

            # Save cache
            print(f"Saving processed transactions to {cache_path}...")
            df.to_parquet(cache_path, index=False)

        # 3. Apply Debugging
        if debug_size is not None:
            print(f"DEBUG MODE: Sampling last {debug_size} rows...")
            df = df.sort_values("t_dat").tail(debug_size).copy()

        # 4. Determine Split Parameters
        max_date = df["t_dat"].max()
        print(f"Dataset Max Date: {max_date}")

        if mode == "validation":
            # Validation: Simulate the test week using the last 7 days of data
            # The 'future' is the last 7 days. 'History' is everything before.
            cutoff_date = max_date - pd.Timedelta(days=7)
            print(f"Validation Mode: Cutoff Date = {cutoff_date}")

            history_df = df[df["t_dat"] <= cutoff_date].copy()
            future_df = df[df["t_dat"] > cutoff_date].copy()

            # Target users for validation are strictly those in the validation set (metadata/val.csv)
            # We load the customer_ids from metadata and map them
            val_meta = pd.read_csv(config.VAL_META_PATH, usecols=["customer_id"])
            target_users_df = val_meta.merge(
                self.user_map, on="customer_id", how="inner"
            )
            target_user_indices = target_users_df["user_idx"].unique()

            # Filter future_df to only include target users (ground truth)
            # This ensures we evaluate exactly on the validation cohort
            future_df = future_df[future_df["user_idx"].isin(target_user_indices)]

        elif mode == "submission":
            # Submission: Use all data as history. Future is unknown.
            cutoff_date = max_date
            print(f"Submission Mode: Cutoff Date = {cutoff_date}")

            history_df = df.copy()
            future_df = None

            # Target users are from sample_submission (metadata/test.csv)
            test_meta = pd.read_csv(config.TEST_META_PATH, usecols=["customer_id"])
            target_users_df = test_meta.merge(
                self.user_map, on="customer_id", how="inner"
            )
            target_user_indices = target_users_df["user_idx"].unique()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"History Rows: {len(history_df)}")
        if future_df is not None:
            print(f"Validation Truth Rows: {len(future_df)}")
        print(f"Target Users: {len(target_user_indices)}")

        return {
            "history_df": history_df,
            "future_df": future_df,
            "target_users": target_user_indices,
            "cutoff_date": cutoff_date,
            "item_map": self.item_map,
            "user_map": self.user_map,
        }

    def get_structure_data(self, history_df, cutoff_date):
        """
        Extracts the subset of history used for Structure Learning (Item-Item Graph).
        Uses config.GRAPH_WINDOW_WEEKS to define the window size.
        """
        start_date = cutoff_date - pd.Timedelta(weeks=config.GRAPH_WINDOW_WEEKS)
        print(f"Structure Learning Window: {start_date} to {cutoff_date}")
        return history_df[history_df["t_dat"] > start_date].copy()

    def get_active_inventory(self, history_df, cutoff_date):
        """
        Identifies items sold in the active inventory window and returns a boolean mask.
        Uses config.INVENTORY_WINDOW_WEEKS to define the window size.

        Returns:
            np.ndarray: Boolean array of shape (num_items,) where True indicates active.
        """
        start_date = cutoff_date - pd.Timedelta(weeks=config.INVENTORY_WINDOW_WEEKS)
        print(f"Inventory Window: {start_date} to {cutoff_date}")

        active_df = history_df[history_df["t_dat"] > start_date]
        active_items = active_df["article_idx"].unique()

        # Create a boolean mask for all items
        # Size is max index + 1
        max_idx = self.item_map["article_idx"].max()
        mask = np.zeros(max_idx + 1, dtype=bool)
        mask[active_items] = True

        print(f"Active Inventory Size: {len(active_items)} items")
        return mask

    def get_user_history(self, history_df):
        """
        Returns the full history per user, sorted by date, optimized for query construction.
        """
        return history_df.sort_values(["user_idx", "t_dat"])
