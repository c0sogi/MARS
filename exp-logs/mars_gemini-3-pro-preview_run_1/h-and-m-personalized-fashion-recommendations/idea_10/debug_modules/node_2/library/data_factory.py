import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from datetime import timedelta
from library.config import Config
from library.utils import reduce_mem_usage, Timer


class DataFactory:
    """
    Responsible for loading raw data, performing time-based filtering,
    calculating time-decay weights, and constructing the sparse interaction matrix.
    """

    @staticmethod
    def load_and_preprocess_data(data_paths, is_validation=False):
        """
        Loads data, handles time splitting for validation, filters by history window,
        and computes decay weights.
        """
        # 1. Load Data
        dfs = []
        for path in data_paths:
            if os.path.exists(path):
                # Load only necessary columns with optimized types
                df = pd.read_csv(
                    path,
                    usecols=["t_dat", "customer_id", "article_id"],
                    dtype={"article_id": "int32"},
                )
                dfs.append(df)
            else:
                print(f"Warning: Data path {path} does not exist. Skipping.")

        if not dfs:
            raise ValueError("No data loaded. Check paths.")

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df["t_dat"] = pd.to_datetime(df["t_dat"])

        # 2. Determine Split Points
        max_date = df["t_dat"].max()

        if is_validation:
            # Validation: Train on [Start, T-7], Validate on [T-6, T]
            # The prompt says "predict ... in the 7-day period immediately after"
            split_date = max_date - timedelta(days=7)

            val_df = df[df["t_dat"] > split_date].copy()
            train_df = df[df["t_dat"] <= split_date].copy()

            # Effective max date for decay calculation is the split date
            eff_max_date = split_date
        else:
            # Submission: Train on all available data
            val_df = None
            train_df = df
            eff_max_date = max_date

        # 3. Filter History Window
        # Keep only transactions within HISTORY_DAYS from the effective max date
        history_start_date = eff_max_date - timedelta(days=Config.HISTORY_DAYS)
        train_df = train_df[train_df["t_dat"] > history_start_date].copy()

        print(
            f"Data Filtered: {len(train_df)} rows. Window: {history_start_date.date()} to {eff_max_date.date()}"
        )

        # 4. Compute Time Decay Weights
        # w(t) = 1 / (days_elapsed + 1) ^ DECAY_POWER
        # days_elapsed = 0 for the most recent day
        train_df["days_elapsed"] = (eff_max_date - train_df["t_dat"]).dt.days

        # Ensure non-negative (sanity check)
        train_df["days_elapsed"] = train_df["days_elapsed"].clip(lower=0)

        # Calculate weight
        # Using float32 for precision as requested
        train_df["weight"] = 1.0 / np.power(
            train_df["days_elapsed"] + 1, Config.DECAY_POWER
        )
        train_df["weight"] = train_df["weight"].astype(np.float32)

        return train_df, val_df

    @staticmethod
    def get_interaction_matrix(
        data_paths=None, is_validation=False, load_cached_data=True
    ):
        """
        Constructs or loads the Time-Embedded Interaction Matrix.

        Returns:
            matrix_csr: Sparse interaction matrix (Users x Items)
            user_map: DataFrame mapping customer_id -> user_idx
            item_map: DataFrame mapping article_id -> item_idx
            val_df: Validation DataFrame (if is_validation=True, else None)
        """
        # Default paths
        if data_paths is None:
            data_paths = [Config.TRAIN_PATH, Config.VAL_PATH]

        # Define Cache Paths
        mode_str = "val" if is_validation else "full"
        cache_dir = Config.WORKING_DIR

        matrix_path = os.path.join(cache_dir, f"{mode_str}_interaction_matrix.npz")
        user_map_path = os.path.join(cache_dir, f"{mode_str}_user_map.parquet")
        item_map_path = os.path.join(cache_dir, f"{mode_str}_item_map.parquet")
        val_data_path = os.path.join(cache_dir, f"{mode_str}_val_data.parquet")

        # 1. Try Loading Cache
        if load_cached_data:
            # Check if main artifacts exist
            artifacts_exist = (
                os.path.exists(matrix_path)
                and os.path.exists(user_map_path)
                and os.path.exists(item_map_path)
            )
            # If validation, check val data too
            if is_validation and not os.path.exists(val_data_path):
                artifacts_exist = False

            if artifacts_exist:
                print(f"[{mode_str}] Loading cached data from {cache_dir}...")
                matrix_csr = sp.load_npz(matrix_path)
                user_map = pd.read_parquet(user_map_path)
                item_map = pd.read_parquet(item_map_path)
                val_df = pd.read_parquet(val_data_path) if is_validation else None
                return matrix_csr, user_map, item_map, val_df

        # 2. Compute from Scratch
        print(f"[{mode_str}] Cache not found or reload requested. Computing...")

        with Timer("Data Preprocessing"):
            train_df, val_df = DataFactory.load_and_preprocess_data(
                data_paths, is_validation
            )
            train_df = reduce_mem_usage(train_df, verbose=False)

        with Timer("Map Creation"):
            # User Map: Must include ALL customers in sample_submission
            sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH, usecols=["customer_id"])
            all_customers = sub_df["customer_id"].unique()
            user_map = pd.DataFrame({"customer_id": all_customers})
            user_map["user_idx"] = np.arange(len(user_map), dtype=np.int32)

            # Item Map: Include ALL articles
            art_df = pd.read_csv(
                Config.ARTICLES_PATH,
                usecols=["article_id"],
                dtype={"article_id": "int32"},
            )
            all_articles = art_df["article_id"].unique()
            item_map = pd.DataFrame({"article_id": all_articles})
            item_map["item_idx"] = np.arange(len(item_map), dtype=np.int32)

        with Timer("Matrix Construction"):
            # Map IDs to indices
            # Use inner join to drop transactions for users/items not in our universe
            train_df = train_df.merge(user_map, on="customer_id", how="inner")
            train_df = train_df.merge(item_map, on="article_id", how="inner")

            # Extract COO components
            rows = train_df["user_idx"].values
            cols = train_df["item_idx"].values
            data = train_df["weight"].values

            shape = (len(user_map), len(item_map))

            # Construct COO matrix (handles duplicates by summing when converting to CSR)
            # Note: coo_matrix constructor doesn't sum, but .tocsr() does sum duplicate entries.
            matrix_coo = sp.coo_matrix(
                (data, (rows, cols)), shape=shape, dtype=np.float32
            )
            matrix_csr = matrix_coo.tocsr()

            print(f"Matrix Shape: {matrix_csr.shape}, Non-zeros: {matrix_csr.nnz}")

        # 3. Save Cache
        print(f"[{mode_str}] Saving artifacts to {cache_dir}...")
        sp.save_npz(matrix_path, matrix_csr)
        user_map.to_parquet(user_map_path)
        item_map.to_parquet(item_map_path)
        if is_validation and val_df is not None:
            val_df.to_parquet(val_data_path)

        return matrix_csr, user_map, item_map, val_df
