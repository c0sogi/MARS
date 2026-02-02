import os
import ast
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import save_parquet, load_parquet, seed_everything, print_metric


class DataLoader:
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.train_path = os.path.join(self.cache_dir, "train_cleaned.parquet")
        self.val_path = os.path.join(self.cache_dir, "val_cleaned.parquet")
        self.test_path = os.path.join(self.cache_dir, "test_cleaned.parquet")

    def load_data(self, load_cached_data: bool = True):
        """
        Loads the train, validation, and test datasets.
        If load_cached_data is True and cached files exist, loads from cache.
        Otherwise, loads from raw metadata CSVs, cleans, aligns features, and caches.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
        """
        seed_everything()

        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(self.train_path)
                and os.path.exists(self.val_path)
                and os.path.exists(self.test_path)
            ):
                print("Loading data from cache...")
                try:
                    train_df = load_parquet(self.train_path)
                    val_df = load_parquet(self.val_path)
                    test_df = load_parquet(self.test_path)
                    return train_df, val_df, test_df
                except Exception as e:
                    print(f"Error loading cache: {e}. Reprocessing raw data.")
            else:
                print("Cache not found. Processing raw data...")
        else:
            print("Ignoring cache. Processing raw data...")

        # 2. Load Raw Data
        print("Loading raw metadata CSVs...")
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # 3. Parse List Columns
        # The CSV format stores lists as strings (e.g. "['a', 'b']"). We parse them back to lists.
        print("Parsing list columns...")
        for df in [train_df, val_df, test_df]:
            if Config.SUBREDDIT_LIST_COL in df.columns:
                df[Config.SUBREDDIT_LIST_COL] = df[Config.SUBREDDIT_LIST_COL].apply(
                    lambda x: (
                        ast.literal_eval(x)
                        if isinstance(x, str)
                        else (x if isinstance(x, list) else [])
                    )
                )

        # 4. Basic Cleaning & Leakage Removal
        # Drop columns that are known to be leaky or artifacts
        drop_cols = ["source_file", "giver_username_if_known", "request_text"]
        # Note: We drop 'request_text' in favor of 'request_text_edit_aware' to avoid leakage
        # from edits like "EDIT: Thanks for the pizza".

        for df in [train_df, val_df, test_df]:
            cols_to_drop = [c for c in drop_cols if c in df.columns]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)

        # 5. Feature Alignment (Intersection)
        # We must ensure that the model is only trained on features available in the test set.
        # Many columns in train.json (e.g., retrieval timestamps, upvotes at retrieval) are not in test.json.

        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        # Identify common features (excluding target)
        common_features = train_cols.intersection(test_cols)

        # Ensure critical ID column is present
        if Config.ID_COL not in common_features:
            # If ID is missing, something is wrong with the dataset or intersection logic
            # However, sometimes ID might be named differently, but here we assume Config is correct.
            # We force ID_COL to be in the list if it exists in both
            pass

        common_features_list = sorted(list(common_features))

        # Define final column lists
        target_col = Config.TARGET_COL

        # Train/Val should have Common Features + Target
        train_final_cols = common_features_list.copy()
        if target_col in train_df.columns:
            train_final_cols.append(target_col)

        val_final_cols = common_features_list.copy()
        if target_col in val_df.columns:
            val_final_cols.append(target_col)

        # Test should have Common Features
        test_final_cols = common_features_list.copy()

        # Apply filtering
        train_df = train_df[train_final_cols].copy()
        val_df = val_df[val_final_cols].copy()
        test_df = test_df[test_final_cols].copy()

        print(
            f"Feature alignment complete. Common features: {len(common_features_list)}"
        )
        print(f"Train shape: {train_df.shape}")
        print(f"Val shape: {val_df.shape}")
        print(f"Test shape: {test_df.shape}")

        # 6. Cache Processed Data
        print("Caching processed data...")
        save_parquet(train_df, self.train_path)
        save_parquet(val_df, self.val_path)
        save_parquet(test_df, self.test_path)

        return train_df, val_df, test_df


def load_data(load_cached_data: bool = True):
    """
    Wrapper function to instantiate DataLoader and load data.
    """
    loader = DataLoader()
    return loader.load_data(load_cached_data=load_cached_data)
