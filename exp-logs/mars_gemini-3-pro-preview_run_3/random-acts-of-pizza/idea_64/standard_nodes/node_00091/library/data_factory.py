import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import Timer


class DataFactory:
    """
    DataFactory is responsible for loading, cleaning, and structuring the raw data.
    It implements the 'Union Dataset' strategy by merging train and validation sets
    and performs strict feature engineering and leakage prevention.
    """

    @staticmethod
    def load_union_dataset(load_cached_data: bool = True):
        """
        Loads the training (union of train+val) and test datasets.
        Applies caching to avoid redundant processing.

        Args:
            load_cached_data (bool): If True, attempts to load from processed cache.

        Returns:
            tuple: (train_df, test_df)
                - train_df (pd.DataFrame): Processed union training data with target.
                - test_df (pd.DataFrame): Processed test data without target.
        """
        # Define cache paths
        Config.setup()  # Ensure directories exist
        cache_train_path = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
        cache_test_path = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

        # Attempt to load from cache
        if load_cached_data:
            if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
                print(f"Loading cached data from {Config.WORKING_DIR}...")
                with Timer("Load Cached Data"):
                    train_df = pd.read_parquet(cache_train_path)
                    test_df = pd.read_parquet(cache_test_path)
                    return train_df, test_df
            else:
                print("Cache not found. Processing from scratch...")
        else:
            print("Ignoring cache. Processing from scratch...")

        with Timer("Data Processing"):
            # 1. Load Raw Metadata
            if not os.path.exists(Config.TRAIN_PATH) or not os.path.exists(
                Config.VAL_PATH
            ):
                raise FileNotFoundError(
                    "Metadata files not found. Ensure ./metadata exists."
                )

            raw_train = pd.read_parquet(Config.TRAIN_PATH)
            raw_val = pd.read_parquet(Config.VAL_PATH)
            raw_test = pd.read_parquet(Config.TEST_PATH)

            # 2. Create Union Dataset (Train + Val)
            # We reset index to ensure unique indices after concatenation
            union_train = pd.concat([raw_train, raw_val], axis=0).reset_index(drop=True)
            test_df = raw_test.copy()

            print(f"Union Train Shape: {union_train.shape}")
            print(f"Test Shape: {test_df.shape}")

            # 3. Feature Engineering & Cleaning

            # Helper function to process a dataframe
            def process_dataframe(df, is_train=True):
                # A. Text Concatenation (Title + Body)
                # Handle NaNs in text columns just in case
                title = df[Config.TEXT_COLS[0]].fillna("").astype(str)
                body = df[Config.TEXT_COLS[1]].fillna("").astype(str)
                df["text_combined"] = title + " " + body

                # B. Behavioral Feature (Subreddits list to string)
                # Convert list of subreddits to space-separated string for TF-IDF
                if Config.SUBREDDIT_COL in df.columns:
                    df["subreddit_text"] = df[Config.SUBREDDIT_COL].apply(
                        lambda x: (
                            " ".join(x)
                            if isinstance(x, list)
                            else (str(x) if x is not None else "")
                        )
                    )
                else:
                    df["subreddit_text"] = ""

                # C. Column Selection
                # We keep ID, engineered features, allow-listed metadata, and target (if train)
                cols_to_keep = [
                    Config.ID_COL,
                    "text_combined",
                    "subreddit_text",
                ] + Config.METADATA_COLS

                if is_train:
                    cols_to_keep.append(Config.TARGET_COL)

                # Filter columns (ensure they exist)
                available_cols = [c for c in cols_to_keep if c in df.columns]
                return df[available_cols].copy()

            # Apply processing
            union_train_proc = process_dataframe(union_train, is_train=True)
            test_proc = process_dataframe(test_df, is_train=False)

            # 4. Imputation (Fit on Train, Transform Train & Test)
            # We use the allow-listed metadata columns
            for col in Config.METADATA_COLS:
                if col in union_train_proc.columns:
                    # Convert to numeric, coercing errors
                    union_train_proc[col] = pd.to_numeric(
                        union_train_proc[col], errors="coerce"
                    )
                    test_proc[col] = pd.to_numeric(test_proc[col], errors="coerce")

                    # Calculate median from training set
                    median_val = union_train_proc[col].median()

                    # Fill NaNs
                    union_train_proc[col] = union_train_proc[col].fillna(median_val)
                    test_proc[col] = test_proc[col].fillna(median_val)

            # 5. Save to Cache
            print(f"Saving processed data to {Config.WORKING_DIR}...")
            union_train_proc.to_parquet(cache_train_path, index=False)
            test_proc.to_parquet(cache_test_path, index=False)

        return union_train_proc, test_proc
