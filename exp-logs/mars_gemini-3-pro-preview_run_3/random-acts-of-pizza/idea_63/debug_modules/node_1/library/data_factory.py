import os
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import save_to_cache, load_from_cache, print_header


class DataFactory:
    """
    DataFactory is responsible for loading, merging, and cleaning the dataset.
    It enforces leakage prevention, manages the 'Union Dataset' strategy,
    and handles caching of processed data.
    """

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies strict cleaning rules to prevent leakage and ensure schema consistency.

        1. Drops columns suffixed with '_at_retrieval' (leakage).
        2. Drops 'request_text' to enforce usage of 'request_text_edit_aware'.
        3. Ensures configured text columns are present and valid strings.
        """
        # 1. Leakage Prevention: Drop retrieval columns
        retrieval_cols = [c for c in df.columns if c.endswith("_at_retrieval")]
        if retrieval_cols:
            df = df.drop(columns=retrieval_cols)

        # 2. Enforce Edit-Aware Text: Drop raw request_text if it exists
        if "request_text" in df.columns:
            df = df.drop(columns=["request_text"])

        # 3. Type Safety: Ensure text columns exist and are strings
        for col in Config.TEXT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
            else:
                # Initialize missing text columns with empty string
                df[col] = ""

        return df

    @staticmethod
    def load_union_data(
        load_cached_data: bool = True, debug_size: int = None
    ) -> pd.DataFrame:
        """
        Loads the Training and Validation sets, merges them into a Union Dataset,
        cleans leakage features, and caches the result.

        Args:
            load_cached_data: If True, attempts to load from disk cache first.
            debug_size: If provided, samples the dataset to this size for debugging.

        Returns:
            pd.DataFrame: The processed Union Dataset.
        """
        cache_path = Config.CACHE_PROCESSED_DATA_PATH
        df = None

        # Attempt to load from cache
        if load_cached_data:
            df = load_from_cache(cache_path)
            if df is not None:
                print(f"Loaded union data from cache: {cache_path}")

        # If not in cache or reload forced, process from scratch
        if df is None:
            print_header("Constructing Union Dataset")

            if not os.path.exists(Config.TRAIN_PATH) or not os.path.exists(
                Config.VAL_PATH
            ):
                raise FileNotFoundError("Train or Validation metadata files not found.")

            train_df = pd.read_parquet(Config.TRAIN_PATH)
            val_df = pd.read_parquet(Config.VAL_PATH)

            print(f"Raw Train shape: {train_df.shape}")
            print(f"Raw Val shape: {val_df.shape}")

            # Merge into Union Dataset
            df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

            # Apply cleaning
            df = DataFactory._clean_dataframe(df)

            # Save to cache (save the full dataset, not the debug sample)
            save_to_cache(df, cache_path)
            print(f"Saved union data to cache: {cache_path}")
            print(f"Union Dataset shape: {df.shape}")

        # Apply debug sampling if requested
        if debug_size is not None and debug_size > 0:
            if len(df) > debug_size:
                df = df.sample(n=debug_size, random_state=Config.SEED).reset_index(
                    drop=True
                )
                print(f"Debug mode: Sampled Union Dataset to {len(df)} rows")

        return df

    @staticmethod
    def load_test_data(debug_size: int = None) -> pd.DataFrame:
        """
        Loads the Test set and applies the same cleaning rules.

        Args:
            debug_size: If provided, samples the dataset to this size for debugging.

        Returns:
            pd.DataFrame: The processed Test Dataset.
        """
        print_header("Loading Test Data")

        if not os.path.exists(Config.TEST_PATH):
            raise FileNotFoundError("Test metadata file not found.")

        test_df = pd.read_parquet(Config.TEST_PATH)

        # Apply cleaning
        test_df = DataFactory._clean_dataframe(test_df)

        print(f"Test Data shape: {test_df.shape}")

        # Apply debug sampling if requested
        if debug_size is not None and debug_size > 0:
            if len(test_df) > debug_size:
                test_df = test_df.sample(
                    n=debug_size, random_state=Config.SEED
                ).reset_index(drop=True)
                print(f"Debug mode: Sampled Test Dataset to {len(test_df)} rows")

        return test_df
