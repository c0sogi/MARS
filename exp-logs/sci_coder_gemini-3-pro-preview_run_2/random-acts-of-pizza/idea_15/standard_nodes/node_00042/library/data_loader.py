import os
import json
import pandas as pd
import numpy as np
from library.config import Config


class DataLoader:
    """
    Handles loading of raw JSON data and metadata, merging them into structured DataFrames,
    and managing caching to Parquet files for efficiency.
    """

    @staticmethod
    def load_data(load_cached_data=True):
        """
        Loads the training, validation, and test datasets.

        Args:
            load_cached_data (bool): If True, attempts to load pre-processed data from Parquet files.
                                     If False or if files are missing, re-processes raw data.

        Returns:
            tuple: (df_train, df_val, df_test) containing the processed features and labels.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Try Loading from Cache
        if load_cached_data:
            if (
                os.path.exists(Config.TRAIN_FEATURES_PATH)
                and os.path.exists(Config.VAL_FEATURES_PATH)
                and os.path.exists(Config.TEST_FEATURES_PATH)
            ):

                print("Loading data from cache...")
                df_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
                df_val = pd.read_parquet(Config.VAL_FEATURES_PATH)
                df_test = pd.read_parquet(Config.TEST_FEATURES_PATH)
                return df_train, df_val, df_test
            else:
                print("Cache not found. Processing raw data...")
        else:
            print("Ignoring cache. Processing raw data...")

        # 2. Load Raw Data
        print("Loading raw JSON files...")
        with open(Config.TRAIN_JSON_PATH, "r") as f:
            raw_train_data = json.load(f)

        with open(Config.TEST_JSON_PATH, "r") as f:
            raw_test_data = json.load(f)

        # 3. Load Metadata
        print("Loading metadata splits...")
        meta_train = pd.read_csv(Config.TRAIN_META_PATH)
        meta_val = pd.read_csv(Config.VAL_META_PATH)
        meta_test = pd.read_csv(Config.TEST_META_PATH)

        # 4. Helper to process splits
        def process_split(meta_df, raw_data_source, is_test=False):
            # Extract records using sample_index
            indices = meta_df["sample_index"].values
            records = [raw_data_source[i] for i in indices]

            # Create DataFrame from records
            df = pd.DataFrame(records)

            # Define columns to keep
            cols_to_keep = ["request_id"] + Config.TEXT_COLS + Config.NUMERIC_COLS

            # Ensure columns exist (handle potential missing keys in JSON though unlikely based on analysis)
            for col in cols_to_keep:
                if col not in df.columns:
                    df[col] = np.nan

            # Filter DataFrame
            df = df[cols_to_keep].copy()

            # Add target variable from metadata (source of truth for labels)
            if not is_test:
                df["requester_received_pizza"] = meta_df[
                    "requester_received_pizza"
                ].values.astype(int)

            # Type casting for consistency
            for col in Config.NUMERIC_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            for col in Config.TEXT_COLS:
                df[col] = df[col].fillna("").astype(str)

            return df

        # 5. Process datasets
        print("Processing Training Set...")
        df_train = process_split(meta_train, raw_train_data, is_test=False)

        print("Processing Validation Set...")
        df_val = process_split(meta_val, raw_train_data, is_test=False)

        print("Processing Test Set...")
        df_test = process_split(meta_test, raw_test_data, is_test=True)

        # 6. Save to Cache
        print("Saving processed data to cache...")
        df_train.to_parquet(Config.TRAIN_FEATURES_PATH, index=False)
        df_val.to_parquet(Config.VAL_FEATURES_PATH, index=False)
        df_test.to_parquet(Config.TEST_FEATURES_PATH, index=False)

        return df_train, df_val, df_test
