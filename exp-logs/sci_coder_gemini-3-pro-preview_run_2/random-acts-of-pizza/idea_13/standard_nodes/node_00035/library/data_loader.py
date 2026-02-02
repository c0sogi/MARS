import os
import json
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import setup_logger


class PizzaDataLoader:
    """
    Data Loader for the Pizza Request Dataset.
    Handles loading raw JSON data, merging with metadata splits,
    validation, and caching using Parquet.
    """

    def __init__(self):
        self.logger = setup_logger("data_loader")

    def load_data(self, load_cached_data: bool = True):
        """
        Loads the training, validation, and test datasets.

        Args:
            load_cached_data (bool): If True, attempts to load from Parquet cache.
                                     If False or cache missing, reprocesses raw data.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # 1. Try loading from cache
        if load_cached_data:
            if self._check_cache_exists():
                self.logger.info("Loading data from cache...")
                try:
                    train_df = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
                    val_df = pd.read_parquet(Config.VAL_FEATURES_PATH)
                    test_df = pd.read_parquet(Config.TEST_FEATURES_PATH)

                    self.logger.info("Successfully loaded data from cache.")
                    return train_df, val_df, test_df
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load cache: {e}. Reprocessing from scratch."
                    )
            else:
                self.logger.info("Cache files not found. Processing from scratch...")
        else:
            self.logger.info("load_cached_data is False. Processing from scratch...")

        # 2. Process raw data
        train_df, val_df, test_df = self._process_raw_data()

        # 3. Save to cache
        self.logger.info("Saving processed data to cache...")
        try:
            # Ensure directory exists
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            train_df.to_parquet(Config.TRAIN_FEATURES_PATH, index=False)
            val_df.to_parquet(Config.VAL_FEATURES_PATH, index=False)
            test_df.to_parquet(Config.TEST_FEATURES_PATH, index=False)
            self.logger.info(f"Data saved to {Config.WORKING_DIR}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

        return train_df, val_df, test_df

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        return (
            os.path.exists(Config.TRAIN_FEATURES_PATH)
            and os.path.exists(Config.VAL_FEATURES_PATH)
            and os.path.exists(Config.TEST_FEATURES_PATH)
        )

    def _process_raw_data(self):
        """
        Reads raw JSON and metadata CSVs, merges them, and validates schema.
        """
        self.logger.info("Reading metadata files...")
        meta_train = pd.read_csv(Config.TRAIN_META_PATH)
        meta_val = pd.read_csv(Config.VAL_META_PATH)
        meta_test = pd.read_csv(Config.TEST_META_PATH)

        self.logger.info("Reading raw JSON files...")
        with open(Config.TRAIN_JSON_PATH, "r") as f:
            raw_train_list = json.load(f)
        df_raw_train = pd.DataFrame(raw_train_list)

        with open(Config.TEST_JSON_PATH, "r") as f:
            raw_test_list = json.load(f)
        df_raw_test = pd.DataFrame(raw_test_list)

        # Define columns to keep
        # We need the ID, the text columns, and the numeric features defined in Config
        cols_to_keep = list(set(Config.TEXT_COLS + Config.NUMERIC_FEATURES))

        # Ensure request_id is included for merging
        if "request_id" not in cols_to_keep:
            cols_to_keep.append("request_id")

        # Filter raw dataframes to only necessary columns to save memory
        train_cols_available = [c for c in cols_to_keep if c in df_raw_train.columns]
        test_cols_available = [c for c in cols_to_keep if c in df_raw_test.columns]

        df_raw_train_subset = df_raw_train[train_cols_available]
        df_raw_test_subset = df_raw_test[test_cols_available]

        self.logger.info("Merging metadata with raw data...")

        # Merge Train
        # Metadata has: request_id, sample_index, source_file, requester_received_pizza
        train_merged = meta_train.merge(
            df_raw_train_subset, on="request_id", how="left"
        )

        # Merge Val
        val_merged = meta_val.merge(df_raw_train_subset, on="request_id", how="left")

        # Merge Test
        # Metadata has: request_id, sample_index, source_file
        test_merged = meta_test.merge(df_raw_test_subset, on="request_id", how="left")

        # Validation
        self._validate_dataframe(train_merged, "Train", expect_target=True)
        self._validate_dataframe(val_merged, "Validation", expect_target=True)
        self._validate_dataframe(test_merged, "Test", expect_target=False)

        return train_merged, val_merged, test_merged

    def _validate_dataframe(self, df, name, expect_target=True):
        """
        Validates the integrity of the dataframe.
        """
        self.logger.info(f"Validating {name} dataframe...")

        # 1. Check Row Count
        self.logger.info(f"{name} row count: {len(df)}")

        # 2. Check Target
        if expect_target:
            if "requester_received_pizza" not in df.columns:
                raise ValueError(
                    f"Target column 'requester_received_pizza' missing in {name} set."
                )
            if df["requester_received_pizza"].isnull().any():
                raise ValueError(f"NaN values found in target column for {name} set.")
            # Ensure target is integer
            if not pd.api.types.is_integer_dtype(df["requester_received_pizza"]):
                try:
                    df["requester_received_pizza"] = df[
                        "requester_received_pizza"
                    ].astype(int)
                except Exception as e:
                    raise ValueError(
                        f"Target column in {name} is not integer and cannot be converted: {e}"
                    )

        # 3. Check Feature Columns
        required_cols = Config.NUMERIC_FEATURES + Config.TEXT_COLS
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            self.logger.error(f"Missing columns in {name}: {missing_cols}")
            raise ValueError(f"Required columns missing in {name} dataset.")

        # 4. Handle Nulls in Text Features
        for col in Config.TEXT_COLS:
            if col in df.columns:
                if df[col].isnull().any():
                    self.logger.info(
                        f"Found NaNs in text column {col} of {name}. Filling with empty string."
                    )
                    df[col] = df[col].fillna("")

        self.logger.info(f"{name} validation passed.")
