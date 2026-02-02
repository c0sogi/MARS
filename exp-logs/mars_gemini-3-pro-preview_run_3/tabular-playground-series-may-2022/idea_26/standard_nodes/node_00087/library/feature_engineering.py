import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class FeatureEngineer:
    def __init__(self):
        """
        Initializes the FeatureEngineer.
        """
        pass

    def decompose_f27(self, df):
        """
        Decomposes the 'f_27' column into 10 separate character columns
        and adds a 'unique_character_count' feature.

        Args:
            df (pd.DataFrame): Input dataframe containing 'f_27'.

        Returns:
            pd.DataFrame: Dataframe with 'f_27' replaced by decomposed features.
        """
        if "f_27" not in df.columns:
            return df

        # Split string into characters (fixed length of 10)
        # We iterate 0-9 to create f_27_0 through f_27_9
        for i in range(10):
            df[f"f_27_{i}"] = df["f_27"].str[i]

        # Calculate unique character count for each string
        df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

        # Drop the original string column
        df = df.drop(columns=["f_27"])
        return df

    def process_data(self, load_cached_data=True, debug_sample_size=None):
        """
        Main data processing pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load processed data from disk.
            debug_sample_size (int, optional): If set, processes only a subset of data
                                               and disables cache saving.

        Returns:
            tuple: (train_df, val_df, test_df, vocab_sizes)
        """
        # Disable caching if we are in debug mode to avoid saving partial data
        if debug_sample_size is not None:
            load_cached_data = False

        # 1. Try to load from cache
        if load_cached_data:
            if (
                os.path.exists(Config.CACHE_TRAIN_PATH)
                and os.path.exists(Config.CACHE_VAL_PATH)
                and os.path.exists(Config.CACHE_TEST_PATH)
                and os.path.exists(Config.CACHE_VOCAB_PATH)
            ):

                print("Loading cached processed data...")
                train_df = pd.read_parquet(Config.CACHE_TRAIN_PATH)
                val_df = pd.read_parquet(Config.CACHE_VAL_PATH)
                test_df = pd.read_parquet(Config.CACHE_TEST_PATH)
                # Load vocab_sizes dict
                vocab_sizes = np.load(Config.CACHE_VOCAB_PATH, allow_pickle=True).item()
                return train_df, val_df, test_df, vocab_sizes

        print("Processing data from scratch...")

        # 2. Load Raw Data using Metadata paths
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Apply debug sampling if requested
        if debug_sample_size is not None:
            print(f"Debug mode: Sampling {debug_sample_size} rows.")
            train_df = train_df.iloc[:debug_sample_size].copy()
            val_df = val_df.iloc[:debug_sample_size].copy()
            test_df = test_df.iloc[:debug_sample_size].copy()

        # 3. Feature Engineering (Decomposition)
        train_df = self.decompose_f27(train_df)
        val_df = self.decompose_f27(val_df)
        test_df = self.decompose_f27(test_df)

        # Identify columns types
        # Categorical: The decomposed f_27 columns
        cat_cols = [f"f_27_{i}" for i in range(10)]

        # Continuous: All f_XX columns (excluding f_27 parts) + unique_character_count
        # We exclude id, target, source_path, and the categorical columns
        exclude_cols = {"id", "target", "source_path"} | set(cat_cols)
        cont_cols = [c for c in train_df.columns if c not in exclude_cols]

        # 4. Transductive Ordinal Encoding
        # We fit on Train + Val + Test to ensure all tokens are handled
        print("Performing transductive ordinal encoding...")
        all_cats = pd.concat(
            [train_df[cat_cols], val_df[cat_cols], test_df[cat_cols]], axis=0
        )

        encoder = OrdinalEncoder(dtype=np.int64)
        encoder.fit(all_cats)

        # Transform all splits
        train_df[cat_cols] = encoder.transform(train_df[cat_cols])
        val_df[cat_cols] = encoder.transform(val_df[cat_cols])
        test_df[cat_cols] = encoder.transform(test_df[cat_cols])

        # Determine vocabulary sizes for embedding layers
        vocab_sizes = {
            col: len(cats) for col, cats in zip(cat_cols, encoder.categories_)
        }

        # 5. Standard Scaling
        # Fit only on Train, transform all
        print("Scaling continuous features...")
        scaler = StandardScaler()
        scaler.fit(train_df[cont_cols])

        train_df[cont_cols] = scaler.transform(train_df[cont_cols])
        val_df[cont_cols] = scaler.transform(val_df[cont_cols])
        test_df[cont_cols] = scaler.transform(test_df[cont_cols])

        # Ensure categorical columns are integers
        for col in cat_cols:
            train_df[col] = train_df[col].astype("int64")
            val_df[col] = val_df[col].astype("int64")
            test_df[col] = test_df[col].astype("int64")

        # 6. Save to Cache (only if not debugging)
        if debug_sample_size is None:
            print(f"Saving processed data to {Config.WORKING_DIR}...")
            # Ensure directory exists (handled in Config, but good practice)
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            train_df.to_parquet(Config.CACHE_TRAIN_PATH, index=False)
            val_df.to_parquet(Config.CACHE_VAL_PATH, index=False)
            test_df.to_parquet(Config.CACHE_TEST_PATH, index=False)
            np.save(Config.CACHE_VOCAB_PATH, vocab_sizes)

        return train_df, val_df, test_df, vocab_sizes
