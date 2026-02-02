import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config


class FeatureEngineer:
    """
    Handles feature engineering including string decomposition and
    computation of aggregate set-theoretic properties.
    """

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Decompose f_27 into fixed character positions
        # f_27 is a string of length 10
        if "f_27" in df.columns:
            # Vectorized slicing is faster than apply
            for i in range(Config.F27_LENGTH):
                col_name = f"{Config.F27_PREFIX}_{i}"
                df[col_name] = df["f_27"].str[i]

            # 2. Compute Aggregate Features
            # unique_character_count - Cite solution_lesson_node_00103
            df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(x)))

            # Removed frequency features - Cite solution_lesson_node_00101

        # 3. Ensure base categorical features are treated as strings for OrdinalEncoder
        for col in Config.BASE_CAT_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df


class DataProcessor:
    """
    Manages data loading, preprocessing, encoding, scaling, and caching.
    """

    @staticmethod
    def process_data(load_cached_data: bool = True):
        """
        Main entry point to get processed data.

        Args:
            load_cached_data (bool): If True, attempts to load from disk.

        Returns:
            tuple: (train_df, val_df, test_df, vocab_sizes)
                vocab_sizes is a dict {col_name: size}
        """
        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # File paths for cache
        train_cache = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
        val_cache = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
        test_cache = os.path.join(Config.CACHE_DIR, "test_processed.parquet")
        vocab_cache = os.path.join(Config.CACHE_DIR, "vocab_sizes.parquet")

        # 1. Try Loading Cache
        if load_cached_data:
            if (
                os.path.exists(train_cache)
                and os.path.exists(val_cache)
                and os.path.exists(test_cache)
                and os.path.exists(vocab_cache)
            ):

                print("Loading processed data from cache...")
                try:
                    train_df = pd.read_parquet(train_cache)
                    val_df = pd.read_parquet(val_cache)
                    test_df = pd.read_parquet(test_cache)

                    vocab_df = pd.read_parquet(vocab_cache)
                    vocab_sizes = dict(zip(vocab_df["feature"], vocab_df["size"]))

                    return train_df, val_df, test_df, vocab_sizes
                except Exception as e:
                    print(f"Failed to load cache: {e}. Recomputing...")
            else:
                print("Cache missing. Recomputing...")
        else:
            print("Force recompute enabled...")

        # 2. Load Raw Data
        print("Loading raw data...")
        train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
        val_df = pd.read_csv(Config.VAL_DATA_PATH)
        test_df = pd.read_csv(Config.TEST_DATA_PATH)

        # 3. Feature Engineering
        print("Applying feature engineering...")
        train_df = FeatureEngineer.transform(train_df)
        val_df = FeatureEngineer.transform(val_df)
        test_df = FeatureEngineer.transform(test_df)

        # 4. Transductive Ordinal Encoding
        print("Fitting Transductive Ordinal Encoder...")
        cat_features = Config.get_all_cat_features()

        # Concatenate all to fit encoder
        # We only need the categorical columns for fitting
        all_cats = pd.concat(
            [train_df[cat_features], val_df[cat_features], test_df[cat_features]],
            axis=0,
        )

        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoder.fit(all_cats)

        # Transform
        train_df[cat_features] = encoder.transform(train_df[cat_features])
        val_df[cat_features] = encoder.transform(val_df[cat_features])
        test_df[cat_features] = encoder.transform(test_df[cat_features])

        # Calculate Vocab Sizes (max index + 1)
        # Since we fit on all data, the categories list in encoder has all unique values
        vocab_sizes = {}
        for i, col in enumerate(cat_features):
            vocab_sizes[col] = len(encoder.categories_[i])

        # 5. Continuous Normalization
        print("Fitting Standard Scaler on Training Data...")
        cont_features = Config.get_all_cont_features()

        scaler = StandardScaler()
        scaler.fit(train_df[cont_features])

        train_df[cont_features] = scaler.transform(train_df[cont_features])
        val_df[cont_features] = scaler.transform(val_df[cont_features])
        test_df[cont_features] = scaler.transform(test_df[cont_features])

        # 6. Save to Cache
        print("Saving processed data to cache...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

        # Save vocab sizes
        vocab_df = pd.DataFrame(list(vocab_sizes.items()), columns=["feature", "size"])
        vocab_df.to_parquet(vocab_cache, index=False)

        return train_df, val_df, test_df, vocab_sizes


class ManufacturingDataset(Dataset):
    """
    PyTorch Dataset for the Manufacturing Control task.
    Returns (continuous_features, categorical_features, target).
    """

    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        self.is_test = is_test

        # Feature Lists
        self.cont_features = Config.get_all_cont_features()
        self.cat_features = Config.get_all_cat_features()

        # Prepare Data
        # Continuous: Float32
        self.cont_data = torch.tensor(
            df[self.cont_features].values, dtype=torch.float32
        )

        # Categorical: Long (Int64)
        self.cat_data = torch.tensor(df[self.cat_features].values, dtype=torch.long)

        # Target: Float32 (for BCEWithLogitsLoss)
        if not self.is_test:
            if "target" in df.columns:
                self.targets = torch.tensor(
                    df["target"].values, dtype=torch.float32
                ).unsqueeze(
                    1
                )  # Shape (N, 1)
            else:
                raise ValueError("Target column missing in training/validation data")
        else:
            self.targets = None

        # Store IDs for submission if needed, though usually handled outside
        self.ids = df["id"].values if "id" in df.columns else None

    def __len__(self):
        return len(self.cont_data)

    def __getitem__(self, idx):
        cont_x = self.cont_data[idx]
        cat_x = self.cat_data[idx]

        if self.is_test:
            return cont_x, cat_x
        else:
            y = self.targets[idx]
            return cont_x, cat_x, y
