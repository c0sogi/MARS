import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from library.config import Config
from library.utils import set_seed


def decompose_f27(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the 'f_27' string column into 10 separate character columns
    named 'ch_0' through 'ch_9'.
    """
    # Ensure f_27 is string
    s = df["f_27"].astype(str)
    # create a dataframe of characters
    chars = s.apply(lambda x: list(x) if len(x) >= 10 else list(x.ljust(10)))
    chars_df = pd.DataFrame(chars.tolist(), index=df.index)
    chars_df.columns = [f"ch_{i}" for i in range(10)]

    # Concatenate with original df
    df = pd.concat([df, chars_df], axis=1)
    return df


def add_unique_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'unique_character_count' column representing the number of
    unique characters in 'f_27'.
    """
    df["unique_character_count"] = df["f_27"].apply(lambda x: len(set(str(x))))
    return df


class DataProcessor:
    """
    Handles data loading, feature engineering, preprocessing, and caching.
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.cache_dir = Config.CACHE_DIR

        # Define file paths for cache
        self.train_cache = os.path.join(self.working_dir, "train_processed.parquet")
        self.val_cache = os.path.join(self.working_dir, "val_processed.parquet")
        self.test_cache = os.path.join(self.working_dir, "test_processed.parquet")
        self.vocab_cache = os.path.join(self.working_dir, "vocab_sizes.npy")

        # Categorical columns: 10 chars from f_27 + f_29 + f_30
        self.cat_cols = [f"ch_{i}" for i in range(10)] + ["f_29", "f_30"]

        # Continuous columns defined in Config
        self.cont_cols = Config.CONTINUOUS_FEATURE_NAMES

    def _load_raw_data(self):
        """Loads raw data from metadata paths."""
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
        return train_df, val_df, test_df

    def _apply_feature_engineering(self, df):
        """Applies decomposition and unique count engineering."""
        df = decompose_f27(df)
        df = add_unique_count(df)
        return df

    def process_data(self, load_cached_data: bool = True):
        """
        Main processing pipeline.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (train_df, val_df, test_df, vocab_sizes)
        """
        set_seed()
        os.makedirs(self.working_dir, exist_ok=True)

        # 1. Check Cache
        if load_cached_data:
            if (
                os.path.exists(self.train_cache)
                and os.path.exists(self.val_cache)
                and os.path.exists(self.test_cache)
                and os.path.exists(self.vocab_cache)
            ):

                # print("Loading data from cache...")
                train_df = pd.read_parquet(self.train_cache)
                val_df = pd.read_parquet(self.val_cache)
                test_df = pd.read_parquet(self.test_cache)
                vocab_sizes = np.load(self.vocab_cache)
                return train_df, val_df, test_df, vocab_sizes

        # 2. Compute from Scratch
        # print("Processing data from scratch...")
        train_df, val_df, test_df = self._load_raw_data()

        # Debugging subset
        if Config.DEBUG:
            train_df = train_df.iloc[: Config.DEBUG_SAMPLES]
            val_df = val_df.iloc[: Config.DEBUG_SAMPLES]
            test_df = test_df.iloc[: Config.DEBUG_SAMPLES]

        # Apply Feature Engineering
        train_df = self._apply_feature_engineering(train_df)
        val_df = self._apply_feature_engineering(val_df)
        test_df = self._apply_feature_engineering(test_df)

        # 3. Transductive Vocabulary Alignment (Categorical)
        # Concatenate all to fit encoder
        # Note: f_29 and f_30 are int, chars are str. OrdinalEncoder handles mixed types if converted to str or handled carefully.
        # We will convert categorical columns to string to be safe and consistent.

        for col in self.cat_cols:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

        all_cats = pd.concat(
            [train_df[self.cat_cols], val_df[self.cat_cols], test_df[self.cat_cols]],
            axis=0,
        )

        encoder = OrdinalEncoder(
            dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoder.fit(all_cats)

        # Transform
        train_df[self.cat_cols] = encoder.transform(train_df[self.cat_cols])
        val_df[self.cat_cols] = encoder.transform(val_df[self.cat_cols])
        test_df[self.cat_cols] = encoder.transform(test_df[self.cat_cols])

        # Get vocab sizes (max index + 1 for each column)
        # Since we fit on everything, max index is len(categories) - 1
        vocab_sizes = [len(cats) for cats in encoder.categories_]
        vocab_sizes = np.array(vocab_sizes)

        # 4. Normalization (Continuous)
        scaler = StandardScaler()
        scaler.fit(train_df[self.cont_cols])

        train_df[self.cont_cols] = scaler.transform(train_df[self.cont_cols])
        val_df[self.cont_cols] = scaler.transform(val_df[self.cont_cols])
        test_df[self.cont_cols] = scaler.transform(test_df[self.cont_cols])

        # 5. Save to Cache
        # Save as parquet for efficiency
        train_df.to_parquet(self.train_cache, index=False)
        val_df.to_parquet(self.val_cache, index=False)
        test_df.to_parquet(self.test_cache, index=False)
        np.save(self.vocab_cache, vocab_sizes)

        return train_df, val_df, test_df, vocab_sizes
