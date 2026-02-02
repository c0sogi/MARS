import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from library.config import CACHE_DIR, METADATA_DIR


class ManufacturingPreprocessor:
    def __init__(self):
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int32
        )
        self.scaler = StandardScaler()
        self.cat_cols = []
        self.cont_cols = []
        self.cat_cardinalities = []

    def _engineer_features(self, df):
        """
        Decomposes f_27 and adds unique_char_count.
        Returns a new dataframe with engineered features.
        """
        df = df.copy()

        # f_27 decomposition
        if "f_27" in df.columns:
            # Vectorized string slicing
            for i in range(10):
                df[f"ch_{i}"] = df["f_27"].str[i]

            # unique_character_count
            df["unique_char_count"] = df["f_27"].apply(lambda x: len(set(x)))

        return df

    def fit(self, train_df, val_df, test_df):
        """
        Fits the encoder on Train+Val+Test (transductive) and Scaler on Train only.
        """
        # Engineer features temporarily to identify columns and fit encoders
        train_eng = self._engineer_features(train_df)
        val_eng = self._engineer_features(val_df)
        test_eng = self._engineer_features(test_df)

        # Define columns based on the engineered structure
        # Continuous: f_00 to f_26, f_28, unique_char_count
        self.cont_cols = [f"f_{i:02d}" for i in range(27)] + [
            "f_28",
            "unique_char_count",
        ]

        # Categorical: ch_0 to ch_9, f_29, f_30
        self.cat_cols = [f"ch_{i}" for i in range(10)] + ["f_29", "f_30"]

        # Transductive Vocabulary Alignment (Train + Val + Test)
        full_cat = pd.concat(
            [train_eng[self.cat_cols], val_eng[self.cat_cols], test_eng[self.cat_cols]],
            axis=0,
        )

        self.encoder.fit(full_cat)

        # Calculate cardinalities based on the full vocabulary
        self.cat_cardinalities = [int(full_cat[col].nunique()) for col in self.cat_cols]

        # Normalization (Fit on Train only)
        self.scaler.fit(train_eng[self.cont_cols])

        return self

    def transform(self, df):
        """
        Applies feature engineering, encoding, and scaling.
        """
        df_eng = self._engineer_features(df)

        # Transform Categorical
        df_eng[self.cat_cols] = self.encoder.transform(df_eng[self.cat_cols])

        # Transform Continuous
        df_eng[self.cont_cols] = self.scaler.transform(df_eng[self.cont_cols])

        # Cast types for memory efficiency and model compatibility
        for col in self.cont_cols:
            df_eng[col] = df_eng[col].astype(np.float32)

        for col in self.cat_cols:
            df_eng[col] = df_eng[col].astype(np.int32)

        return df_eng


def process_data(load_cached_data=True):
    """
    Orchestrates data loading, preprocessing, and caching.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")
    meta_cache = os.path.join(CACHE_DIR, "metadata.npy")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
        and os.path.exists(meta_cache)
    ):
        print("Loading data from cache...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        metadata = np.load(meta_cache, allow_pickle=True).item()
        return train_df, val_df, test_df, metadata

    print("Processing data from scratch...")
    # Load metadata splits
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Metadata file not found: {train_path}")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Instantiate and run preprocessor
    preprocessor = ManufacturingPreprocessor()
    preprocessor.fit(df_train, df_val, df_test)

    df_train_proc = preprocessor.transform(df_train)
    df_val_proc = preprocessor.transform(df_val)
    df_test_proc = preprocessor.transform(df_test)

    # Save to cache
    print("Saving data to cache...")
    df_train_proc.to_parquet(train_cache)
    df_val_proc.to_parquet(val_cache)
    df_test_proc.to_parquet(test_cache)

    metadata = {
        "cont_cols": preprocessor.cont_cols,
        "cat_cols": preprocessor.cat_cols,
        "cat_cardinalities": preprocessor.cat_cardinalities,
    }
    np.save(meta_cache, metadata)

    return df_train_proc, df_val_proc, df_test_proc, metadata
