import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class TextPipeline:
    """
    Handles text feature extraction for View A (Text Expert).
    Generates 384-dimensional embeddings using a frozen transformer
    and applies L2 normalization.
    """

    def __init__(self):
        self.model_name = Config.TRANSFORMER_MODEL
        self.text_cols = Config.TEXT_COLS
        self.cache_paths = {
            "train": Config.CACHE_TEXT_TRAIN,
            "val": Config.CACHE_TEXT_VAL,
            "test": Config.CACHE_TEXT_TEST,
        }

    def _get_sentences(self, df):
        """Helper to combine text columns into a single string per sample."""
        # Start with the first column, fill NaNs with empty string
        combined = df[self.text_cols[0]].fillna("").astype(str)
        # Append subsequent columns with a space separator
        for col in self.text_cols[1:]:
            combined = combined + " " + df[col].fillna("").astype(str)
        return combined.tolist()

    def execute(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates or loads text embeddings.

        Args:
            df_train, df_val, df_test: DataFrames for the respective splits.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (X_train, X_val, X_test) as numpy arrays.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Check if cache exists
        if load_cached_data and all(
            os.path.exists(p) for p in self.cache_paths.values()
        ):
            print("Loading text features from cache...")
            return (
                np.load(self.cache_paths["train"]),
                np.load(self.cache_paths["val"]),
                np.load(self.cache_paths["test"]),
            )

        print("Computing text features (Embeddings + L2 Norm)...")
        set_seed()

        # Load the frozen transformer model
        # sentence-transformers handles device placement automatically
        model = SentenceTransformer(self.model_name)

        # Helper to encode a dataframe
        def encode_split(df, name):
            print(f"  Encoding {name} set...")
            sentences = self._get_sentences(df)
            # normalize_embeddings=True applies L2 normalization
            return model.encode(
                sentences,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        X_train = encode_split(df_train, "Train")
        X_val = encode_split(df_val, "Validation")
        X_test = encode_split(df_test, "Test")

        # Save to cache
        print("Saving text features to cache...")
        np.save(self.cache_paths["train"], X_train)
        np.save(self.cache_paths["val"], X_val)
        np.save(self.cache_paths["test"], X_test)

        return X_train, X_val, X_test


class TabularPipeline:
    """
    Handles tabular feature extraction for View B (Metadata Expert).
    Imputes missing values and applies RankGauss (QuantileTransformer) normalization.
    """

    def __init__(self):
        self.cols = Config.NUMERICAL_COLS
        self.cache_paths = {
            "train": Config.CACHE_META_TRAIN,
            "val": Config.CACHE_META_VAL,
            "test": Config.CACHE_META_TEST,
        }

    def execute(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates or loads tabular features.

        Args:
            df_train, df_val, df_test: DataFrames for the respective splits.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (df_train, df_val, df_test) as pandas DataFrames.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Check if cache exists
        if load_cached_data and all(
            os.path.exists(p) for p in self.cache_paths.values()
        ):
            print("Loading tabular features from cache...")
            return (
                pd.read_parquet(self.cache_paths["train"]),
                pd.read_parquet(self.cache_paths["val"]),
                pd.read_parquet(self.cache_paths["test"]),
            )

        print("Computing tabular features (Imputation + RankGauss)...")
        set_seed()

        # Extract raw numerical data
        # We use .copy() to avoid SettingWithCopy warnings on the original DFs
        X_train_raw = df_train[self.cols].values
        X_val_raw = df_val[self.cols].values
        X_test_raw = df_test[self.cols].values

        # 1. Imputation (Median)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp = imputer.transform(X_val_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # 2. RankGauss (QuantileTransformer)
        # output_distribution='normal' transforms data to a Gaussian distribution
        scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )
        X_train_trans = scaler.fit_transform(X_train_imp)
        X_val_trans = scaler.transform(X_val_imp)
        X_test_trans = scaler.transform(X_test_imp)

        # Convert back to DataFrame to preserve indices and column names
        df_train_out = pd.DataFrame(
            X_train_trans, columns=self.cols, index=df_train.index
        )
        df_val_out = pd.DataFrame(X_val_trans, columns=self.cols, index=df_val.index)
        df_test_out = pd.DataFrame(X_test_trans, columns=self.cols, index=df_test.index)

        # Save to cache
        print("Saving tabular features to cache...")
        df_train_out.to_parquet(self.cache_paths["train"])
        df_val_out.to_parquet(self.cache_paths["val"])
        df_test_out.to_parquet(self.cache_paths["test"])

        return df_train_out, df_val_out, df_test_out


def extract_features(df_train, df_val, df_test, load_cached_data=True):
    """
    Main entry point to run both text and tabular pipelines.

    Args:
        df_train, df_val, df_test: DataFrames containing raw data.
        load_cached_data (bool): Whether to use cached features if available.

    Returns:
        tuple: (X_text_train, X_meta_train, X_text_val, X_meta_val, X_text_test, X_meta_test)
    """
    text_pipeline = TextPipeline()
    tabular_pipeline = TabularPipeline()

    # Execute Text Pipeline
    X_text_train, X_text_val, X_text_test = text_pipeline.execute(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )

    # Execute Tabular Pipeline
    X_meta_train, X_meta_val, X_meta_test = tabular_pipeline.execute(
        df_train, df_val, df_test, load_cached_data=load_cached_data
    )

    return (
        X_text_train,
        X_meta_train,
        X_text_val,
        X_meta_val,
        X_text_test,
        X_meta_test,
    )
