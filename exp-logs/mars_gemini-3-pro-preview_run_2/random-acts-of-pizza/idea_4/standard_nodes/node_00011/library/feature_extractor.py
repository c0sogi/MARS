import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from library.config import Config
from library.data_loader import load_datasets


class TextEmbedder:
    """
    Generates dense vector embeddings from text using Sentence Transformers.
    """

    def __init__(
        self, model_name=Config.TEXT_MODEL_NAME, device=Config.DEVICE, batch_size=32
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    def transform(self, df):
        """
        Expects a DataFrame with 'request_title' and 'request_text_edit_aware'.
        Returns a numpy array of embeddings.
        """
        # Combine title and text for a richer representation
        # Handle potential missing values by filling with empty string
        titles = df["request_title"].fillna("").astype(str)
        texts = df["request_text_edit_aware"].fillna("").astype(str)

        # Concatenate with a separator
        combined_text = titles + " " + texts
        combined_text_list = combined_text.tolist()

        model = self._get_model()

        # Generate embeddings
        embeddings = model.encode(
            combined_text_list,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings


class TabularProcessor:
    """
    Handles selection, imputation, and scaling of numerical metadata.
    """

    def __init__(self):
        self.numeric_features = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            # "unix_timestamp_of_request", # Removed to replace with derived features
        ]
        self.derived_features = ["hour_of_request", "day_of_week_request"]
        self.all_features = self.numeric_features + self.derived_features

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()

    def _extract_features(self, df):
        """Extracts numeric and derived features."""
        # Base numeric features
        X = df[self.numeric_features].copy()

        # Derived temporal features
        # Convert timestamp to datetime
        dt = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
        X["hour_of_request"] = dt.dt.hour
        X["day_of_week_request"] = dt.dt.dayofweek

        return X

    def fit(self, df):
        """
        Fits imputer and scaler on the provided DataFrame (should be Train set).
        """
        X = self._extract_features(df)

        # Fit Imputer
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)

        # Fit Scaler
        self.scaler.fit(X_imputed)
        return self

    def transform(self, df):
        """
        Transforms the DataFrame using fitted imputer and scaler.
        Returns a DataFrame with the same index.
        """
        X = self._extract_features(df)
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return pd.DataFrame(X_scaled, columns=self.all_features, index=df.index)


def prepare_features(load_cached_data=True, logger=None, debug=False):
    """
    Orchestrates the feature extraction pipeline.

    Args:
        load_cached_data (bool): Whether to try loading from disk.
        logger (logging.Logger): Logger instance.
        debug (bool): Whether to run in debug mode (smaller dataset).

    Returns:
        tuple: (df_train, df_val, df_test) containing processed features and labels.
    """
    # Define cache paths
    train_path = Config.TRAIN_FEATURES_PATH
    val_path = Config.VAL_FEATURES_PATH
    test_path = Config.TEST_FEATURES_PATH

    # Check if all cache files exist
    cache_exists = (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
    )

    # 1. Try Loading from Cache
    if load_cached_data and cache_exists:
        if logger:
            logger.info("Loading features from cache...")
        try:
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)
            df_test = pd.read_parquet(test_path)
            return df_train, df_val, df_test
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute Features
    if logger:
        logger.info("Computing features from scratch...")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load raw datasets
    raw_train, raw_val, raw_test = load_datasets(logger=logger, debug=debug)

    # --- Text Processing ---
    if logger:
        logger.info("Generating text embeddings...")
    embedder = TextEmbedder()

    # Generate embeddings (returns numpy arrays)
    emb_train = embedder.transform(raw_train)
    emb_val = embedder.transform(raw_val)
    emb_test = embedder.transform(raw_test)

    # Convert embeddings to DataFrames
    emb_cols = [f"emb_{i}" for i in range(emb_train.shape[1])]
    df_emb_train = pd.DataFrame(emb_train, columns=emb_cols, index=raw_train.index)
    df_emb_val = pd.DataFrame(emb_val, columns=emb_cols, index=raw_val.index)
    df_emb_test = pd.DataFrame(emb_test, columns=emb_cols, index=raw_test.index)

    # --- Tabular Processing ---
    if logger:
        logger.info("Processing tabular metadata...")
    tab_processor = TabularProcessor()

    # Fit on Train ONLY
    tab_processor.fit(raw_train)

    # Transform all
    df_meta_train = tab_processor.transform(raw_train)
    df_meta_val = tab_processor.transform(raw_val)
    df_meta_test = tab_processor.transform(raw_test)

    # --- Combine Features ---
    if logger:
        logger.info("Combining features...")

    # Helper to combine
    def combine(meta_df, emb_df, raw_df, is_test=False):
        # Concatenate features
        combined = pd.concat([meta_df, emb_df], axis=1)

        # Add target if available
        if not is_test and "requester_received_pizza" in raw_df.columns:
            combined["requester_received_pizza"] = raw_df[
                "requester_received_pizza"
            ].values

        # Add request_id for tracking/submission
        combined["request_id"] = raw_df["request_id"].values
        return combined

    df_train_final = combine(df_meta_train, df_emb_train, raw_train)
    df_val_final = combine(df_meta_val, df_emb_val, raw_val)
    df_test_final = combine(df_meta_test, df_emb_test, raw_test, is_test=True)

    # 3. Save to Cache
    if logger:
        logger.info(f"Saving features to {Config.WORKING_DIR}...")

    df_train_final.to_parquet(train_path)
    df_val_final.to_parquet(val_path)
    df_test_final.to_parquet(test_path)

    return df_train_final, df_val_final, df_test_final
