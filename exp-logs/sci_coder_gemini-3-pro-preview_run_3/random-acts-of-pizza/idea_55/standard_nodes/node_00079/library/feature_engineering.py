import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import load_or_process_data, set_seed


class FeaturePipeline:
    """
    Encapsulates the feature engineering logic including:
    - Sparse Lexical Features (TF-IDF on Text)
    - Sparse Behavioral Features (TF-IDF on Subreddit History)
    - Dense Semantic Features (Embeddings)
    - Contextual Metadata (Imputation + Scaling)
    """

    def __init__(self):
        # Lexical (Text) Vectorizer
        self.lexical_vectorizer = TfidfVectorizer(**Config.TEXT_VECTORIZER_PARAMS)

        # Behavioral (Community) Vectorizer
        self.community_vectorizer = TfidfVectorizer(
            **Config.COMMUNITY_VECTORIZER_PARAMS
        )

        # Metadata Preprocessing
        self.metadata_imputer = SimpleImputer(strategy="median")
        self.metadata_scaler = StandardScaler()

        # Placeholder for embedding model (lazy load)
        self._embedding_model = None
        self.embedding_model_name = Config.EMBEDDING_MODEL

    def __getstate__(self):
        """Exclude the heavy embedding model from pickling."""
        state = self.__dict__.copy()
        if "_embedding_model" in state:
            del state["_embedding_model"]
        return state

    def __setstate__(self, state):
        """Restore state and reset embedding model to None."""
        self.__dict__.update(state)
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazily load the SentenceTransformer model."""
        if self._embedding_model is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(
                f"Loading SentenceTransformer: {self.embedding_model_name} on {device}"
            )
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name, device=device
            )
        return self._embedding_model

    def _prepare_text(self, df):
        """Concatenate Title and Edit-Aware Body."""
        title = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        body = df[Config.TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _prepare_community(self, df):
        """Convert list of subreddits to space-separated string."""

        def join_subreddits(x):
            if isinstance(x, list):
                return " ".join(x)
            return str(x) if x is not None else ""

        return df[Config.COMMUNITY_COL].apply(join_subreddits)

    def fit(self, df):
        """Fit all stateful transformers on the training data."""
        logging.info("Fitting FeaturePipeline...")
        set_seed()

        # 1. Lexical
        text_data = self._prepare_text(df)
        self.lexical_vectorizer.fit(text_data)

        # 2. Behavioral
        community_data = self._prepare_community(df)
        self.community_vectorizer.fit(community_data)

        # 3. Metadata
        meta_data = df[Config.METADATA_COLS].copy()
        self.metadata_imputer.fit(meta_data)
        meta_imputed = self.metadata_imputer.transform(meta_data)
        self.metadata_scaler.fit(meta_imputed)

        logging.info("FeaturePipeline fitted.")
        return self

    def transform(self, df):
        """Transform data into feature arrays."""
        logging.info("Transforming data with FeaturePipeline...")
        set_seed()

        features = {}

        # 1. Lexical (Sparse -> Dense for storage compatibility)
        text_data = self._prepare_text(df)
        X_lexical = self.lexical_vectorizer.transform(text_data)
        features["X_lexical"] = X_lexical.toarray().astype(np.float32)

        # 2. Behavioral (Sparse -> Dense)
        community_data = self._prepare_community(df)
        X_behavioral = self.community_vectorizer.transform(community_data)
        features["X_behavioral"] = X_behavioral.toarray().astype(np.float32)

        # 3. Metadata (Dense)
        meta_data = df[Config.METADATA_COLS].copy()
        meta_imputed = self.metadata_imputer.transform(meta_data)
        features["X_metadata"] = self.metadata_scaler.transform(meta_imputed).astype(
            np.float32
        )

        # 4. Semantic (Dense Embeddings)
        model = self._get_embedding_model()
        text_list = text_data.tolist()
        logging.info(f"Generating embeddings for {len(text_list)} samples...")
        embeddings = model.encode(
            text_list,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        features["X_semantic"] = embeddings.astype(np.float32)

        return features


def get_features(df, split_name, pipeline=None, load_cache=True):
    """
    Orchestrates feature extraction with caching.

    Args:
        df (pd.DataFrame): Input dataframe.
        split_name (str): 'train', 'val', or 'test'.
        pipeline (FeaturePipeline): Fitted pipeline (required for val/test).
        load_cache (bool): Whether to use cached files.

    Returns:
        dict: Dictionary of feature arrays.
        FeaturePipeline: The fitted pipeline (only returned if split_name='train').
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    features_file_name = f"cache/{split_name}_features.npz"
    pipeline_path = os.path.join(
        Config.WORKING_DIR, "models", "feature_pipeline.joblib"
    )

    # Logic for Training Split (Pipeline Management)
    if split_name == "train":
        # Check if we can load everything from cache
        if (
            load_cache
            and os.path.exists(os.path.join(Config.WORKING_DIR, features_file_name))
            and os.path.exists(pipeline_path)
        ):
            logging.info("Loading cached training features and pipeline.")
            # Load Data
            features_npz = np.load(os.path.join(Config.WORKING_DIR, features_file_name))
            features = {k: features_npz[k] for k in features_npz.files}

            # Validate dimensions (Cite debug_lesson_2)
            cached_rows = features[next(iter(features))].shape[0]
            if cached_rows == len(df):
                # Load Pipeline
                pipeline = joblib.load(pipeline_path)
                return features, pipeline
            else:
                logging.warning(
                    f"Cache mismatch: Cached features have {cached_rows} rows, but input df has {len(df)}. Recomputing..."
                )

        # Compute from scratch
        logging.info("Computing training features from scratch.")
        pipeline = FeaturePipeline()
        pipeline.fit(df)

        # Save Pipeline
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(pipeline, pipeline_path)
        logging.info(f"Pipeline saved to {pipeline_path}")

        # Transform and Save Data
        features = pipeline.transform(df)

        # Use load_or_process_data's saving mechanism logic manually or via wrapper?
        # Since we already computed it, we just save it to match the cache expectation.
        # We use np.savez directly here to ensure it's saved correctly for future cached runs.
        full_path = os.path.join(Config.WORKING_DIR, features_file_name)
        np.savez(full_path, **features)
        logging.info(f"Features saved to {full_path}")

        return features, pipeline

    # Logic for Val/Test Splits
    else:
        if pipeline is None:
            # Try loading pipeline if not provided
            if os.path.exists(pipeline_path):
                pipeline = joblib.load(pipeline_path)
            else:
                raise ValueError(
                    f"Pipeline must be provided or exist at {pipeline_path} for {split_name} split."
                )

        def process_fn(**kwargs):
            return pipeline.transform(df)

        features_npz = load_or_process_data(
            file_name=features_file_name,
            process_fn=process_fn,
            load_cache=load_cache,
            file_type="npz",
        )

        # Convert NpzFile to dict
        if hasattr(features_npz, "files"):
            features = {k: features_npz[k] for k in features_npz.files}
        else:
            features = dict(features_npz)

        # Validate dimensions (Cite debug_lesson_2)
        cached_rows = features[next(iter(features))].shape[0]
        if cached_rows != len(df):
            logging.warning(
                f"Cache mismatch: Cached features have {cached_rows} rows, but input df has {len(df)}. Recomputing..."
            )
            features = process_fn()
            # Update cache
            full_path = os.path.join(Config.WORKING_DIR, features_file_name)
            np.savez(full_path, **features)
            logging.info(f"Features re-saved to {full_path}")

        return features
