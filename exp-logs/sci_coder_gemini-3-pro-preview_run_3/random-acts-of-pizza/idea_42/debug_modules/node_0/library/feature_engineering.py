import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library import config, utils

# Initialize logger
logger = utils.setup_logging("feature_engineering")


class StaticFeatureExtractor:
    """
    Handles deterministic feature extraction that is independent of cross-validation splits.
    This includes generating dense embeddings using pre-trained models and extracting
    raw metadata fields. Results are cached to disk to optimize runtime.
    """

    def __init__(self):
        self.embedding_model = None
        # Explicit Allow-List for Augmented Global Metadata
        self.meta_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
            "unix_timestamp_of_request_utc",
        ]

    def _get_embedding_model(self):
        """Lazy loader for the heavy embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL_NAME, device=config.DEVICE
            )
        return self.embedding_model

    def _generate_embeddings(self, text_series):
        """Generates dense embeddings for a text series."""
        model = self._get_embedding_model()
        # Encode in batches
        embeddings = model.encode(
            text_series.tolist(),
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def extract(self, df, cache_prefix, load_cached_data=True):
        """
        Extracts static features (embeddings and metadata) with caching.

        Args:
            df (pd.DataFrame): Input dataframe containing text and metadata.
            cache_prefix (str): Unique identifier for caching (e.g., 'train', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing:
                - 'embeddings': np.ndarray (Dense semantic features)
                - 'metadata': pd.DataFrame (Raw numerical metadata)
        """
        cache_dir = config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        emb_path = os.path.join(cache_dir, f"{cache_prefix}_semantic.npy")
        meta_path = os.path.join(cache_dir, f"{cache_prefix}_metadata.parquet")

        # 1. Try Load from Cache
        if load_cached_data and os.path.exists(emb_path) and os.path.exists(meta_path):
            logger.info(f"Loading static features from cache: {cache_prefix}")
            embeddings = utils.load_data_cache(emb_path)
            metadata = utils.load_data_cache(meta_path)
            return {"embeddings": embeddings, "metadata": metadata}

        # 2. Compute from Scratch
        logger.info(f"Computing static features for {cache_prefix}...")

        # A. Dense Embeddings
        # Ensure text_combined exists (created in data_loader)
        if "text_combined" not in df.columns:
            raise ValueError("Column 'text_combined' missing from DataFrame.")

        logger.info("Generating dense embeddings...")
        embeddings = self._generate_embeddings(df["text_combined"])

        # B. Metadata Extraction
        logger.info("Extracting global metadata...")
        # Select columns, filling missing columns with 0 (safe default for counts/stats)
        # Note: Actual imputation happens in DynamicFeatureExtractor
        metadata = df[self.meta_cols].copy()
        # Ensure numeric types
        for col in self.meta_cols:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

        # 3. Save to Cache
        logger.info(f"Saving static features to cache: {cache_prefix}")
        utils.save_data_cache(embeddings, emb_path)
        utils.save_data_cache(metadata, meta_path)

        return {"embeddings": embeddings, "metadata": metadata}


class DynamicFeatureExtractor:
    """
    Handles features that must be fitted on the training fold to prevent leakage.
    - Sparse TF-IDF on Text
    - Latent Community Injection (NMF on Subreddit History)
    - Metadata Imputation and Scaling
    """

    def __init__(self):
        # Text Modality
        self.text_tfidf = TfidfVectorizer(**config.TFIDF_PARAMS)

        # Behavioral Modality (Community Injection)
        # We treat subreddits as tokens in a document
        self.subreddit_tfidf = TfidfVectorizer(
            max_features=config.TOP_K_SUBREDDITS,
            token_pattern=r"(?u)\b\w+\b",  # Capture subreddit names as tokens
            sublinear_tf=True,
        )
        self.subreddit_nmf = NMF(
            n_components=config.NMF_N_COMPONENTS,
            random_state=config.SEED,
            init="nndsvd",
        )

        # Metadata Modality
        self.meta_imputer = SimpleImputer(strategy="median")
        self.meta_scaler = StandardScaler()

    def _process_subreddits(self, subreddits_series):
        """
        Converts a Series of lists (subreddits) into a Series of space-separated strings.
        Handles NaNs and empty lists.
        """

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            return ""

        return subreddits_series.apply(join_subs)

    def fit(self, df, metadata_df):
        """
        Fits the dynamic feature extractors on the training data.

        Args:
            df (pd.DataFrame): Training dataframe with text and subreddits.
            metadata_df (pd.DataFrame): Extracted metadata for training.
        """
        # 1. Fit Text TF-IDF
        logger.info("Fitting Text TF-IDF...")
        self.text_tfidf.fit(df["text_combined"])

        # 2. Fit Subreddit NMF Pipeline
        logger.info("Fitting Subreddit NMF...")
        subs_str = self._process_subreddits(df["requester_subreddits_at_request"])
        X_subs_tfidf = self.subreddit_tfidf.fit_transform(subs_str)
        self.subreddit_nmf.fit(X_subs_tfidf)

        # 3. Fit Metadata Preprocessors
        logger.info("Fitting Metadata Imputer and Scaler...")
        self.meta_imputer.fit(metadata_df)
        # Transform to fit scaler
        meta_imputed = self.meta_imputer.transform(metadata_df)
        self.meta_scaler.fit(meta_imputed)

        return self

    def transform(self, df, metadata_df):
        """
        Transforms data using the fitted extractors.

        Args:
            df (pd.DataFrame): Dataframe to transform.
            metadata_df (pd.DataFrame): Metadata to transform.

        Returns:
            dict: Dictionary containing transformed features:
                - 'X_lexical': Sparse matrix (Text TF-IDF)
                - 'X_behavioral_sparse': Sparse matrix (Subreddit TF-IDF)
                - 'X_community_latent': np.ndarray (NMF Topics)
                - 'X_metadata_scaled': np.ndarray (Scaled Metadata)
        """
        # 1. Transform Text
        X_lexical = self.text_tfidf.transform(df["text_combined"])

        # 2. Transform Subreddits
        subs_str = self._process_subreddits(df["requester_subreddits_at_request"])
        X_behavioral_sparse = self.subreddit_tfidf.transform(subs_str)
        X_community_latent = self.subreddit_nmf.transform(X_behavioral_sparse)

        # 3. Transform Metadata
        meta_imputed = self.meta_imputer.transform(metadata_df)
        X_metadata_scaled = self.meta_scaler.transform(meta_imputed)

        return {
            "X_lexical": X_lexical,
            "X_behavioral_sparse": X_behavioral_sparse,
            "X_community_latent": X_community_latent,
            "X_metadata_scaled": X_metadata_scaled,
        }
