import os
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import set_seed, timer


class FeaturePipeline:
    def __init__(self):
        """
        Initializes the FeaturePipeline with necessary transformers and models.
        """
        set_seed(Config.SEED)

        # Metadata Preprocessing
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Sparse Feature Extractors
        self.lexical_vectorizer = TfidfVectorizer(**Config.LEXICAL_TFIDF_PARAMS)
        self.community_vectorizer = TfidfVectorizer(**Config.COMMUNITY_TFIDF_PARAMS)

        # Dense Feature Extractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            Config.EMBEDDING_MODEL_NAME, device=device
        )

        self.is_fitted = False

    def _process_community_data(self, community_series):
        """
        Converts a Series of subreddit lists into space-separated strings for TF-IDF.
        """
        # Handle cases where data might be a list or already a string
        return community_series.apply(
            lambda x: (
                " ".join(x)
                if isinstance(x, list)
                else (str(x) if x is not None else "")
            )
        )

    def fit(self, train_data):
        """
        Fits the internal transformers on the training data.

        Args:
            train_data (dict): Dictionary containing 'metadata', 'text', 'community'.
        """
        print("Fitting FeaturePipeline...")

        # 1. Fit Metadata Transformers
        meta_df = train_data["metadata"]
        self.imputer.fit(meta_df)
        meta_imputed = self.imputer.transform(meta_df)
        self.scaler.fit(meta_imputed)

        # 2. Fit Lexical Vectorizer (Text)
        text_series = train_data["text"]
        self.lexical_vectorizer.fit(text_series)

        # 3. Fit Behavioral Vectorizer (Community)
        community_processed = self._process_community_data(train_data["community"])
        self.community_vectorizer.fit(community_processed)

        self.is_fitted = True
        print("FeaturePipeline fitted successfully.")

    def transform(self, data, split_name, load_cached_data=True):
        """
        Transforms the input data into Lexical, Behavioral, Semantic, and Metadata views.
        Uses caching to speed up subsequent runs.

        Args:
            data (dict): Dictionary containing 'metadata', 'text', 'community'.
            split_name (str): 'train', 'val', or 'test' for cache naming.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            dict: Dictionary containing 'lexical', 'behavioral', 'semantic', 'metadata' features.
        """
        if not self.is_fitted and split_name != "train":
            # Note: Ideally fit() is called before transform() even for train,
            # but we check here to ensure safety.
            raise RuntimeError("FeaturePipeline must be fitted before transform.")

        # Define Cache Paths
        cache_dir = Config.CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        path_lexical = os.path.join(cache_dir, f"X_{split_name}_lexical.npz")
        path_behavioral = os.path.join(cache_dir, f"X_{split_name}_behavioral.npz")
        path_semantic = os.path.join(cache_dir, f"X_{split_name}_semantic.npy")
        path_metadata = os.path.join(cache_dir, f"X_{split_name}_metadata.npy")

        # Check if cache exists
        cache_exists = (
            os.path.exists(path_lexical)
            and os.path.exists(path_behavioral)
            and os.path.exists(path_semantic)
            and os.path.exists(path_metadata)
        )

        if load_cached_data and cache_exists:
            print(f"Loading {split_name} features from cache...")
            with timer(f"Load {split_name} cache"):
                X_lexical = scipy.sparse.load_npz(path_lexical)
                X_behavioral = scipy.sparse.load_npz(path_behavioral)
                X_semantic = np.load(path_semantic)
                X_metadata = np.load(path_metadata)

            return {
                "lexical": X_lexical,
                "behavioral": X_behavioral,
                "semantic": X_semantic,
                "metadata": X_metadata,
            }

        # Process from scratch
        print(f"Generating {split_name} features from scratch...")

        # 1. Process Metadata (Dense Base)
        # This vector is appended to all other views
        with timer("Metadata Processing"):
            meta_df = data["metadata"]
            meta_imputed = self.imputer.transform(meta_df)
            X_metadata = self.scaler.transform(meta_imputed).astype(np.float32)

        # 2. Process Lexical (Sparse Text + Metadata)
        with timer("Lexical Processing"):
            text_series = data["text"]
            X_text_tfidf = self.lexical_vectorizer.transform(text_series)
            # Horizontal Stack: Sparse + Dense -> Sparse
            X_lexical = scipy.sparse.hstack([X_text_tfidf, X_metadata]).tocsr()

        # 3. Process Behavioral (Sparse Community + Metadata)
        with timer("Behavioral Processing"):
            community_processed = self._process_community_data(data["community"])
            X_community_tfidf = self.community_vectorizer.transform(community_processed)
            # Horizontal Stack: Sparse + Dense -> Sparse
            X_behavioral = scipy.sparse.hstack([X_community_tfidf, X_metadata]).tocsr()

        # 4. Process Semantic (Dense Embeddings + Metadata)
        with timer("Semantic Processing"):
            # SentenceTransformer encodes to numpy array
            embeddings = self.embedding_model.encode(
                data["text"].tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Horizontal Stack: Dense + Dense -> Dense
            X_semantic = np.hstack([embeddings, X_metadata]).astype(np.float32)

        # Save to Cache
        print(f"Saving {split_name} features to cache...")
        scipy.sparse.save_npz(path_lexical, X_lexical)
        scipy.sparse.save_npz(path_behavioral, X_behavioral)
        np.save(path_semantic, X_semantic)
        np.save(path_metadata, X_metadata)

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "metadata": X_metadata,
        }
