import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library import config
from library import utils


class ViewBuilder:
    """
    Constructs the four data views (Lexical, Behavioral, Semantic, Metadata)
    required by the Pent-View architecture.

    Manages the fitting of transformers on training data and the consistent
    transformation of validation and test data. Implements strict caching
    to optimize runtime.
    """

    def __init__(self):
        # Metadata transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # Sparse Lexical Branch (Text Modality)
        self.lexical_vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)

        # Sparse Behavioral Branch (History Modality)
        self.behavioral_vectorizer = TfidfVectorizer(**config.TFIDF_PARAMS)

        # Dense Semantic Branch (Text Modality)
        # Lazy initialization to save resources if not used immediately
        self.embedding_model = None

    def fit(self, df):
        """
        Fits the internal transformers on the training data.

        Args:
            df (pd.DataFrame): The training dataframe.

        Returns:
            self: Returns the instance itself.
        """
        utils.print_info("Fitting ViewBuilder transformers...")

        # 1. Fit Metadata Transformers
        # Select only allow-listed features
        meta_data = df[config.METADATA_FEATURES]
        self.imputer.fit(meta_data)
        # Transform temporarily to fit the scaler
        meta_imputed = self.imputer.transform(meta_data)
        self.scaler.fit(meta_imputed)

        # 2. Fit Lexical Vectorizer
        texts = df[config.TEXT_COL].fillna("").astype(str)
        self.lexical_vectorizer.fit(texts)

        # 3. Fit Behavioral Vectorizer
        # Convert list of subreddits to space-separated string for TF-IDF
        subreddits = df[config.SUBREDDIT_LIST_COL].apply(
            lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
        )
        self.behavioral_vectorizer.fit(subreddits)

        # Note: Semantic model is pretrained (all-MiniLM-L6-v2), so no fitting is required.

        utils.print_info("ViewBuilder fitting complete.")
        return self

    def transform(self, df, split_name, load_cached=True):
        """
        Transforms the dataframe into the 4 views required by the ensemble.
        Uses caching to avoid re-computation of expensive features.

        Args:
            df (pd.DataFrame): The dataframe to transform.
            split_name (str): The name of the split (e.g., 'train', 'val', 'test') for cache naming.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            dict: A dictionary containing 'metadata', 'lexical', 'behavioral', and 'semantic' views.
        """
        utils.print_info(f"Generating views for split: {split_name}")

        # --- 1. Metadata View ---
        def compute_metadata():
            meta_data = df[config.METADATA_FEATURES]
            meta_imputed = self.imputer.transform(meta_data)
            meta_scaled = self.scaler.transform(meta_imputed)
            return meta_scaled.astype(np.float32)

        meta_path = os.path.join(config.CACHE_DIR, f"{split_name}_metadata.npy")
        X_metadata = utils.get_cached_data(
            meta_path, compute_metadata, load_cached=load_cached
        )

        # --- 2. Lexical View (Sparse) ---
        def compute_lexical():
            texts = df[config.TEXT_COL].fillna("").astype(str)
            tfidf = self.lexical_vectorizer.transform(texts)
            # Concatenate with Metadata (Sparse + Dense -> Sparse)
            # We cast metadata to sparse csr for efficient hstack
            X_combined = sparse.hstack([tfidf, sparse.csr_matrix(X_metadata)])
            return X_combined

        lex_path = os.path.join(config.CACHE_DIR, f"{split_name}_lexical.npz")
        X_lexical_res = utils.get_cached_data(
            lex_path, compute_lexical, load_cached=load_cached
        )

        # Handle npz return type if loaded from cache (extract sparse matrix from NpzFile)
        if isinstance(X_lexical_res, np.lib.npyio.NpzFile):
            X_lexical = X_lexical_res["arr_0"].item()
        else:
            X_lexical = X_lexical_res

        # --- 3. Behavioral View (Sparse) ---
        def compute_behavioral():
            subreddits = df[config.SUBREDDIT_LIST_COL].apply(
                lambda x: " ".join(x) if isinstance(x, (list, np.ndarray)) else ""
            )
            tfidf = self.behavioral_vectorizer.transform(subreddits)
            # Concatenate with Metadata
            X_combined = sparse.hstack([tfidf, sparse.csr_matrix(X_metadata)])
            return X_combined

        beh_path = os.path.join(config.CACHE_DIR, f"{split_name}_behavioral.npz")
        X_behavioral_res = utils.get_cached_data(
            beh_path, compute_behavioral, load_cached=load_cached
        )

        if isinstance(X_behavioral_res, np.lib.npyio.NpzFile):
            X_behavioral = X_behavioral_res["arr_0"].item()
        else:
            X_behavioral = X_behavioral_res

        # --- 4. Semantic View (Dense) ---
        def compute_semantic():
            if self.embedding_model is None:
                utils.print_info(
                    f"Loading SentenceTransformer: {config.TRANSFORMER_MODEL}"
                )
                self.embedding_model = SentenceTransformer(config.TRANSFORMER_MODEL)

            texts = df[config.TEXT_COL].fillna("").astype(str).tolist()
            embeddings = self.embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False
            )

            # Concatenate with Metadata (Dense + Dense -> Dense)
            X_combined = np.hstack([embeddings, X_metadata])
            return X_combined.astype(np.float32)

        sem_path = os.path.join(config.CACHE_DIR, f"{split_name}_semantic.npy")
        X_semantic = utils.get_cached_data(
            sem_path, compute_semantic, load_cached=load_cached
        )

        return {
            "metadata": X_metadata,
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
        }
