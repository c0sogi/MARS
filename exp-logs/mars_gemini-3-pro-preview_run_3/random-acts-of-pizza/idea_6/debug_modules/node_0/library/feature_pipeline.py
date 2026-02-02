import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from library.config import (
    SEED,
    TEXT_COL,
    SUBREDDIT_COL,
    ID_COL,
    TARGET_COL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    SBERT_MODEL_NAME,
    SBERT_BATCH_SIZE,
    SUBREDDIT_TFIDF_MAX_FEATURES,
    SVD_COMPONENTS,
    SVD_RANDOM_STATE,
    CACHE_TRAIN_FEATURES,
    CACHE_VAL_FEATURES,
    CACHE_TEST_FEATURES,
    WORKING_DIR,
)
from library.utils import timer, set_seed


class FeaturePipeline:
    """
    Orchestrates the generation of Lexical, Semantic, and Community feature views.
    """

    def __init__(self):
        self.lexical_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words="english",
        )

        self.community_vectorizer = TfidfVectorizer(
            max_features=SUBREDDIT_TFIDF_MAX_FEATURES, stop_words="english"
        )

        self.community_svd = TruncatedSVD(
            n_components=SVD_COMPONENTS, random_state=SVD_RANDOM_STATE
        )

        # SBERT model is loaded lazily or in transform to save memory if needed,
        # but here we init it to be ready.
        # We use CPU or GPU automatically.
        self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

        self.meta_cols = []

    def _get_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts numerical metadata columns."""
        # Filter out non-numeric and special columns
        # We assume df has already been cleaned/imputed by data_loader
        if not self.meta_cols:
            # Define meta cols based on the first dataframe seen (fit)
            all_cols = df.columns.tolist()
            exclude = {
                TEXT_COL,
                SUBREDDIT_COL,
                ID_COL,
                TARGET_COL,
                "request_title",
                "request_text",
            }
            self.meta_cols = [
                c
                for c in all_cols
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            # Sort for deterministic order
            self.meta_cols.sort()

        return df[self.meta_cols].values.astype(np.float32)

    def _process_subreddits(self, series: pd.Series) -> list:
        """Joins list of subreddits into a single string per user."""
        # Handle cases where it might be a list or already a string (though data_loader keeps it as list usually)
        # or numpy array of lists
        processed = []
        for item in series:
            if isinstance(item, list):
                processed.append(" ".join(item))
            elif isinstance(item, np.ndarray):
                processed.append(" ".join(item))
            else:
                # Fallback for empty or string
                processed.append(str(item) if item else "")
        return processed

    def fit(self, X_train: pd.DataFrame):
        """Fits the transformers on the training data."""
        print("Fitting FeaturePipeline...")

        # 1. Identify Metadata Columns
        _ = self._get_metadata(X_train)
        print(f"Identified {len(self.meta_cols)} metadata columns.")

        # 2. Fit Lexical Vectorizer
        print("Fitting Lexical TF-IDF...")
        text_data = X_train[TEXT_COL].fillna("").astype(str).tolist()
        self.lexical_vectorizer.fit(text_data)

        # 3. Fit Community Vectorizer & SVD
        print("Fitting Community TF-IDF & SVD...")
        subreddit_data = self._process_subreddits(X_train[SUBREDDIT_COL])
        subreddit_tfidf = self.community_vectorizer.fit_transform(subreddit_data)
        self.community_svd.fit(subreddit_tfidf)

        return self

    def transform(self, df: pd.DataFrame, desc: str = "Data"):
        """Transforms the data into three feature views."""
        print(f"Transforming {desc}...")

        # 1. Metadata
        metadata = self._get_metadata(df)

        # 2. Lexical View (Sparse)
        # TF-IDF on text
        text_data = df[TEXT_COL].fillna("").astype(str).tolist()
        lexical_tfidf = self.lexical_vectorizer.transform(text_data)
        # Concatenate: [TF-IDF (Sparse) | Metadata (Dense)] -> Sparse
        # Convert metadata to sparse for efficient hstack
        meta_sparse = sp.csr_matrix(metadata)
        X_lexical = sp.hstack([lexical_tfidf, meta_sparse], format="csr")

        # 3. Semantic View (Dense)
        # SBERT Embeddings
        # Note: encode returns numpy array
        print(f"  Encoding SBERT for {desc}...")
        embeddings = self.sbert_model.encode(
            text_data,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Concatenate: [Embeddings (Dense) | Metadata (Dense)] -> Dense
        X_semantic = np.hstack([embeddings, metadata])

        # 4. Community View (Dense)
        # Subreddit -> TF-IDF -> SVD
        subreddit_data = self._process_subreddits(df[SUBREDDIT_COL])
        comm_tfidf = self.community_vectorizer.transform(subreddit_data)
        comm_svd = self.community_svd.transform(comm_tfidf)
        # Concatenate: [SVD (Dense) | Metadata (Dense)] -> Dense
        X_community = np.hstack([comm_svd, metadata])

        return {"lexical": X_lexical, "semantic": X_semantic, "community": X_community}


def save_features(features_dict, path):
    """Saves the features dictionary to a .npz file."""
    # We split sparse and dense for efficient storage
    save_dict = {}

    # Lexical is sparse CSR
    save_dict["lexical_data"] = features_dict["lexical"].data
    save_dict["lexical_indices"] = features_dict["lexical"].indices
    save_dict["lexical_indptr"] = features_dict["lexical"].indptr
    save_dict["lexical_shape"] = features_dict["lexical"].shape

    # Semantic and Community are dense
    save_dict["semantic"] = features_dict["semantic"]
    save_dict["community"] = features_dict["community"]

    np.savez_compressed(path, **save_dict)
    print(f"Saved features to {path}")


def load_features(path):
    """Loads features from a .npz file."""
    loaded = np.load(path)

    # Reconstruct Lexical Sparse Matrix
    lexical = sp.csr_matrix(
        (loaded["lexical_data"], loaded["lexical_indices"], loaded["lexical_indptr"]),
        shape=loaded["lexical_shape"],
    )

    return {
        "lexical": lexical,
        "semantic": loaded["semantic"],
        "community": loaded["community"],
    }


def generate_features(X_train, X_val, X_test, load_cached_data=True):
    """
    Main function to generate or load features.

    Args:
        X_train, X_val, X_test: DataFrames from data_loader.
        load_cached_data: Boolean to use cache.

    Returns:
        Tuple of dictionaries (train_feats, val_feats, test_feats).
        Each dictionary contains keys: 'lexical', 'semantic', 'community'.
    """
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Check if all caches exist
    cache_exists = (
        os.path.exists(CACHE_TRAIN_FEATURES)
        and os.path.exists(CACHE_VAL_FEATURES)
        and os.path.exists(CACHE_TEST_FEATURES)
    )

    if load_cached_data and cache_exists:
        print("Loading features from cache...")
        with timer("Load Features"):
            train_feats = load_features(CACHE_TRAIN_FEATURES)
            val_feats = load_features(CACHE_VAL_FEATURES)
            test_feats = load_features(CACHE_TEST_FEATURES)
            return train_feats, val_feats, test_feats

    # If not loaded, compute
    print("Generating features from scratch...")

    pipeline = FeaturePipeline()

    with timer("Fit Pipeline"):
        pipeline.fit(X_train)

    with timer("Transform Train"):
        train_feats = pipeline.transform(X_train, "Train")
        save_features(train_feats, CACHE_TRAIN_FEATURES)

    with timer("Transform Val"):
        val_feats = pipeline.transform(X_val, "Val")
        save_features(val_feats, CACHE_VAL_FEATURES)

    with timer("Transform Test"):
        test_feats = pipeline.transform(X_test, "Test")
        save_features(test_feats, CACHE_TEST_FEATURES)

    return train_feats, val_feats, test_feats
