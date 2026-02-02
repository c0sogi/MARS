import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import Timer, set_seed


class PentViewFeatureGenerator:
    """
    Generates four distinct feature views:
    1. Lexical: Sparse TF-IDF of request text.
    2. Behavioral: Sparse TF-IDF of subreddit history (bag-of-communities).
    3. Semantic: Dense embeddings of request text.
    4. Metadata: Scaled numerical features (Contextual).
    """

    def __init__(self):
        self.lexical_vectorizer = None
        self.behavioral_vectorizer = None
        self.metadata_imputer = None
        self.metadata_scaler = None
        self.embedding_model = None

    def _get_subreddit_string(self, df: pd.DataFrame) -> pd.Series:
        """Helper to convert list of subreddits to space-separated string."""
        return df[Config.SUBREDDIT_COL].apply(
            lambda x: " ".join(x) if isinstance(x, list) else ""
        )

    def fit(self, train_df: pd.DataFrame):
        """
        Fits the vectorizers, imputer, and scaler on the training data.
        """
        with Timer("Feature Generator Fit"):
            # 1. Lexical View (Text)
            print("Fitting Lexical Vectorizer...")
            self.lexical_vectorizer = TfidfVectorizer(
                max_features=Config.TFIDF_MAX_FEATURES, **Config.TFIDF_PARAMS
            )
            self.lexical_vectorizer.fit(train_df[Config.TEXT_COL].fillna(""))

            # 2. Behavioral View (Subreddits)
            print("Fitting Behavioral Vectorizer...")
            self.behavioral_vectorizer = TfidfVectorizer(
                max_features=Config.TFIDF_MAX_FEATURES, **Config.TFIDF_PARAMS
            )
            subreddits_str = self._get_subreddit_string(train_df)
            self.behavioral_vectorizer.fit(subreddits_str)

            # 3. Metadata View (Numerical)
            print("Fitting Metadata Imputer and Scaler...")
            self.metadata_imputer = SimpleImputer(strategy="median")
            self.metadata_scaler = StandardScaler()

            meta_data = train_df[Config.NUMERICAL_COLS]
            meta_data_imputed = self.metadata_imputer.fit_transform(meta_data)
            self.metadata_scaler.fit(meta_data_imputed)

            # 4. Semantic View (Embeddings)
            # No fitting required for pre-trained models, but we verify we can load it
            print("Initializing Embedding Model...")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def transform(self, df: pd.DataFrame) -> dict:
        """
        Transforms the dataframe into a dictionary of feature matrices.
        """
        if self.lexical_vectorizer is None:
            raise RuntimeError("Generator has not been fitted yet.")

        features = {}

        # 1. Lexical View
        features["lexical"] = self.lexical_vectorizer.transform(
            df[Config.TEXT_COL].fillna("")
        )

        # 2. Behavioral View
        subreddits_str = self._get_subreddit_string(df)
        features["behavioral"] = self.behavioral_vectorizer.transform(subreddits_str)

        # 3. Metadata View
        meta_data = df[Config.NUMERICAL_COLS]
        meta_data_imputed = self.metadata_imputer.transform(meta_data)
        features["metadata"] = self.metadata_scaler.transform(meta_data_imputed)

        # 4. Semantic View
        # Ensure model is loaded (in case transform is called in a new process/after pickle)
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        # Encode
        sentences = df[Config.TEXT_COL].fillna("").tolist()
        features["semantic"] = self.embedding_model.encode(
            sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        return features


def _save_features_to_cache(features: dict, split_name: str):
    """Saves feature dictionary to cache directory."""
    for view_name, data in features.items():
        filename = f"{split_name}_{view_name}"
        path = os.path.join(Config.CACHE_DIR, filename)

        if sparse.issparse(data):
            sparse.save_npz(path + ".npz", data)
        else:
            np.save(path + ".npy", data)


def _load_features_from_cache(split_name: str) -> dict:
    """Loads feature dictionary from cache directory."""
    features = {}
    views = ["lexical", "behavioral", "metadata", "semantic"]

    for view in views:
        filename = f"{split_name}_{view}"
        path_base = os.path.join(Config.CACHE_DIR, filename)

        if os.path.exists(path_base + ".npz"):
            features[view] = sparse.load_npz(path_base + ".npz")
        elif os.path.exists(path_base + ".npy"):
            features[view] = np.load(path_base + ".npy")
        else:
            raise FileNotFoundError(f"Cache missing for {split_name} view {view}")

    return features


def generate_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    load_cached_data: bool = True,
):
    """
    Orchestrates feature generation with caching.

    Args:
        train_df, val_df, test_df: DataFrames.
        load_cached_data: If True, attempts to load features from disk.

    Returns:
        tuple: (train_features, val_features, test_features)
               Each element is a dictionary of feature matrices.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Check if cache exists
    cache_complete = True
    splits = ["train", "val", "test"]
    views = ["lexical", "behavioral", "metadata", "semantic"]

    for split in splits:
        for view in views:
            ext = ".npz" if view in ["lexical", "behavioral"] else ".npy"
            if not os.path.exists(
                os.path.join(Config.CACHE_DIR, f"{split}_{view}{ext}")
            ):
                cache_complete = False
                break

    if load_cached_data and cache_complete:
        with Timer("Load Features (Cached)"):
            try:
                train_feats = _load_features_from_cache("train")
                val_feats = _load_features_from_cache("val")
                test_feats = _load_features_from_cache("test")
                print(f"Successfully loaded features from {Config.CACHE_DIR}")
                return train_feats, val_feats, test_feats
            except Exception as e:
                print(f"Failed to load cache: {e}. Regenerating...")

    # Regenerate
    with Timer("Generate Features (Fresh)"):
        generator = PentViewFeatureGenerator()

        # Fit on Train
        generator.fit(train_df)

        # Transform all
        print("Transforming Train...")
        train_feats = generator.transform(train_df)

        print("Transforming Val...")
        val_feats = generator.transform(val_df)

        print("Transforming Test...")
        test_feats = generator.transform(test_df)

        # Save to cache
        print("Saving to cache...")
        _save_features_to_cache(train_feats, "train")
        _save_features_to_cache(val_feats, "val")
        _save_features_to_cache(test_feats, "test")

    return train_feats, val_feats, test_feats
