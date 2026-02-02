import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import Timer


class FeaturePipeline:
    """
    Manages the creation of the Pent-View feature sets:
    1. Sparse Lexical (Text TFIDF + Metadata)
    2. Sparse Behavioral (Subreddit TFIDF + Metadata)
    3. Dense Semantic (Text Embeddings + Metadata)
    4. Contextual (Metadata Only)
    """

    def __init__(self):
        # 1. Lexical Vectorizer (Text)
        self.tfidf_text = TfidfVectorizer(**Config.TFIDF_PARAMS)

        # 2. Behavioral Vectorizer (Subreddits)
        # Enforce vocabulary constraint for behavioral data
        self.tfidf_behavioral = TfidfVectorizer(
            strip_accents="unicode",
            stop_words="english",
            min_df=2,
            max_features=Config.MAX_SUBREDDIT_VOCAB,
            token_pattern=r"(?u)\b\w+\b",
        )

        # 3. Metadata Preprocessors
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # 4. Semantic Model (Lazy loaded)
        self.embedding_model = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        return self.embedding_model

    def _preprocess_text(self, df: pd.DataFrame) -> pd.Series:
        """Concatenates title and body for text processing."""
        title = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        body = df[Config.TEXT_COLS[1]].fillna("").astype(str)
        return title + " " + body

    def _preprocess_subreddits(self, df: pd.DataFrame) -> pd.Series:
        """Converts list of subreddits to space-separated string."""

        def join_subs(x):
            if isinstance(x, list):
                return " ".join(x)
            return str(x) if x is not None else ""

        return df[Config.SUBREDDIT_COL].apply(join_subs)

    def fit(self, df: pd.DataFrame):
        """Fits vectorizers, imputer, and scaler on training data."""
        with Timer("FeaturePipeline Fit"):
            # Text
            text_data = self._preprocess_text(df)
            self.tfidf_text.fit(text_data)

            # Behavioral
            sub_data = self._preprocess_subreddits(df)
            self.tfidf_behavioral.fit(sub_data)

            # Metadata
            meta_data = df[Config.DENSE_FEATURES].copy()
            self.imputer.fit(meta_data)
            # We transform to fit the scaler on imputed data
            meta_imputed = self.imputer.transform(meta_data)
            self.scaler.fit(meta_imputed)

        return self

    def transform(self, df: pd.DataFrame) -> dict:
        """Transforms data into the four feature views."""
        with Timer("FeaturePipeline Transform"):
            # 1. Prepare Base Components
            # Text Sparse
            text_data = self._preprocess_text(df)
            X_text_sparse = self.tfidf_text.transform(text_data)

            # Behavioral Sparse
            sub_data = self._preprocess_subreddits(df)
            X_sub_sparse = self.tfidf_behavioral.transform(sub_data)

            # Metadata Dense
            meta_data = df[Config.DENSE_FEATURES].copy()
            meta_imputed = self.imputer.transform(meta_data)
            X_meta_dense = self.scaler.transform(meta_imputed)

            # Semantic Dense (Embeddings)
            model = self._get_embedding_model()
            # encode expects a list of strings
            X_embeddings = model.encode(
                text_data.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # 2. Construct Views (Concatenating Metadata)

            # View 1: Lexical (Sparse Text + Dense Metadata)
            # We cast metadata to sparse to stack efficiently
            X_meta_sparse = sp.csr_matrix(X_meta_dense)
            X_lexical = sp.hstack([X_text_sparse, X_meta_sparse], format="csr")

            # View 2: Behavioral (Sparse Subreddits + Dense Metadata)
            X_behavioral = sp.hstack([X_sub_sparse, X_meta_sparse], format="csr")

            # View 3: Semantic (Dense Embeddings + Dense Metadata)
            X_semantic = np.hstack([X_embeddings, X_meta_dense])

            # View 4: Contextual (Metadata Only)
            X_metadata = X_meta_dense

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
            "metadata": X_metadata,
        }


def create_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Main entry point. Checks cache, processes data, saves cache, returns features.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define filenames
    splits = ["train", "val", "test"]
    views = ["lexical", "behavioral", "semantic", "metadata"]

    # Helper to generate paths
    def get_paths(split):
        return {
            "lexical": os.path.join(Config.WORKING_DIR, f"X_{split}_lexical.npz"),
            "behavioral": os.path.join(Config.WORKING_DIR, f"X_{split}_behavioral.npz"),
            "semantic": os.path.join(Config.WORKING_DIR, f"X_{split}_semantic.npy"),
            "metadata": os.path.join(Config.WORKING_DIR, f"X_{split}_metadata.npy"),
        }

    # Check if all files exist
    all_files_exist = True
    for split in splits:
        paths = get_paths(split)
        for p in paths.values():
            if not os.path.exists(p):
                all_files_exist = False
                break

    # Load from cache if requested and available
    if load_cached_data and all_files_exist:
        print("Loading features from cache...")
        results = {}
        for split in splits:
            paths = get_paths(split)
            split_data = {}
            split_data["lexical"] = sp.load_npz(paths["lexical"])
            split_data["behavioral"] = sp.load_npz(paths["behavioral"])
            split_data["semantic"] = np.load(paths["semantic"])
            split_data["metadata"] = np.load(paths["metadata"])
            results[split] = split_data

        return results["train"], results["val"], results["test"]

    # Process from scratch
    print("Generating features from scratch...")

    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    # Save pipeline for potential future inference use
    joblib.dump(pipeline, os.path.join(Config.WORKING_DIR, "fe_pipeline.joblib"))

    # Transform and Save
    results = {}
    datasets = zip(splits, [train_df, val_df, test_df])

    for split_name, df in datasets:
        print(f"Transforming {split_name} set...")
        feats = pipeline.transform(df)
        results[split_name] = feats

        paths = get_paths(split_name)

        # Save Sparse
        sp.save_npz(paths["lexical"], feats["lexical"])
        sp.save_npz(paths["behavioral"], feats["behavioral"])

        # Save Dense
        np.save(paths["semantic"], feats["semantic"])
        np.save(paths["metadata"], feats["metadata"])

    return results["train"], results["val"], results["test"]
