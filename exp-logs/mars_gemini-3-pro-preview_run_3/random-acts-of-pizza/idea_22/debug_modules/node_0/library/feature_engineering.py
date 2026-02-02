import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import (
    set_seed,
    clean_text,
    load_data,
    save_to_cache,
    load_from_cache,
    get_device,
)


class FeaturePipeline:
    """
    Orchestrates the generation of multi-modal features for the Hex-View Ensemble.
    Handles Metadata, Lexical (TF-IDF), Behavioral (History TF-IDF),
    Semantic (SBERT), and Manifold (PCA) views.
    """

    def __init__(self, load_cached_data=True):
        self.load_cached_data = load_cached_data
        self.device = get_device()
        set_seed(Config.SEED)

    def _get_cache_filenames(self):
        """Returns a dictionary of filenames for all feature views and splits."""
        splits = ["train", "val", "test"]
        views = ["metadata", "lexical", "behavioral", "semantic", "manifold"]
        filenames = {}

        for split in splits:
            # Target variable (only for train/val)
            if split != "test":
                filenames[f"{split}_y"] = f"y_{split}.npy"

            # Feature views
            filenames[f"{split}_metadata"] = f"X_{split}_metadata.npy"
            filenames[f"{split}_lexical"] = f"X_{split}_lexical.npz"
            filenames[f"{split}_behavioral"] = f"X_{split}_behavioral.npz"
            filenames[f"{split}_semantic"] = f"X_{split}_semantic.npy"
            filenames[f"{split}_manifold"] = f"X_{split}_manifold.npy"

        return filenames

    def _check_cache_exists(self, filenames):
        """Checks if all required cache files exist."""
        for key, filename in filenames.items():
            filepath = os.path.join(Config.WORKING_DIR, filename)
            if not os.path.exists(filepath):
                return False
        return True

    def _load_from_cache(self, filenames):
        """Loads all data from cache."""
        data = {}
        for key, filename in filenames.items():
            data[key] = load_from_cache(filename)

        # Structure into nested dictionary
        result = {"train": {}, "val": {}, "test": {}}
        for key, value in data.items():
            split, view = key.split("_", 1)
            result[split][view] = value

        return result

    def _process_metadata(self, df_train, df_val, df_test):
        """
        Processes numerical metadata: selection, temporal extraction, imputation, scaling.
        """
        # Identify candidate numerical columns
        all_cols = df_train.columns
        exclude_cols = [
            Config.ID_COL,
            Config.TARGET_COL,
            Config.TEXT_COL,
            Config.TITLE_COL,
            "requester_username",
            "source_file",
            "request_text",
            "request_id",
        ]

        # Filter columns based on exclusion patterns (leakage prevention)
        # and explicit exclusions
        meta_cols = []
        for col in all_cols:
            if col in exclude_cols:
                continue
            if df_train[col].dtype == "object":
                continue
            # Check for leakage patterns like "_at_retrieval"
            if any(pattern in col for pattern in Config.EXCLUDE_PATTERNS):
                continue
            meta_cols.append(col)

        # Helper to extract and enhance features
        def extract_features(df):
            # Select base numerical cols
            data = df[meta_cols].copy()

            # Temporal features
            if "unix_timestamp_of_request" in df.columns:
                # Convert to datetime
                dt = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
                data["hour_of_request"] = dt.dt.hour
                data["day_of_week"] = dt.dt.dayofweek

            return data

        X_train_meta = extract_features(df_train)
        X_val_meta = extract_features(df_val)
        X_test_meta = extract_features(df_test)

        # Imputation
        imputer = SimpleImputer(strategy="median")
        X_train_meta = imputer.fit_transform(X_train_meta)
        X_val_meta = imputer.transform(X_val_meta)
        X_test_meta = imputer.transform(X_test_meta)

        # Scaling
        scaler = StandardScaler()
        X_train_meta = scaler.fit_transform(X_train_meta)
        X_val_meta = scaler.transform(X_val_meta)
        X_test_meta = scaler.transform(X_test_meta)

        return X_train_meta, X_val_meta, X_test_meta

    def _process_lexical(self, df_train, df_val, df_test):
        """
        Processes text data using TF-IDF.
        """
        # Clean text
        train_text = clean_text(df_train[Config.TEXT_COL])
        val_text = clean_text(df_val[Config.TEXT_COL])
        test_text = clean_text(df_test[Config.TEXT_COL])

        # TF-IDF
        vectorizer = TfidfVectorizer(**Config.TFIDF_TEXT_PARAMS)
        X_train_lex = vectorizer.fit_transform(train_text)
        X_val_lex = vectorizer.transform(val_text)
        X_test_lex = vectorizer.transform(test_text)

        return X_train_lex, X_val_lex, X_test_lex

    def _process_behavioral(self, df_train, df_val, df_test):
        """
        Processes subreddit history using TF-IDF (Bag of Concepts).
        """

        def join_subreddits(series):
            # Handle cases where it might be a list or already a string or NaN
            return series.apply(
                lambda x: (
                    " ".join(x)
                    if isinstance(x, list)
                    else str(x) if pd.notnull(x) else ""
                )
            )

        train_subs = join_subreddits(df_train[Config.SUBREDDIT_COL])
        val_subs = join_subreddits(df_val[Config.SUBREDDIT_COL])
        test_subs = join_subreddits(df_test[Config.SUBREDDIT_COL])

        vectorizer = TfidfVectorizer(**Config.TFIDF_HISTORY_PARAMS)
        X_train_beh = vectorizer.fit_transform(train_subs)
        X_val_beh = vectorizer.transform(val_subs)
        X_test_beh = vectorizer.transform(test_subs)

        return X_train_beh, X_val_beh, X_test_beh

    def _process_semantic(self, df_train, df_val, df_test):
        """
        Generates dense embeddings using SBERT.
        """
        model = SentenceTransformer(Config.SBERT_MODEL_NAME, device=self.device)

        train_text = clean_text(df_train[Config.TEXT_COL]).tolist()
        val_text = clean_text(df_val[Config.TEXT_COL]).tolist()
        test_text = clean_text(df_test[Config.TEXT_COL]).tolist()

        # Encode
        X_train_sem = model.encode(
            train_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_val_sem = model.encode(
            val_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        X_test_sem = model.encode(
            test_text, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        return X_train_sem, X_val_sem, X_test_sem

    def _process_manifold(self, X_train_sem, X_val_sem, X_test_sem):
        """
        Applies PCA to semantic embeddings for the Manifold view.
        """
        pca = PCA(n_components=Config.PCA_N_COMPONENTS, random_state=Config.SEED)
        X_train_man = pca.fit_transform(X_train_sem)
        X_val_man = pca.transform(X_val_sem)
        X_test_man = pca.transform(X_test_sem)

        # Scale PCA components (important for kNN)
        scaler = StandardScaler()
        X_train_man = scaler.fit_transform(X_train_man)
        X_val_man = scaler.transform(X_val_man)
        X_test_man = scaler.transform(X_test_man)

        return X_train_man, X_val_man, X_test_man

    def run(self):
        """
        Main execution method. Checks cache, computes features if needed, saves, and returns.
        """
        filenames = self._get_cache_filenames()

        # 1. Check Cache
        if self.load_cached_data and self._check_cache_exists(filenames):
            return self._load_from_cache(filenames)

        # 2. Load Raw Data
        df_train = load_data("train")
        df_val = load_data("val")
        df_test = load_data("test")

        # Extract Targets
        y_train = df_train[Config.TARGET_COL].values.astype(int)
        y_val = df_val[Config.TARGET_COL].values.astype(int)

        # 3. Compute Features

        # A. Contextual (Metadata)
        X_train_meta, X_val_meta, X_test_meta = self._process_metadata(
            df_train, df_val, df_test
        )

        # B. Lexical (Text TF-IDF)
        X_train_lex, X_val_lex, X_test_lex = self._process_lexical(
            df_train, df_val, df_test
        )

        # C. Behavioral (Subreddit TF-IDF)
        X_train_beh, X_val_beh, X_test_beh = self._process_behavioral(
            df_train, df_val, df_test
        )

        # D. Semantic (SBERT Embeddings)
        X_train_sem, X_val_sem, X_test_sem = self._process_semantic(
            df_train, df_val, df_test
        )

        # E. Manifold (PCA on Semantic)
        X_train_man, X_val_man, X_test_man = self._process_manifold(
            X_train_sem, X_val_sem, X_test_sem
        )

        # 4. Save to Cache
        data_map = {
            # Train
            filenames["train_y"]: y_train,
            filenames["train_metadata"]: X_train_meta,
            filenames["train_lexical"]: X_train_lex,
            filenames["train_behavioral"]: X_train_beh,
            filenames["train_semantic"]: X_train_sem,
            filenames["train_manifold"]: X_train_man,
            # Val
            filenames["val_y"]: y_val,
            filenames["val_metadata"]: X_val_meta,
            filenames["val_lexical"]: X_val_lex,
            filenames["val_behavioral"]: X_val_beh,
            filenames["val_semantic"]: X_val_sem,
            filenames["val_manifold"]: X_val_man,
            # Test
            filenames["test_metadata"]: X_test_meta,
            filenames["test_lexical"]: X_test_lex,
            filenames["test_behavioral"]: X_test_beh,
            filenames["test_semantic"]: X_test_sem,
            filenames["test_manifold"]: X_test_man,
        }

        for filename, data in data_map.items():
            save_to_cache(data, filename)

        # 5. Return Structured Data
        return {
            "train": {
                "y": y_train,
                "metadata": X_train_meta,
                "lexical": X_train_lex,
                "behavioral": X_train_beh,
                "semantic": X_train_sem,
                "manifold": X_train_man,
            },
            "val": {
                "y": y_val,
                "metadata": X_val_meta,
                "lexical": X_val_lex,
                "behavioral": X_val_beh,
                "semantic": X_val_sem,
                "manifold": X_val_man,
            },
            "test": {
                "metadata": X_test_meta,
                "lexical": X_test_lex,
                "behavioral": X_test_beh,
                "semantic": X_test_sem,
                "manifold": X_test_man,
            },
        }
