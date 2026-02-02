import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import library.config as config


class FeatureProcessor:
    """
    Handles data loading, cleaning, and feature generation for the
    Clean-Signal Hex-View Stacking Ensemble.
    Generates Lexical, Behavioral, Semantic, and Contextual (Metadata) views.
    """

    def __init__(self):
        self.cache_dir = config.WORKING_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Checks cache, loads data, processes features, and returns a dictionary of views.

        Args:
            load_cached_data (bool): If True, attempts to load features from disk.

        Returns:
            dict: Nested dictionary containing X matrices and y arrays for train/val/test.
        """
        # Define expected cache files for each split
        # Lexical/Behavioral are sparse (.npz), Semantic/Meta/Y are dense (.npy)
        files = {
            "train": ["X_lexical", "X_behavioral", "X_semantic", "X_meta", "y"],
            "val": ["X_lexical", "X_behavioral", "X_semantic", "X_meta", "y"],
            "test": ["X_lexical", "X_behavioral", "X_semantic", "X_meta"],
        }

        # Attempt to load from cache
        if load_cached_data and self._check_cache(files):
            print(f"Loading features from cache at {self.cache_dir}...")
            return self._load_cache(files)

        print("Cache not found or invalid. Computing features from scratch...")

        # 1. Load Raw Metadata
        print("Loading metadata parquet files...")
        train_df = pd.read_parquet(config.TRAIN_METADATA_PATH)
        val_df = pd.read_parquet(config.VAL_METADATA_PATH)
        test_df = pd.read_parquet(config.TEST_METADATA_PATH)

        # Extract Targets
        y_train = train_df[config.TARGET_COL].values.astype(int)
        y_val = val_df[config.TARGET_COL].values.astype(int)

        # 2. Text Processing (Lexical & Semantic Views)
        print("Processing Text Features (Lexical & Semantic)...")
        train_text = self._concat_text(train_df)
        val_text = self._concat_text(val_df)
        test_text = self._concat_text(test_df)

        # A. Lexical View (TF-IDF)
        tfidf = TfidfVectorizer(**config.TFIDF_PARAMS)
        X_train_lexical = tfidf.fit_transform(train_text)
        X_val_lexical = tfidf.transform(val_text)
        X_test_lexical = tfidf.transform(test_text)

        # B. Semantic View (Embeddings)
        # Using sentence-transformers for dense vector representation
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        # Encode in batches (handled internally by encode)
        X_train_semantic = model.encode(
            train_text, show_progress_bar=False, convert_to_numpy=True
        )
        X_val_semantic = model.encode(
            val_text, show_progress_bar=False, convert_to_numpy=True
        )
        X_test_semantic = model.encode(
            test_text, show_progress_bar=False, convert_to_numpy=True
        )

        # 3. Behavioral Processing (Community View)
        print("Processing Behavioral Features (Subreddit History)...")
        train_comm = self._process_subreddits(train_df)
        val_comm = self._process_subreddits(val_df)
        test_comm = self._process_subreddits(test_df)

        # Bag-of-Concepts using CountVectorizer on subreddit names
        cv = CountVectorizer(**config.COMMUNITY_VEC_PARAMS)
        X_train_behavioral = cv.fit_transform(train_comm)
        X_val_behavioral = cv.transform(val_comm)
        X_test_behavioral = cv.transform(test_comm)

        # 4. Metadata Processing (Contextual View)
        print("Processing Contextual Metadata...")
        meta_cols = config.ALLOW_LIST_META

        # Extract raw values
        X_train_meta_raw = train_df[meta_cols].values
        X_val_meta_raw = val_df[meta_cols].values
        X_test_meta_raw = test_df[meta_cols].values

        # Impute missing values (Median)
        imputer = SimpleImputer(strategy="median")
        X_train_meta_imp = imputer.fit_transform(X_train_meta_raw)
        X_val_meta_imp = imputer.transform(X_val_meta_raw)
        X_test_meta_imp = imputer.transform(X_test_meta_raw)

        # Scale features (StandardScaler)
        scaler = StandardScaler()
        X_train_meta = scaler.fit_transform(X_train_meta_imp)
        X_val_meta = scaler.transform(X_val_meta_imp)
        X_test_meta = scaler.transform(X_test_meta_imp)

        # 5. Assemble Data Dictionary
        data = {
            "train": {
                "X_lexical": X_train_lexical,
                "X_behavioral": X_train_behavioral,
                "X_semantic": X_train_semantic,
                "X_meta": X_train_meta,
                "y": y_train,
            },
            "val": {
                "X_lexical": X_val_lexical,
                "X_behavioral": X_val_behavioral,
                "X_semantic": X_val_semantic,
                "X_meta": X_val_meta,
                "y": y_val,
            },
            "test": {
                "X_lexical": X_test_lexical,
                "X_behavioral": X_test_behavioral,
                "X_semantic": X_test_semantic,
                "X_meta": X_test_meta,
            },
        }

        # 6. Save to Cache
        print("Saving processed features to cache...")
        self._save_cache(data)

        return data

    def _concat_text(self, df):
        """
        Concatenates request title and edit-aware body text.
        """
        # Ensure string type and handle potential NaNs
        t = df[config.TEXT_COLS[0]].fillna("").astype(str)
        b = df[config.TEXT_COLS[1]].fillna("").astype(str)
        return (t + " " + b).tolist()

    def _process_subreddits(self, df):
        """
        Converts list of subreddits into a space-separated string for CountVectorizer.
        """

        def join_subs(x):
            if isinstance(x, (list, np.ndarray)):
                return " ".join([str(s) for s in x])
            return ""

        return df["requester_subreddits_at_request"].apply(join_subs).tolist()

    def _check_cache(self, files_dict):
        """
        Verifies if all required cache files exist.
        """
        for split, keys in files_dict.items():
            for key in keys:
                # Determine extension based on feature type
                is_sparse = "lexical" in key or "behavioral" in key
                ext = ".npz" if is_sparse else ".npy"
                path = os.path.join(self.cache_dir, f"{split}_{key}{ext}")
                if not os.path.exists(path):
                    return False
        return True

    def _save_cache(self, data):
        """
        Saves data dictionary to disk using numpy/scipy formats.
        """
        for split, items in data.items():
            for key, val in items.items():
                ext = ".npz" if sp.issparse(val) else ".npy"
                path = os.path.join(self.cache_dir, f"{split}_{key}{ext}")
                if sp.issparse(val):
                    sp.save_npz(path, val)
                else:
                    np.save(path, val)

    def _load_cache(self, files_dict):
        """
        Loads data dictionary from disk.
        """
        data = {}
        for split, keys in files_dict.items():
            data[split] = {}
            for key in keys:
                is_sparse = "lexical" in key or "behavioral" in key
                ext = ".npz" if is_sparse else ".npy"
                path = os.path.join(self.cache_dir, f"{split}_{key}{ext}")
                if ext == ".npz":
                    data[split][key] = sp.load_npz(path)
                else:
                    data[split][key] = np.load(path)
        return data
