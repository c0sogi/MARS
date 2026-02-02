import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from datetime import datetime
from library.config import Config
from library.utils import set_seed


class FeaturePipeline:
    """
    Manages the generation of three distinct feature views:
    1. Lexical (Sparse): TF-IDF of text + Metadata
    2. Behavioral (Sparse): TF-IDF of subreddits + Metadata
    3. Semantic (Dense): SBERT Embeddings + Scaled Metadata
    """

    def __init__(self):
        self.lexical_vectorizer = TfidfVectorizer(
            ngram_range=Config.LEXICAL_NGRAM_RANGE,
            max_features=Config.LEXICAL_MAX_FEATURES,
            stop_words="english",
            sublinear_tf=True,
        )

        self.behavioral_vectorizer = TfidfVectorizer(
            ngram_range=Config.BEHAVIORAL_NGRAM_RANGE,
            max_features=Config.BEHAVIORAL_MAX_FEATURES,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",  # Simple token pattern for subreddit names
        )

        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # SBERT model loaded on demand to save memory if not fitting
        self.sbert_model = None

    def _load_sbert(self):
        if self.sbert_model is None:
            # Load frozen SBERT model
            self.sbert_model = SentenceTransformer(Config.SBERT_MODEL_NAME)
            if hasattr(self.sbert_model, "eval"):
                self.sbert_model.eval()

    def _extract_metadata(self, df):
        """
        Extracts numerical metadata and generates derived features.
        """
        # 1. Base Numerical Features
        cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_comments_in_raop_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_posts_on_raop_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        # Ensure columns exist, fill with 0 if missing (safety check)
        meta_df = df[cols].copy()
        for c in cols:
            if c not in meta_df.columns:
                meta_df[c] = 0

        # 2. Derived Text Features
        # Handle potential NaNs in text columns before length calculation
        text_series = df["request_text_edit_aware"].fillna("").astype(str)
        title_series = df["request_title"].fillna("").astype(str)

        meta_df["text_len_char"] = text_series.apply(len)
        meta_df["text_len_word"] = text_series.apply(lambda x: len(x.split()))
        meta_df["title_len_char"] = title_series.apply(len)
        meta_df["title_len_word"] = title_series.apply(lambda x: len(x.split()))

        # 3. Derived Temporal Features
        if "unix_timestamp_of_request" in df.columns:
            # Convert timestamp to datetime objects
            dt_series = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
            meta_df["hour_of_request"] = dt_series.dt.hour
            meta_df["day_of_week"] = dt_series.dt.dayofweek
        else:
            meta_df["hour_of_request"] = 0
            meta_df["day_of_week"] = 0

        return meta_df.values

    def _prepare_text(self, df):
        """Combines title and text for lexical analysis."""
        title = df["request_title"].fillna("").astype(str)
        text = df["request_text_edit_aware"].fillna("").astype(str)
        return (title + " " + text).tolist()

    def _prepare_subreddits(self, df):
        """Flattens list of subreddits into a space-separated string."""

        def join_subs(subs):
            if isinstance(subs, list):
                return " ".join([str(s) for s in subs])
            return ""

        if "requester_subreddits_at_request" in df.columns:
            return df["requester_subreddits_at_request"].apply(join_subs).tolist()
        return [""] * len(df)

    def fit(self, df):
        """Fits transformers on the training data."""
        print("Fitting Feature Pipeline...")

        # 1. Fit Metadata Transformers
        raw_meta = self._extract_metadata(df)
        self.imputer.fit(raw_meta)
        # We fit scaler on the imputed data
        imputed_meta = self.imputer.transform(raw_meta)
        self.scaler.fit(imputed_meta)

        # 2. Fit Lexical Vectorizer
        text_data = self._prepare_text(df)
        self.lexical_vectorizer.fit(text_data)

        # 3. Fit Behavioral Vectorizer
        sub_data = self._prepare_subreddits(df)
        self.behavioral_vectorizer.fit(sub_data)

        # SBERT is pre-trained, no fitting required
        return self

    def transform(self, df):
        """
        Transforms data into three views.
        Returns a dictionary containing the views.
        """
        # 1. Process Metadata
        raw_meta = self._extract_metadata(df)
        imputed_meta = self.imputer.transform(raw_meta)  # For Sparse Views
        scaled_meta = self.scaler.transform(imputed_meta)  # For Dense View

        # 2. Lexical View (Sparse)
        text_data = self._prepare_text(df)
        X_lexical_tfidf = self.lexical_vectorizer.transform(text_data)
        # Concatenate TF-IDF with Unscaled (Imputed) Metadata
        X_lexical = sp.hstack([X_lexical_tfidf, sp.csr_matrix(imputed_meta)])

        # 3. Behavioral View (Sparse)
        sub_data = self._prepare_subreddits(df)
        X_behavioral_tfidf = self.behavioral_vectorizer.transform(sub_data)
        # Concatenate TF-IDF with Unscaled (Imputed) Metadata
        X_behavioral = sp.hstack([X_behavioral_tfidf, sp.csr_matrix(imputed_meta)])

        # 4. Semantic View (Dense)
        self._load_sbert()
        # Encode text to dense embeddings
        # Note: SBERT handles batching internally
        X_embeddings = self.sbert_model.encode(
            text_data, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        # Concatenate Embeddings with Scaled Metadata
        X_semantic = np.hstack([X_embeddings, scaled_meta])

        return {
            "lexical": X_lexical.tocsr(),
            "behavioral": X_behavioral.tocsr(),
            "semantic": X_semantic.astype(np.float32),
        }


def get_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Orchestrates feature generation with caching.
    """
    set_seed()

    # Define filenames
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "X_train_lexical": "X_train_lexical.npz",
        "X_train_behavioral": "X_train_behavioral.npz",
        "X_train_semantic": "X_train_semantic.npy",
        "y_train": "y_train.npy",
        "X_val_lexical": "X_val_lexical.npz",
        "X_val_behavioral": "X_val_behavioral.npz",
        "X_val_semantic": "X_val_semantic.npy",
        "y_val": "y_val.npy",
        "X_test_lexical": "X_test_lexical.npz",
        "X_test_behavioral": "X_test_behavioral.npz",
        "X_test_semantic": "X_test_semantic.npy",
        "test_ids": "test_ids.npy",
    }

    # Check if all files exist
    all_exist = all(os.path.exists(os.path.join(cache_dir, f)) for f in files.values())

    if load_cached_data and all_exist:
        print("Loading features from cache...")
        data = {}
        try:
            # Load Sparse
            data["X_train_lexical"] = sp.load_npz(
                os.path.join(cache_dir, files["X_train_lexical"])
            )
            data["X_train_behavioral"] = sp.load_npz(
                os.path.join(cache_dir, files["X_train_behavioral"])
            )
            data["X_val_lexical"] = sp.load_npz(
                os.path.join(cache_dir, files["X_val_lexical"])
            )
            data["X_val_behavioral"] = sp.load_npz(
                os.path.join(cache_dir, files["X_val_behavioral"])
            )
            data["X_test_lexical"] = sp.load_npz(
                os.path.join(cache_dir, files["X_test_lexical"])
            )
            data["X_test_behavioral"] = sp.load_npz(
                os.path.join(cache_dir, files["X_test_behavioral"])
            )

            # Load Dense
            data["X_train_semantic"] = np.load(
                os.path.join(cache_dir, files["X_train_semantic"])
            )
            data["y_train"] = np.load(os.path.join(cache_dir, files["y_train"]))
            data["X_val_semantic"] = np.load(
                os.path.join(cache_dir, files["X_val_semantic"])
            )
            data["y_val"] = np.load(os.path.join(cache_dir, files["y_val"]))
            data["X_test_semantic"] = np.load(
                os.path.join(cache_dir, files["X_test_semantic"])
            )
            data["test_ids"] = np.load(
                os.path.join(cache_dir, files["test_ids"]), allow_pickle=True
            )

            return data
        except Exception as e:
            print(f"Cache load failed ({e}). Re-generating features...")

    print("Generating features from scratch...")

    # Initialize and Fit Pipeline
    pipeline = FeaturePipeline()
    pipeline.fit(train_df)

    # Transform
    print("Transforming Train set...")
    train_feats = pipeline.transform(train_df)
    print("Transforming Validation set...")
    val_feats = pipeline.transform(val_df)
    print("Transforming Test set...")
    test_feats = pipeline.transform(test_df)

    # Extract Targets and IDs
    y_train = train_df["requester_received_pizza"].values.astype(int)
    y_val = val_df["requester_received_pizza"].values.astype(int)
    test_ids = test_df["request_id"].values

    # Construct Data Dictionary
    data = {
        "X_train_lexical": train_feats["lexical"],
        "X_train_behavioral": train_feats["behavioral"],
        "X_train_semantic": train_feats["semantic"],
        "y_train": y_train,
        "X_val_lexical": val_feats["lexical"],
        "X_val_behavioral": val_feats["behavioral"],
        "X_val_semantic": val_feats["semantic"],
        "y_val": y_val,
        "X_test_lexical": test_feats["lexical"],
        "X_test_behavioral": test_feats["behavioral"],
        "X_test_semantic": test_feats["semantic"],
        "test_ids": test_ids,
    }

    # Save to Cache
    print(f"Saving features to {cache_dir}...")
    sp.save_npz(
        os.path.join(cache_dir, files["X_train_lexical"]), data["X_train_lexical"]
    )
    sp.save_npz(
        os.path.join(cache_dir, files["X_train_behavioral"]), data["X_train_behavioral"]
    )
    np.save(
        os.path.join(cache_dir, files["X_train_semantic"]), data["X_train_semantic"]
    )
    np.save(os.path.join(cache_dir, files["y_train"]), data["y_train"])

    sp.save_npz(os.path.join(cache_dir, files["X_val_lexical"]), data["X_val_lexical"])
    sp.save_npz(
        os.path.join(cache_dir, files["X_val_behavioral"]), data["X_val_behavioral"]
    )
    np.save(os.path.join(cache_dir, files["X_val_semantic"]), data["X_val_semantic"])
    np.save(os.path.join(cache_dir, files["y_val"]), data["y_val"])

    sp.save_npz(
        os.path.join(cache_dir, files["X_test_lexical"]), data["X_test_lexical"]
    )
    sp.save_npz(
        os.path.join(cache_dir, files["X_test_behavioral"]), data["X_test_behavioral"]
    )
    np.save(os.path.join(cache_dir, files["X_test_semantic"]), data["X_test_semantic"])
    np.save(os.path.join(cache_dir, files["test_ids"]), data["test_ids"])

    return data
