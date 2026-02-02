import os
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer
from library.config import (
    WORKING_DIR,
    NUMERIC_COLS,
    TEXT_TITLE_COL,
    TEXT_BODY_COL,
    HISTORY_COL,
    TARGET_COL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    BAYESIAN_SMOOTHING_K,
    SBERT_MODEL_NAME,
    SBERT_BATCH_SIZE,
    SBERT_EMBEDDING_DIM,
    MAX_HISTORY_LEN,
    RANDOM_SEED,
)
from library.utils import ensure_directory


class MetadataExtractor:
    """
    Extracts and engineers numerical features from the dataset.
    Handles raw magnitudes, ratios, and text meta-features.
    """

    def __init__(self):
        self.numeric_cols = NUMERIC_COLS

    def process(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        cache_path = os.path.join(WORKING_DIR, f"metadata_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached metadata from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Generating metadata for {split_name}...")

        # 1. Extract base numeric columns (intersection of train/test)
        # Fill NaNs with 0 for safety, though imputation happens later in pipeline
        features = df[self.numeric_cols].fillna(0).copy()

        # 2. Engineered Ratios
        # Upvote Ratio
        up = df.get("requester_upvotes_plus_downvotes_at_request", 0)
        diff = df.get("requester_upvotes_minus_downvotes_at_request", 0)
        # derive downvotes from (sum - diff) / 2
        # ratio = upvotes / (total + 1)
        # Actually, let's just use the diff ratio directly as a proxy for quality
        features["upvote_ratio"] = diff / (up + 1.0)

        # Comment/Post Ratio
        comments = df.get("requester_number_of_comments_at_request", 0)
        posts = df.get("requester_number_of_posts_at_request", 0)
        features["comment_post_ratio"] = comments / (posts + 1.0)

        # RAOP Interaction Ratio
        raop_comments = df.get("requester_number_of_comments_in_raop_at_request", 0)
        raop_posts = df.get("requester_number_of_posts_on_raop_at_request", 0)
        features["raop_activity_ratio"] = (raop_comments + raop_posts) / (
            comments + posts + 1.0
        )

        # 3. Text Meta-Features
        # Handle potential NaNs in text columns
        title = df[TEXT_TITLE_COL].fillna("").astype(str)
        body = df[TEXT_BODY_COL].fillna("").astype(str)

        features["title_len_char"] = title.apply(len)
        features["body_len_char"] = body.apply(len)
        features["title_len_word"] = title.apply(lambda x: len(x.split()))
        features["body_len_word"] = body.apply(lambda x: len(x.split()))

        # Caps Ratio (shouting)
        def get_caps_ratio(text):
            if len(text) == 0:
                return 0.0
            return sum(1 for c in text if c.isupper()) / len(text)

        features["title_caps_ratio"] = title.apply(get_caps_ratio)
        features["body_caps_ratio"] = body.apply(get_caps_ratio)

        # Save to cache
        ensure_directory(cache_path)
        features.to_parquet(cache_path)
        print(f"Saved metadata to {cache_path}")

        return features


class TextProcessor:
    """
    Handles TF-IDF vectorization for Stream A (Random Forest).
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        self.is_fitted = False

    def _get_text_corpus(self, df: pd.DataFrame) -> pd.Series:
        return df[TEXT_TITLE_COL].fillna("") + " " + df[TEXT_BODY_COL].fillna("")

    def fit_transform(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ):
        cache_path = os.path.join(WORKING_DIR, f"tfidf_{split_name}.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached TF-IDF from {cache_path}")
            # We still need to fit the vectorizer for subsequent transforms if we are in a pipeline,
            # but usually we load the vectorizer from disk too.
            # For this simplified script, if cache exists, we assume we might not need the vectorizer object
            # immediately unless transform is called.
            # However, to be safe, we re-fit if we can't load a pickled vectorizer (not implemented here).
            # We will just re-fit quickly since we have the data, OR just return the matrix.
            # Re-fitting is safer to ensure state consistency for transform() calls.
            corpus = self._get_text_corpus(df)
            self.vectorizer.fit(corpus)
            self.is_fitted = True
            return scipy.sparse.load_npz(cache_path)

        print(f"Computing TF-IDF for {split_name}...")
        corpus = self._get_text_corpus(df)
        matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

        ensure_directory(cache_path)
        scipy.sparse.save_npz(cache_path, matrix)
        print(f"Saved TF-IDF to {cache_path}")
        return matrix

    def transform(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ):
        if not self.is_fitted:
            raise ValueError("TextProcessor must be fitted before transform.")

        cache_path = os.path.join(WORKING_DIR, f"tfidf_{split_name}.npz")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached TF-IDF from {cache_path}")
            return scipy.sparse.load_npz(cache_path)

        print(f"Computing TF-IDF for {split_name}...")
        corpus = self._get_text_corpus(df)
        matrix = self.vectorizer.transform(corpus)

        ensure_directory(cache_path)
        scipy.sparse.save_npz(cache_path, matrix)
        print(f"Saved TF-IDF to {cache_path}")
        return matrix


class BayesianHistoryEncoder:
    """
    Implements Bayesian Target Encoding for user history (subreddits).
    """

    def __init__(self):
        self.subreddit_stats = {}
        self.global_mean = 0.0
        self.k = BAYESIAN_SMOOTHING_K

    def _calculate_smoothed_score(self, n, mean):
        return (n * mean + self.k * self.global_mean) / (n + self.k)

    def fit_transform_train(
        self, df: pd.DataFrame, split_name: str = "train", load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Uses Stratified K-Fold to generate out-of-fold encodings for the training set
        to prevent data leakage.
        """
        cache_path = os.path.join(WORKING_DIR, f"bayes_history_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Bayesian encoding from {cache_path}")
            # We still need to fit the global stats for inference on test set
            self._fit_global_stats(df)
            return pd.read_parquet(cache_path)

        print(f"Computing Bayesian History Encoding for {split_name}...")

        # Initialize output features
        feature_names = ["hist_mean_success", "hist_max_success", "hist_min_success"]
        output_df = pd.DataFrame(0.0, index=df.index, columns=feature_names)

        # 1. Fit Global Stats (for use in transform and smoothing)
        self._fit_global_stats(df)

        # 2. Stratified K-Fold for OOF generation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        # Reset index to ensure alignment
        df_reset = df.reset_index(drop=True)
        y = df_reset[TARGET_COL]

        for fold, (train_idx, val_idx) in enumerate(skf.split(df_reset, y)):
            # Training subset for this fold
            fold_train = df_reset.iloc[train_idx]
            fold_val = df_reset.iloc[val_idx]

            # Calculate stats on fold_train
            fold_exploded = fold_train[[HISTORY_COL, TARGET_COL]].explode(HISTORY_COL)
            # Filter out empty/NaN subreddits
            fold_exploded = fold_exploded[fold_exploded[HISTORY_COL].notna()]

            stats = fold_exploded.groupby(HISTORY_COL)[TARGET_COL].agg(
                ["count", "mean"]
            )
            fold_global_mean = fold_train[TARGET_COL].mean()

            # Compute scores for subreddits
            stats["score"] = (
                stats["count"] * stats["mean"] + self.k * fold_global_mean
            ) / (stats["count"] + self.k)
            score_map = stats["score"].to_dict()

            # Apply to fold_val
            self._apply_scores(
                fold_val, score_map, output_df, val_idx, fold_global_mean
            )

        # Handle index alignment if original df had non-range index
        output_df.index = df.index

        ensure_directory(cache_path)
        output_df.to_parquet(cache_path)
        print(f"Saved Bayesian encoding to {cache_path}")
        return output_df

    def _fit_global_stats(self, df: pd.DataFrame):
        """Fit stats on full training data for use on test set."""
        exploded = df[[HISTORY_COL, TARGET_COL]].explode(HISTORY_COL)
        exploded = exploded[exploded[HISTORY_COL].notna()]

        stats = exploded.groupby(HISTORY_COL)[TARGET_COL].agg(["count", "mean"])
        self.global_mean = df[TARGET_COL].mean()

        stats["score"] = (
            stats["count"] * stats["mean"] + self.k * self.global_mean
        ) / (stats["count"] + self.k)
        self.subreddit_stats = stats["score"].to_dict()

    def transform(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        cache_path = os.path.join(WORKING_DIR, f"bayes_history_{split_name}.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Bayesian encoding from {cache_path}")
            return pd.read_parquet(cache_path)

        print(f"Computing Bayesian History Encoding for {split_name}...")
        feature_names = ["hist_mean_success", "hist_max_success", "hist_min_success"]
        output_df = pd.DataFrame(0.0, index=df.index, columns=feature_names)

        # Apply globally learned stats
        # We pass range(len(df)) as indices because we are processing the whole DF
        self._apply_scores(
            df.reset_index(drop=True),
            self.subreddit_stats,
            output_df,
            range(len(df)),
            self.global_mean,
        )

        output_df.index = df.index

        ensure_directory(cache_path)
        output_df.to_parquet(cache_path)
        print(f"Saved Bayesian encoding to {cache_path}")
        return output_df

    def _apply_scores(self, df_subset, score_map, output_df, indices, default_val):
        """Helper to map scores to history lists and aggregate."""
        # This is the slow part, optimizing with list comprehension

        histories = df_subset[HISTORY_COL].tolist()

        means, maxs, mins = [], [], []

        for hist in histories:
            if not isinstance(hist, list) or len(hist) == 0:
                means.append(default_val)
                maxs.append(default_val)
                mins.append(default_val)
                continue

            scores = [score_map.get(sub, default_val) for sub in hist]

            if not scores:
                means.append(default_val)
                maxs.append(default_val)
                mins.append(default_val)
            else:
                means.append(np.mean(scores))
                maxs.append(np.max(scores))
                mins.append(np.min(scores))

        # Assign back using iloc for safety
        output_df.iloc[indices, 0] = means
        output_df.iloc[indices, 1] = maxs
        output_df.iloc[indices, 2] = mins


class SBERTHandler:
    """
    Generates SBERT embeddings for Requests and User History.
    Handles sequence padding for history.
    """

    def __init__(self):
        self.model_name = SBERT_MODEL_NAME
        self.batch_size = SBERT_BATCH_SIZE
        self.embedding_dim = SBERT_EMBEDDING_DIM
        self.max_len = MAX_HISTORY_LEN
        self.model = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name, device="cpu"
            )  # Use CPU to save GPU for training
            if os.environ.get("CUDA_VISIBLE_DEVICES") and hasattr(self.model, "to"):
                self.model.to("cuda")

    def encode_requests(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        cache_path = os.path.join(WORKING_DIR, f"sbert_request_{split_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached Request Embeddings from {cache_path}")
            return np.load(cache_path)

        self._load_model()
        print(f"Encoding Requests for {split_name}...")

        # Combine Title + Body
        texts = (
            df[TEXT_TITLE_COL].fillna("") + " " + df[TEXT_BODY_COL].fillna("")
        ).tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        ensure_directory(cache_path)
        np.save(cache_path, embeddings)
        print(f"Saved Request Embeddings to {cache_path}")
        return embeddings

    def encode_history(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        cache_path = os.path.join(WORKING_DIR, f"sbert_history_{split_name}.npy")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached History Embeddings from {cache_path}")
            return np.load(cache_path)

        self._load_model()
        print(f"Encoding History for {split_name}...")

        # 1. Identify all unique subreddits in this split to minimize encoding calls
        all_subreddits = set()
        for hist in df[HISTORY_COL]:
            if isinstance(hist, list):
                all_subreddits.update(hist)

        unique_subs = list(all_subreddits)
        print(f"Unique subreddits to encode: {len(unique_subs)}")

        # 2. Encode unique subreddits
        if len(unique_subs) > 0:
            sub_embeddings = self.model.encode(
                unique_subs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}
        else:
            sub_map = {}

        # 3. Construct 3D array (N, MAX_LEN, DIM)
        num_samples = len(df)
        output_tensor = np.zeros(
            (num_samples, self.max_len, self.embedding_dim), dtype=np.float32
        )

        histories = df[HISTORY_COL].tolist()

        for i, hist in enumerate(histories):
            if not isinstance(hist, list) or len(hist) == 0:
                continue

            # Truncate if too long
            hist = hist[: self.max_len]

            for j, sub in enumerate(hist):
                if sub in sub_map:
                    output_tensor[i, j, :] = sub_map[sub]

        ensure_directory(cache_path)
        np.save(cache_path, output_tensor)
        print(f"Saved History Embeddings to {cache_path}")
        return output_tensor
