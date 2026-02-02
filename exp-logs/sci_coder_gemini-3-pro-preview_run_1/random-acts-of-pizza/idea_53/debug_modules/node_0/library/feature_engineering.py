import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from library.config import Config
from library.utils import load_json, load_metadata_splits


class FeatureEngineer:
    """
    Handles feature engineering for both Random Forest (Stream A) and MLP (Stream B).

    Responsibilities:
    1. Extract and impute numerical metadata.
    2. Engineer ratio features (Credibility Metrics).
    3. Generate Top-K Subreddit binary indicators.
    4. Generate TF-IDF features for text.
    5. Create Interaction Features (Consistency * Credibility).
    6. Scale features for MLP (Arcsinh + StandardScaler).
    """

    def __init__(self):
        # Transformers
        self.tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            stop_words="english",
            norm="l2",
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")

        # State
        self.top_k_subreddits_list = []
        self.is_fitted = False

        # Raw numerical columns to extract from metadata
        self.raw_num_cols = [
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

    def _fit_transformers(self):
        """
        Fits all stateful transformers (Imputer, Scaler, TF-IDF, Top-K) on the training set.
        This ensures no leakage from validation/test sets.
        """
        print("Fitting feature transformers on training data...")

        # Load Train Data
        train_df, _, _ = load_metadata_splits()
        raw_train = load_json(Config.TRAIN_JSON_PATH)
        id_to_data = {d[Config.ID_COL]: d for d in raw_train}

        # 1. Fit TF-IDF
        texts = []
        for rid in train_df[Config.ID_COL]:
            item = id_to_data.get(rid, {})
            title = item.get("request_title", "")
            body = item.get("request_text_edit_aware", "")
            texts.append(f"{title} {body}")
        self.tfidf.fit(texts)

        # 2. Fit Top-K Subreddits
        subreddit_counts = {}
        for rid in train_df[Config.ID_COL]:
            item = id_to_data.get(rid, {})
            subs = item.get("requester_subreddits_at_request", [])
            for sub in subs:
                subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

        sorted_subs = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
        self.top_k_subreddits_list = [
            s[0] for s in sorted_subs[: Config.TOP_K_SUBREDDITS]
        ]

        # 3. Fit Numerical Transformers (Imputer & Scaler)
        # Extract raw numericals
        X_raw = train_df[self.raw_num_cols].values

        # Fit Imputer
        self.imputer.fit(X_raw)
        X_imputed = self.imputer.transform(X_raw)

        # Generate Ratios for Scaling Fit
        # Upvote Ratio: (U-D) / (U+D)
        plus = train_df["requester_upvotes_plus_downvotes_at_request"].values
        minus = train_df["requester_upvotes_minus_downvotes_at_request"].values
        denom = plus.copy()
        denom[denom == 0] = 1.0
        up_ratio = (minus / denom).reshape(-1, 1)

        # Activity Ratio: RAOP Posts / Total Posts
        t_posts = train_df["requester_number_of_posts_at_request"].values
        r_posts = train_df["requester_number_of_posts_on_raop_at_request"].values
        d_posts = t_posts.copy()
        d_posts[d_posts == 0] = 1.0
        act_ratio = (r_posts / d_posts).reshape(-1, 1)

        # Extended feature set
        X_extended = np.hstack([X_imputed, up_ratio, act_ratio])

        # Fit Scaler on Arcsinh-transformed extended features
        self.scaler.fit(np.arcsinh(X_extended))

        self.is_fitted = True
        print("Transformers fitted successfully.")

    def _extract_base_features(self, df):
        """
        Extracts raw numerical features, imputes them, and generates ratio features.
        Returns:
            X_base (np.array): The base feature matrix (Imputed Raw + Ratios).
        """
        # Raw extraction
        X_raw = df[self.raw_num_cols].values

        # Impute
        X_imputed = self.imputer.transform(X_raw)

        # Ratios
        plus = df["requester_upvotes_plus_downvotes_at_request"].values
        minus = df["requester_upvotes_minus_downvotes_at_request"].values
        denom = plus.copy()
        denom[denom == 0] = 1.0
        up_ratio = (minus / denom).reshape(-1, 1)

        t_posts = df["requester_number_of_posts_at_request"].values
        r_posts = df["requester_number_of_posts_on_raop_at_request"].values
        d_posts = t_posts.copy()
        d_posts[d_posts == 0] = 1.0
        act_ratio = (r_posts / d_posts).reshape(-1, 1)

        # Combine
        X_base = np.hstack([X_imputed, up_ratio, act_ratio])
        return X_base

    def _get_top_k_features(self, df, id_to_data):
        """Generates binary indicators for the top-k subreddits."""
        N = len(df)
        K = len(self.top_k_subreddits_list)
        X_topk = np.zeros((N, K), dtype=np.float32)

        sub_to_idx = {sub: i for i, sub in enumerate(self.top_k_subreddits_list)}

        for i, rid in enumerate(df[Config.ID_COL]):
            item = id_to_data.get(rid, {})
            subs = item.get("requester_subreddits_at_request", [])
            for sub in subs:
                if sub in sub_to_idx:
                    X_topk[i, sub_to_idx[sub]] = 1.0

        return X_topk

    def _get_interaction_features(self, X_base, semantic_features):
        """
        Generates interaction features: Consistency_Scalar * log(1 + Credibility_Metric).

        X_base indices mapping (based on self.raw_num_cols + 2 ratios):
        0: account_age
        4: num_posts
        9: upvote_ratio (first appended ratio)
        """
        if semantic_features is None:
            return None

        topic_sim = semantic_features["topic_consistency"]  # (N, 1)
        narr_sim = semantic_features["narrative_consistency"]  # (N, 1)

        # Select Credibility Metrics from X_base
        # 1. Account Age (Index 0)
        acc_age = np.log1p(X_base[:, 0:1])
        # 2. Total Posts (Index 4)
        num_posts = np.log1p(X_base[:, 4:5])
        # 3. Upvote Ratio (Index 9) - Already a ratio, no log needed
        up_ratio = X_base[:, 9:10]

        # Compute Cross-Products
        interactions = []
        # We interact both consistency scalars with all 3 credibility metrics
        for scalar in [topic_sim, narr_sim]:
            interactions.append(scalar * acc_age)
            interactions.append(scalar * num_posts)
            interactions.append(scalar * up_ratio)

        return np.hstack(interactions).astype(np.float32)

    def process_split(
        self, df, split_name, load_cached_data=True, semantic_features=None
    ):
        """
        Main pipeline to generate all feature sets for a given split.

        Args:
            df (pd.DataFrame): Metadata dataframe for the split.
            split_name (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from disk if available.
            semantic_features (dict): Output from SemanticProcessor (required for interactions).

        Returns:
            dict: Dictionary containing 'metadata_rf', 'metadata_mlp', 'top_k', 'tfidf', 'interaction'.
        """
        cache_file = os.path.join(Config.IDEA_DIR, f"tabular_features_{split_name}.npz")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_file):
            print(
                f"Loading cached tabular features for '{split_name}' from {cache_file}"
            )
            try:
                with np.load(cache_file) as data:
                    return {key: data[key] for key in data.files}
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        # 2. Ensure Transformers are Fitted
        if not self.is_fitted:
            self._fit_transformers()

        print(f"Generating tabular features for '{split_name}'...")

        # 3. Load Raw JSON Data
        # Select source file based on split name
        if "train" in split_name or "val" in split_name:
            raw_data = load_json(Config.TRAIN_JSON_PATH)
        else:
            raw_data = load_json(Config.TEST_JSON_PATH)
        id_to_data = {d[Config.ID_COL]: d for d in raw_data}

        # 4. Generate Features

        # A. Base Metadata (Raw Imputed + Ratios) - For RF
        X_base = self._extract_base_features(df)

        # B. MLP Metadata (Arcsinh + Scaled)
        X_mlp = self.scaler.transform(np.arcsinh(X_base)).astype(np.float32)

        # C. Top-K Subreddits
        X_topk = self._get_top_k_features(df, id_to_data)

        # D. TF-IDF
        texts = []
        for rid in df[Config.ID_COL]:
            item = id_to_data.get(rid, {})
            t = item.get("request_title", "")
            b = item.get("request_text_edit_aware", "")
            texts.append(f"{t} {b}")

        # Convert to dense for storage in .npz (vocab is small, ~5000)
        X_tfidf = self.tfidf.transform(texts).astype(np.float32).toarray()

        # E. Interaction Features
        if semantic_features is not None:
            X_inter = self._get_interaction_features(X_base, semantic_features)
        else:
            print(
                f"Warning: Semantic features missing for {split_name}. Interaction features set to zero."
            )
            # 2 scalars * 3 metrics = 6 features
            X_inter = np.zeros((len(df), 6), dtype=np.float32)

        # 5. Save and Return
        data_dict = {
            "metadata_rf": X_base.astype(np.float32),
            "metadata_mlp": X_mlp,
            "top_k": X_topk,
            "tfidf": X_tfidf,
            "interaction": X_inter,
        }

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, **data_dict)
        print(f"Saved tabular features to {cache_file}")

        return data_dict
