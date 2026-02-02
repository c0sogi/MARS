import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from collections import Counter
from library.config import (
    WORKING_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TOP_K_SUBREDDITS,
    RANDOM_STATE,
)


class FeatureEngineer:
    def __init__(self):
        """
        Initializes the FeatureEngineer with necessary scalers and vectorizers.
        """
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            stop_words="english",
        )
        self.top_k_subreddits = []
        self.numerical_cols = []
        self.is_fitted = False

        # Ensure working directory exists
        os.makedirs(WORKING_DIR, exist_ok=True)

    def _get_cache_path(self, dataset_name):
        """
        Returns the path for the cached features file.
        """
        return os.path.join(WORKING_DIR, f"tabular_features_{dataset_name}.npz")

    def _identify_numerical_cols(self, df):
        """
        Identifies numerical columns to be used, excluding leakage and text columns.
        """
        exclude = [
            "requester_received_pizza",
            "request_id",
            "giver_username_if_known",
            "request_text",
            "request_title",
            "request_text_edit_aware",
            "requester_subreddits_at_request",
            "source_file",
            "post_was_edited",
        ]

        # Select numeric types
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Select boolean types (will be converted to int)
        bool_cols = df.select_dtypes(include=[bool]).columns.tolist()

        candidates = cols + bool_cols
        return sorted([c for c in candidates if c not in exclude])

    def fit(self, train_df):
        """
        Fits the imputer, scaler, TF-IDF vectorizer, and determines top-K subreddits.
        """
        # 1. Numerical Columns
        self.numerical_cols = self._identify_numerical_cols(train_df)

        # Extract and prepare numerical data
        num_data = train_df[self.numerical_cols].copy()
        for c in num_data.columns:
            if num_data[c].dtype == bool:
                num_data[c] = num_data[c].astype(int)

        # Fit Imputer
        num_data = self.imputer.fit_transform(num_data)

        # Fit Scaler (on Arcsinh transformed data)
        num_data_arcsinh = np.arcsinh(num_data)
        self.scaler.fit(num_data_arcsinh)

        # 2. TF-IDF
        # Combine title and body for a rich text representation
        titles = train_df["request_title"].fillna("").astype(str)
        if "request_text_edit_aware" in train_df.columns:
            bodies = train_df["request_text_edit_aware"].fillna("").astype(str)
        else:
            bodies = train_df["request_text"].fillna("").astype(str)

        text_data = (titles + " " + bodies).tolist()
        self.tfidf.fit(text_data)

        # 3. Top K Subreddits
        all_subs = []
        for sub_list in train_df["requester_subreddits_at_request"]:
            if isinstance(sub_list, list):
                all_subs.extend(sub_list)
            elif isinstance(sub_list, np.ndarray):
                all_subs.extend(sub_list.tolist())

        if all_subs:
            counts = Counter(all_subs)
            self.top_k_subreddits = [x[0] for x in counts.most_common(TOP_K_SUBREDDITS)]
        else:
            self.top_k_subreddits = []

        self.is_fitted = True

    def transform(self, df, sbert_features=None):
        """
        Transforms the dataframe into feature sets for MLP and RF.
        """
        # 1. Metadata Processing
        num_df = df[self.numerical_cols].copy()
        for c in num_df.columns:
            if num_df[c].dtype == bool:
                num_df[c] = num_df[c].astype(int)

        # Impute
        num_data = self.imputer.transform(num_df)

        # MLP Features: Arcsinh + Scaled
        mlp_meta = np.arcsinh(num_data)
        mlp_meta = self.scaler.transform(mlp_meta)

        # RF Features: Raw (imputed) magnitude
        rf_meta = num_data

        # 2. TF-IDF (Dense for RF)
        titles = df["request_title"].fillna("").astype(str)
        if "request_text_edit_aware" in df.columns:
            bodies = df["request_text_edit_aware"].fillna("").astype(str)
        else:
            bodies = df["request_text"].fillna("").astype(str)
        text_data = (titles + " " + bodies).tolist()

        tfidf_mat = self.tfidf.transform(text_data).toarray().astype(np.float32)

        # 3. Top K Subreddits (Binary Flags)
        top_k_mat = np.zeros((len(df), len(self.top_k_subreddits)), dtype=np.float32)
        if self.top_k_subreddits:
            sub_to_idx = {sub: i for i, sub in enumerate(self.top_k_subreddits)}
            for i, sub_list in enumerate(df["requester_subreddits_at_request"]):
                if isinstance(sub_list, (list, np.ndarray)):
                    for sub in sub_list:
                        if sub in sub_to_idx:
                            top_k_mat[i, sub_to_idx[sub]] = 1.0

        # 4. Consistency Scalars (Cosine Similarity)
        consistency_feats = np.zeros((len(df), 2), dtype=np.float32)
        if sbert_features is not None:

            def get_cosine(a, b):
                # a, b: (N, D)
                norm_a = np.linalg.norm(a, axis=1)
                norm_b = np.linalg.norm(b, axis=1)
                dot = np.sum(a * b, axis=1)
                denom = norm_a * norm_b
                # Avoid div by zero
                res = np.zeros_like(dot)
                mask = denom > 1e-9
                res[mask] = dot[mask] / denom[mask]
                return res

            title_emb = sbert_features["title_emb"]
            body_emb = sbert_features["body_emb"]
            hist_centroid = sbert_features["hist_centroid"]

            consistency_feats[:, 0] = get_cosine(
                title_emb, hist_centroid
            )  # Topic Consistency
            consistency_feats[:, 1] = get_cosine(
                body_emb, hist_centroid
            )  # Narrative Consistency

        # 5. Explicit Interactions
        # Interaction between Consistency and Credibility (Age, Score)
        interactions = np.zeros((len(df), 3), dtype=np.float32)

        try:
            age_col_name = "requester_account_age_in_days_at_request"
            score_col_name = "requester_upvotes_minus_downvotes_at_request"

            if (
                age_col_name in self.numerical_cols
                and score_col_name in self.numerical_cols
            ):
                age_idx = self.numerical_cols.index(age_col_name)
                score_idx = self.numerical_cols.index(score_col_name)

                # Log transform age for interaction to dampen outliers
                age_vals = np.log1p(np.maximum(0, rf_meta[:, age_idx]))
                score_vals = rf_meta[:, score_idx]

                interactions[:, 0] = consistency_feats[:, 0] * age_vals  # Topic * Age
                interactions[:, 1] = (
                    consistency_feats[:, 1] * age_vals
                )  # Narrative * Age
                interactions[:, 2] = (
                    consistency_feats[:, 0] * score_vals
                )  # Topic * Score
        except Exception:
            pass  # Keep zeros if columns missing

        return {
            "mlp_metadata": mlp_meta.astype(np.float32),
            "rf_tfidf": tfidf_mat,
            "rf_metadata": rf_meta.astype(np.float32),
            "rf_top_k": top_k_mat,
            "consistency": consistency_feats,
            "rf_interactions": interactions,
        }

    def generate_features(
        self,
        df,
        dataset_name,
        sbert_features=None,
        train_df=None,
        load_cached_data=True,
    ):
        """
        Orchestrates feature generation with caching.
        """
        cache_path = self._get_cache_path(dataset_name)

        # Always fit if train_df provided, to ensure state is ready for this or future calls
        # This is fast and ensures the object is stateful even if we load output from cache
        if train_df is not None:
            print("Fitting FeatureEngineer on training data...")
            self.fit(train_df)

        if load_cached_data and os.path.exists(cache_path):
            print(
                f"Loading cached tabular features for {dataset_name} from {cache_path}..."
            )
            try:
                loaded = np.load(cache_path)
                return {k: loaded[k] for k in loaded.files}
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        if not self.is_fitted:
            raise ValueError(
                "FeatureEngineer must be fitted on training data before transforming."
            )

        print(f"Generating tabular features for {dataset_name}...")
        features = self.transform(df, sbert_features)

        print(f"Saving tabular features for {dataset_name} to {cache_path}...")
        np.savez_compressed(cache_path, **features)

        return features
