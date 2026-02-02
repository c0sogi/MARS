import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

from library.config import (
    WORKING_DIR,
    CACHE_RF_FEATURES,
    CACHE_MLP_FEATURES,
    RAW_NUMERIC_COLS,
    LIST_COL,
    TOP_K_CONFIG,
    TARGET_COL,
    SEED,
    INTERACTION_FEATURES,
)
from library.data_loader import get_common_features
from library.text_processing import compute_consistency_scores

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)


class MetadataProcessor:
    """
    Handles processing of numerical metadata and Top-K subreddit generation.
    """

    def __init__(self, top_k=50):
        self.top_k = top_k
        self.top_subreddits = []
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols = []

    def fit_top_k(self, train_df):
        """Identifies the top K most frequent subreddits in the training set."""
        all_subreddits = []
        # Flatten the list of subreddits
        for sub_list in train_df[LIST_COL]:
            if isinstance(sub_list, list):
                all_subreddits.extend(sub_list)

        counts = Counter(all_subreddits)
        self.top_subreddits = [sub for sub, count in counts.most_common(self.top_k)]
        return self

    def transform_top_k(self, df):
        """Generates a binary matrix (N, K) for the top subreddits."""
        if not self.top_subreddits:
            raise ValueError("Top K subreddits not fitted.")

        # Create a mapping for speed
        sub_to_idx = {sub: i for i, sub in enumerate(self.top_subreddits)}
        n_samples = len(df)
        matrix = np.zeros((n_samples, self.top_k), dtype=np.float32)

        for row_idx, sub_list in enumerate(df[LIST_COL]):
            if isinstance(sub_list, list):
                for sub in sub_list:
                    if sub in sub_to_idx:
                        matrix[row_idx, sub_to_idx[sub]] = 1.0
        return matrix

    def fit_numeric(self, train_df):
        """Fits imputer and scaler on raw numeric columns."""
        # Restrict to columns present in the dataframe
        self.numeric_cols = [c for c in RAW_NUMERIC_COLS if c in train_df.columns]

        X = train_df[self.numeric_cols].values
        # Impute first
        X_imputed = self.imputer.fit_transform(X)
        # Apply Arcsinh (log-like) transformation for skewness
        X_trans = np.arcsinh(X_imputed)
        # Fit scaler
        self.scaler.fit(X_trans)
        return self

    def transform_numeric(self, df, apply_scaling=True):
        """Transforms numeric columns: Impute -> Arcsinh -> (Optional) Scale."""
        if not self.numeric_cols:
            return np.zeros((len(df), 0))

        X = df[self.numeric_cols].values
        X_imputed = self.imputer.transform(X)
        X_trans = np.arcsinh(X_imputed)

        if apply_scaling:
            return self.scaler.transform(X_trans).astype(np.float32)
        else:
            return X_trans.astype(np.float32)


class InteractionProjector:
    """
    Generates interaction features specifically for the Random Forest model.
    """

    @staticmethod
    def generate(df, consistency_scores, top_k_matrix, split_name):
        """
        Computes interaction terms based on consistency scores and metadata.

        Args:
            df: DataFrame containing raw metadata.
            consistency_scores: Dict containing '{split}_title_consistency', etc.
            top_k_matrix: Binary matrix of top-k subreddits.
            split_name: 'train', 'val', or 'test'.

        Returns:
            np.array: Matrix of interaction features.
        """
        # Extract base vectors
        title_cons = consistency_scores[f"{split_name}_title_consistency"]
        body_cons = consistency_scores[f"{split_name}_body_consistency"]

        # Metadata vectors (handle NaNs with 0 for calculation safety, though RF handles NaNs,
        # explicit calculation needs values. We use fillna(0) temporarily).
        age = (
            df.get("requester_account_age_in_days_at_request", pd.Series(0))
            .fillna(0)
            .values
        )
        upvotes = (
            df.get("requester_upvotes_minus_downvotes_at_request", pd.Series(0))
            .fillna(0)
            .values
        )
        total_votes = (
            df.get("requester_upvotes_plus_downvotes_at_request", pd.Series(0))
            .fillna(0)
            .values
        )

        sum_top_k = top_k_matrix.sum(axis=1)

        # 1. Title Consistency * Log(Age)
        # I1 = title_consistency * log(1 + Account_Age)
        i1 = title_cons * np.log1p(age)

        # 2. Body Consistency * Upvote Ratio
        # I2 = body_consistency * (Up - Down) / (Up + Down + 1)
        # Note: upvotes_minus_downvotes is (Up - Down).
        # total_votes is (Up + Down).
        ratio = upvotes / (total_votes + 1.0)
        i2 = body_cons * ratio

        # 3. Title Consistency * Sum(Top-K Flags)
        i3 = title_cons * sum_top_k

        # Stack features
        interactions = np.column_stack([i1, i2, i3]).astype(np.float32)
        return interactions


def prepare_rf_features(
    train_df, val_df, test_df, tfidf_data, sbert_data, load_cached_data=True
):
    """
    Prepares the comprehensive feature set for the Random Forest model (Stream A).
    Combines TF-IDF, Raw Metadata, Top-K Flags, Consistency Scores, and Interactions.
    """
    if load_cached_data and os.path.exists(CACHE_RF_FEATURES):
        print(f"Loading RF features from {CACHE_RF_FEATURES}...")
        return np.load(CACHE_RF_FEATURES)

    print("Generating RF features from scratch...")

    # 1. Process Metadata (Fit on Train)
    processor = MetadataProcessor(top_k=TOP_K_CONFIG["k"])
    processor.fit_top_k(train_df)

    # For RF, we use raw (imputed) numeric features, not scaled ones,
    # but we use the processor to handle imputation consistently.
    processor.fit_numeric(train_df)

    # 2. Compute Consistency Scores
    consistency_scores = compute_consistency_scores(sbert_data)

    output_data = {}
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    for split_name, df in splits:
        print(f"  Processing {split_name} features for RF...")

        # A. TF-IDF
        X_tfidf = tfidf_data[f"{split_name}_tfidf"]

        # B. Numeric Metadata (Imputed, unscaled for RF usually, but Arcsinh helps trees too.
        # We will use the transformed version from processor but unscaled?
        # Actually, trees are scale-invariant, but Arcsinh helps with skew.
        # Let's use the transformed (Arcsinh) output but maybe not StandardScaled.
        # For simplicity and robustness, we use the output of transform_numeric(apply_scaling=False)).
        X_numeric = processor.transform_numeric(df, apply_scaling=False)

        # C. Top-K Flags
        X_topk = processor.transform_top_k(df)

        # D. Consistency Scores
        c_title = consistency_scores[f"{split_name}_title_consistency"][:, np.newaxis]
        c_body = consistency_scores[f"{split_name}_body_consistency"][:, np.newaxis]

        # E. Interaction Features
        X_interaction = InteractionProjector.generate(
            df, consistency_scores, X_topk, split_name
        )

        # Concatenate all
        X_combined = np.hstack(
            [X_tfidf, X_numeric, X_topk, c_title, c_body, X_interaction]
        ).astype(np.float32)

        output_data[f"X_{split_name}"] = X_combined

        # Handle Target
        if TARGET_COL in df.columns:
            output_data[f"y_{split_name}"] = df[TARGET_COL].astype(int).values
        else:
            # For test set, placeholder
            output_data[f"y_{split_name}"] = np.zeros(len(df), dtype=int)

    # Save to cache
    print(f"Saving RF features to {CACHE_RF_FEATURES}...")
    np.savez_compressed(CACHE_RF_FEATURES, **output_data)

    return output_data


def prepare_mlp_features(train_df, val_df, test_df, sbert_data, load_cached_data=True):
    """
    Prepares the tensor dictionary for the MLP model (Stream B).
    Includes SBERT embeddings (Title, Body, History) and Processed Metadata (FiLM input).
    """
    if load_cached_data and os.path.exists(CACHE_MLP_FEATURES):
        print(f"Loading MLP features from {CACHE_MLP_FEATURES}...")
        return np.load(CACHE_MLP_FEATURES)

    print("Generating MLP features from scratch...")

    # 1. Process Metadata (Fit on Train)
    # For MLP, we strictly need Scaling.
    processor = MetadataProcessor(top_k=TOP_K_CONFIG["k"])
    processor.fit_top_k(train_df)
    processor.fit_numeric(train_df)

    output_data = {}
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    for split_name, df in splits:
        print(f"  Processing {split_name} features for MLP...")

        # A. Metadata for FiLM (Scaled Numeric + Binary Top-K)
        X_numeric_scaled = processor.transform_numeric(df, apply_scaling=True)
        X_topk = processor.transform_top_k(df)

        # Concatenate to form the conditioning vector 'z'
        metadata_vec = np.hstack([X_numeric_scaled, X_topk]).astype(np.float32)
        output_data[f"{split_name}_metadata"] = metadata_vec

        # B. SBERT Embeddings (Directly from text_processing output)
        output_data[f"{split_name}_title_emb"] = sbert_data[f"{split_name}_title"]
        output_data[f"{split_name}_body_emb"] = sbert_data[f"{split_name}_body"]
        output_data[f"{split_name}_hist_centroid"] = sbert_data[
            f"{split_name}_hist_centroid"
        ]
        output_data[f"{split_name}_hist_seq"] = sbert_data[f"{split_name}_hist_seq"]

        # C. Target
        if TARGET_COL in df.columns:
            output_data[f"{split_name}_target"] = (
                df[TARGET_COL].astype(np.float32).values
            )
        else:
            output_data[f"{split_name}_target"] = np.zeros(len(df), dtype=np.float32)

    # Save to cache
    print(f"Saving MLP features to {CACHE_MLP_FEATURES}...")
    np.savez_compressed(CACHE_MLP_FEATURES, **output_data)

    return output_data
