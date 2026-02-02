import os
import ast
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from library import config


class FeatureEngineer:
    """
    Handles feature engineering for the Pizza Request dataset, including:
    1. Metadata Engineering (Ratios, Arcsinh transforms)
    2. Dual-Lexical TF-IDF (Title + Body)
    3. Zero-Shot Action Profiling (SBERT-based behavioral clustering)
    """

    def __init__(self):
        self.sbert_model = None
        self.scaler = StandardScaler()
        self.tfidf_title = TfidfVectorizer(
            min_df=5,
            max_features=5000,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            stop_words="english",
        )
        self.tfidf_body = TfidfVectorizer(
            min_df=5,
            max_features=10000,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            stop_words="english",
        )

    def _load_sbert(self):
        """Lazy loader for SBERT model."""
        if self.sbert_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading SBERT model ({config.SBERT_MODEL_NAME}) on {device}...")
            self.sbert_model = SentenceTransformer(
                config.SBERT_MODEL_NAME, device=device
            )

    def generate_metadata_features(self, df):
        """
        Generates numerical metadata features including:
        - Raw magnitudes (Arcsinh transformed)
        - Engineered ratios
        - Text meta-features

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Dataframe containing only the numerical engineered features.
        """
        # Initialize output dataframe
        meta_df = pd.DataFrame(index=df.index)

        # --- 1. Text Meta-Features ---
        # Fill NaNs for text processing
        title = df[config.TEXT_COL_TITLE].fillna("").astype(str)
        body = df[config.TEXT_COL_BODY].fillna("").astype(str)

        meta_df["title_len_char"] = title.apply(len)
        meta_df["body_len_char"] = body.apply(len)
        meta_df["title_len_word"] = title.apply(lambda x: len(x.split()))
        meta_df["body_len_word"] = body.apply(lambda x: len(x.split()))

        # Caps ratio (shouting indicator)
        def get_caps_ratio(text):
            if len(text) == 0:
                return 0.0
            return sum(1 for c in text if c.isupper()) / len(text)

        meta_df["title_caps_ratio"] = title.apply(get_caps_ratio)
        meta_df["body_caps_ratio"] = body.apply(get_caps_ratio)

        # --- 2. User History (Raw & Arcsinh Transformed) ---
        # We apply arcsinh to handle skewed distributions (count data)
        # Arcsinh is similar to log but handles 0 gracefully: asinh(x) ~= log(x + sqrt(x^2+1))

        cols_to_transform = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        for col in cols_to_transform:
            if col in df.columns:
                # Handle potential negative values for 'minus_downvotes' correctly with arcsinh
                meta_df[f"{col}_asinh"] = np.arcsinh(df[col].fillna(0))
            else:
                # Fallback if column missing (leakage prevention might remove some)
                pass

        # --- 3. Engineered Ratios ---
        # Upvote Ratio: Up / (Up + Down)
        # Approximated by: (Total + Net) / 2 / Total
        # Total = Up + Down; Net = Up - Down => Up = (Total + Net) / 2
        if (
            "requester_upvotes_plus_downvotes_at_request" in df.columns
            and "requester_upvotes_minus_downvotes_at_request" in df.columns
        ):

            total = df["requester_upvotes_plus_downvotes_at_request"].fillna(0)
            net = df["requester_upvotes_minus_downvotes_at_request"].fillna(0)

            # Avoid division by zero
            safe_total = total.replace(0, 1)

            upvotes = (total + net) / 2
            meta_df["requester_upvote_ratio"] = upvotes / safe_total
            # If total was 0, ratio is 0.5 (neutral) or 0.0
            meta_df.loc[total == 0, "requester_upvote_ratio"] = 0.5

        # Comment to Post Ratio
        if (
            "requester_number_of_comments_at_request" in df.columns
            and "requester_number_of_posts_at_request" in df.columns
        ):

            comments = df["requester_number_of_comments_at_request"].fillna(0)
            posts = df["requester_number_of_posts_at_request"].fillna(0)
            meta_df["requester_comment_post_ratio"] = comments / posts.replace(0, 1)

        return meta_df

    def generate_tfidf_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates Dual-Lexical TF-IDF features (Title + Body).
        Fits on Train, transforms Val and Test.

        Args:
            train_df, val_df, test_df: Dataframes.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            tuple: (train_tfidf, val_tfidf, test_tfidf) as scipy.sparse.csr_matrix
        """
        cache_file = os.path.join(config.WORKING_DIR, "tfidf_features.npz")

        if load_cached_data and os.path.exists(cache_file):
            print("Loading TF-IDF features from cache...")
            data = np.load(cache_file, allow_pickle=True)
            # Reconstruct sparse matrices
            return (data["train"].item(), data["val"].item(), data["test"].item())

        print("Generating TF-IDF features...")

        # Prepare text data
        train_title = train_df[config.TEXT_COL_TITLE].fillna("").astype(str)
        train_body = train_df[config.TEXT_COL_BODY].fillna("").astype(str)

        val_title = val_df[config.TEXT_COL_TITLE].fillna("").astype(str)
        val_body = val_df[config.TEXT_COL_BODY].fillna("").astype(str)

        test_title = test_df[config.TEXT_COL_TITLE].fillna("").astype(str)
        test_body = test_df[config.TEXT_COL_BODY].fillna("").astype(str)

        # Fit and Transform Title
        print("  Fitting Title TF-IDF...")
        self.tfidf_title.fit(train_title)
        train_title_vec = self.tfidf_title.transform(train_title)
        val_title_vec = self.tfidf_title.transform(val_title)
        test_title_vec = self.tfidf_title.transform(test_title)

        # Fit and Transform Body
        print("  Fitting Body TF-IDF...")
        self.tfidf_body.fit(train_body)
        train_body_vec = self.tfidf_body.transform(train_body)
        val_body_vec = self.tfidf_body.transform(val_body)
        test_body_vec = self.tfidf_body.transform(test_body)

        # Stack features
        train_features = scipy.sparse.hstack([train_title_vec, train_body_vec]).tocsr()
        val_features = scipy.sparse.hstack([val_title_vec, val_body_vec]).tocsr()
        test_features = scipy.sparse.hstack([test_title_vec, test_body_vec]).tocsr()

        # Save to cache
        print(f"Saving TF-IDF features to {cache_file}...")
        np.savez_compressed(
            cache_file, train=train_features, val=val_features, test=test_features
        )

        return train_features, val_features, test_features

    def generate_zero_shot_profiles(self, df, split_name, load_cached_data=True):
        """
        Generates Zero-Shot Action Profiles.
        Encodes user subreddits and computes semantic similarity to predefined anchors.

        Args:
            df (pd.DataFrame): Input dataframe containing 'requester_subreddits_at_request'.
            split_name (str): Name of the split (train/val/test) for caching.
            load_cached_data (bool): Whether to load from cache.

        Returns:
            pd.DataFrame: Dataframe with profile scores (e.g., 'profile_sim_0', 'profile_sim_1'...).
        """
        cache_file = os.path.join(
            config.WORKING_DIR, f"action_profiles_{split_name}.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading Action Profiles for {split_name} from cache...")
            return pd.read_parquet(cache_file)

        print(f"Generating Action Profiles for {split_name}...")
        self._load_sbert()

        # 1. Parse Subreddits
        # The column might contain string representations of lists "['a', 'b']" or actual lists
        def parse_subreddits(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return []
            elif isinstance(x, list):
                return x
            return []

        # Extract list of subreddits per user
        user_subreddits = df["requester_subreddits_at_request"].apply(parse_subreddits)

        # 2. Identify Unique Subreddits
        unique_subs = set()
        for subs in user_subreddits:
            unique_subs.update(subs)
        unique_subs = sorted(list(unique_subs))

        # Map subreddit to index
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subs)}

        print(
            f"  Encoding {len(unique_subs)} unique subreddits and {len(config.SEMANTIC_ANCHORS)} anchors..."
        )

        # 3. Encode Anchors and Subreddits
        # Anchors: [N_anchors, Dim]
        anchor_embeddings = self.sbert_model.encode(
            config.SEMANTIC_ANCHORS, convert_to_tensor=True
        )
        # Subreddits: [N_subs, Dim]
        # Batch encode subreddits to avoid OOM
        sub_embeddings = self.sbert_model.encode(
            unique_subs, batch_size=64, show_progress_bar=False, convert_to_tensor=True
        )

        # 4. Compute Similarity Matrix [N_subs, N_anchors]
        # Cosine similarity
        from sentence_transformers import util

        # util.cos_sim returns [N1, N2]
        sim_matrix = util.cos_sim(sub_embeddings, anchor_embeddings).cpu().numpy()

        # 5. Map back to Users
        # For each user, get indices of their subreddits, look up rows in sim_matrix, and average
        profile_features = []

        for subs in user_subreddits:
            if not subs:
                # No history -> Zero profile
                profile_features.append(np.zeros(len(config.SEMANTIC_ANCHORS)))
                continue

            # Get indices for valid subreddits (ignore any that might have been missed, though unique_subs covers all)
            indices = [sub_to_idx[s] for s in subs if s in sub_to_idx]

            if not indices:
                profile_features.append(np.zeros(len(config.SEMANTIC_ANCHORS)))
                continue

            # Average similarity to each anchor
            user_sims = sim_matrix[indices]  # [N_user_subs, N_anchors]
            avg_sims = np.mean(user_sims, axis=0)  # [N_anchors]
            profile_features.append(avg_sims)

        # Create DataFrame
        col_names = [f"action_profile_{i}" for i in range(len(config.SEMANTIC_ANCHORS))]
        profile_df = pd.DataFrame(profile_features, columns=col_names, index=df.index)

        # Save to cache
        print(f"Saving Action Profiles to {cache_file}...")
        profile_df.to_parquet(cache_file)

        return profile_df
