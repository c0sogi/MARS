import os
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library import config, utils, features_text


class HistoryProcessor:
    def __init__(
        self,
        max_seq_len=50,
        embedding_dim=config.EMBEDDING_DIM,
        model_name=config.SBERT_MODEL_NAME,
    ):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _parse_subreddits(self, series):
        parsed = []
        for x in series:
            if isinstance(x, list):
                parsed.append(x)
            elif isinstance(x, str):
                try:
                    p = ast.literal_eval(x)
                    parsed.append(p if isinstance(p, list) else [])
                except:
                    parsed.append([])
            else:
                parsed.append([])
        return parsed

    def process_sequences(self, train_subs, val_subs, test_subs):
        """
        Generates padded sequences of embeddings for user history.
        Returns: (train_seq, val_seq, test_seq) as (N, T, D) arrays.
        """
        self._load_model()

        # Parse all lists
        train_parsed = self._parse_subreddits(train_subs)
        test_parsed = self._parse_subreddits(test_subs)
        val_parsed = self._parse_subreddits(val_subs) if val_subs is not None else []

        # Identify unique subreddits to embed efficiently
        all_subs = set()
        for seq in train_parsed + test_parsed + val_parsed:
            all_subs.update(seq)

        unique_subs = sorted(list(all_subs))
        if not unique_subs:
            # Handle edge case of no subreddits
            empty_seq = np.zeros(
                (1, self.max_seq_len, self.embedding_dim), dtype=np.float32
            )
            return (
                np.zeros(
                    (len(train_subs), self.max_seq_len, self.embedding_dim),
                    dtype=np.float32,
                ),
                np.zeros(
                    (
                        len(val_subs) if val_subs is not None else 0,
                        self.max_seq_len,
                        self.embedding_dim,
                    ),
                    dtype=np.float32,
                ),
                np.zeros(
                    (len(test_subs), self.max_seq_len, self.embedding_dim),
                    dtype=np.float32,
                ),
            )

        # Embed unique subreddits
        # Batch size can be large for simple strings
        sub_embeddings = self.model.encode(
            unique_subs, show_progress_bar=False, convert_to_numpy=True, batch_size=128
        )
        sub_map = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}

        # Helper to construct padded array
        def build_array(parsed_list):
            N = len(parsed_list)
            out = np.zeros((N, self.max_seq_len, self.embedding_dim), dtype=np.float32)
            for i, seq in enumerate(parsed_list):
                # Truncate if too long, take most recent (assuming list is chronological or order doesn't matter much)
                # We'll take the first K as provided
                curr_seq = seq[: self.max_seq_len]
                for t, sub in enumerate(curr_seq):
                    if sub in sub_map:
                        out[i, t, :] = sub_map[sub]
            return out

        train_seq = build_array(train_parsed)
        test_seq = build_array(test_parsed)
        val_seq = build_array(val_parsed) if val_subs is not None else None

        return train_seq, val_seq, test_seq


class ConsistencyCalculator:
    def compute(self, title_emb, body_emb, hist_centroid):
        """
        Computes cosine similarity between request (title/body) and history centroid.
        """

        # Normalize for cosine similarity
        def normalize(x):
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            # Avoid division by zero
            norm[norm == 0] = 1e-10
            return x / norm

        title_norm = normalize(title_emb)
        body_norm = normalize(body_emb)
        hist_norm = normalize(hist_centroid)

        # Dot product
        # (N, D) * (N, D) -> (N,)
        title_sim = np.sum(title_norm * hist_norm, axis=1, keepdims=True)
        body_sim = np.sum(body_norm * hist_norm, axis=1, keepdims=True)

        return title_sim.astype(np.float32), body_sim.astype(np.float32)


class TopKSubredditEncoder:
    def __init__(self, k=config.TOP_K_SUBREDDITS):
        self.k = k
        self.top_subs = []

    def fit(self, subreddits_series):
        """
        Identifies top K subreddits from the training series.
        """
        counter = Counter()
        for x in subreddits_series:
            # Handle stringified lists
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except:
                    x = []
            if isinstance(x, list):
                counter.update(x)

        self.top_subs = [sub for sub, count in counter.most_common(self.k)]
        return self

    def transform(self, subreddits_series):
        """
        Returns binary matrix (N, K).
        """
        N = len(subreddits_series)
        out = np.zeros((N, self.k), dtype=np.float32)

        for i, x in enumerate(subreddits_series):
            if isinstance(x, str):
                try:
                    x = ast.literal_eval(x)
                except:
                    x = []
            if not isinstance(x, list):
                x = []

            x_set = set(x)
            for j, sub in enumerate(self.top_subs):
                if sub in x_set:
                    out[i, j] = 1.0
        return out


class MetadataPreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.selected_cols = []

    def engineer_features(self, df):
        """
        Creates ratio features and selects numerical columns.
        """
        df_eng = df.copy()

        # Ratios
        # Upvotes / (Up + Down)
        total_votes = df_eng.get("requester_upvotes_plus_downvotes_at_request", 0)
        up_votes = (
            total_votes + df_eng.get("requester_upvotes_minus_downvotes_at_request", 0)
        ) / 2
        # Avoid div by zero
        df_eng["meta_upvote_ratio"] = np.where(
            total_votes > 0, up_votes / total_votes, 0.5
        )

        # Comments / Posts
        n_posts = df_eng.get("requester_number_of_posts_at_request", 0)
        n_comments = df_eng.get("requester_number_of_comments_at_request", 0)
        df_eng["meta_interaction_ratio"] = np.where(
            n_posts > 0, n_comments / n_posts, 0
        )

        # RAOP Activity
        raop_posts = df_eng.get("requester_number_of_posts_on_raop_at_request", 0)
        df_eng["meta_raop_ratio"] = np.where(n_posts > 0, raop_posts / n_posts, 0)

        return df_eng

    def fit_transform(self, train_df, val_df, test_df):
        """
        Process metadata for both RF (Raw/Ratios) and MLP (Arcsinh/Scaled).
        """
        # 1. Engineer Features
        train_eng = self.engineer_features(train_df)
        test_eng = self.engineer_features(test_df)
        val_eng = self.engineer_features(val_df) if val_df is not None else None

        # 2. Identify Intersection Columns (exclude leakage)
        # We rely on utils.get_feature_intersection which checks train vs test columns
        # However, we must ensure we only pick numeric ones.
        common_cols = utils.get_feature_intersection(train_eng, test_eng)

        # Filter for numeric types only
        numeric_cols = (
            train_eng[common_cols].select_dtypes(include=[np.number]).columns.tolist()
        )

        # Exclude ID or specific non-feature columns if any remain
        exclude = [
            "requester_received_pizza",
            "unix_timestamp_of_request",
            "unix_timestamp_of_request_utc",
        ]
        self.selected_cols = [c for c in numeric_cols if c not in exclude]

        # 3. Extract Raw Data
        X_train_raw = train_eng[self.selected_cols].values.astype(np.float32)
        X_test_raw = test_eng[self.selected_cols].values.astype(np.float32)
        X_val_raw = (
            val_eng[self.selected_cols].values.astype(np.float32)
            if val_eng is not None
            else None
        )

        # 4. Impute
        self.imputer.fit(X_train_raw)
        X_train_imp = self.imputer.transform(X_train_raw)
        X_test_imp = self.imputer.transform(X_test_raw)
        X_val_imp = self.imputer.transform(X_val_raw) if X_val_raw is not None else None

        # 5. Create MLP Version (Arcsinh + Scale)
        X_train_mlp = utils.arcsinh_transform(X_train_imp)
        X_test_mlp = utils.arcsinh_transform(X_test_imp)
        X_val_mlp = (
            utils.arcsinh_transform(X_val_imp) if X_val_imp is not None else None
        )

        self.scaler.fit(X_train_mlp)
        X_train_mlp = self.scaler.transform(X_train_mlp)
        X_test_mlp = self.scaler.transform(X_test_mlp)
        X_val_mlp = self.scaler.transform(X_val_mlp) if X_val_mlp is not None else None

        return (X_train_imp, X_val_imp, X_test_imp), (
            X_train_mlp,
            X_val_mlp,
            X_test_mlp,
        )


def generate_meta_features(train_df, val_df, test_df, load_cached_data=True):
    """
    Orchestrates the generation of metadata features.
    Returns a dictionary of numpy arrays.
    """
    cache_file = os.path.join(config.CACHE_DIR, "meta_features.npz")

    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading meta features from cache: {cache_file}")
        try:
            loaded = np.load(cache_file)
            return {k: v for k, v in loaded.items()}
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    print("Computing meta features from scratch...")

    # Load text features for consistency calculation
    # We assume text features are available or can be generated
    text_feats = features_text.generate_text_features(
        train_df, val_df, test_df, load_cached_data=True
    )

    # 1. History Sequences (MLP)
    print("Generating History Sequences...")
    hist_proc = HistoryProcessor()
    train_seq, val_seq, test_seq = hist_proc.process_sequences(
        train_df["requester_subreddits_at_request"],
        val_df["requester_subreddits_at_request"] if val_df is not None else None,
        test_df["requester_subreddits_at_request"],
    )

    # 2. Consistency Scalars (RF)
    print("Generating Consistency Scalars...")
    cons_calc = ConsistencyCalculator()

    train_cons_title, train_cons_body = cons_calc.compute(
        text_feats["train_title_emb"],
        text_feats["train_body_emb"],
        text_feats["train_hist_centroid"],
    )
    test_cons_title, test_cons_body = cons_calc.compute(
        text_feats["test_title_emb"],
        text_feats["test_body_emb"],
        text_feats["test_hist_centroid"],
    )

    if val_df is not None:
        val_cons_title, val_cons_body = cons_calc.compute(
            text_feats["val_title_emb"],
            text_feats["val_body_emb"],
            text_feats["val_hist_centroid"],
        )
    else:
        val_cons_title, val_cons_body = None, None

    # 3. Top-K Subreddits (RF)
    print("Generating Top-K Subreddit Flags...")
    topk_enc = TopKSubredditEncoder()
    topk_enc.fit(train_df["requester_subreddits_at_request"])

    train_topk = topk_enc.transform(train_df["requester_subreddits_at_request"])
    test_topk = topk_enc.transform(test_df["requester_subreddits_at_request"])
    val_topk = (
        topk_enc.transform(val_df["requester_subreddits_at_request"])
        if val_df is not None
        else None
    )

    # 4. Numerical Metadata (RF & MLP)
    print("Processing Numerical Metadata...")
    meta_proc = MetadataPreprocessor()
    (train_meta_rf, val_meta_rf, test_meta_rf), (
        train_meta_mlp,
        val_meta_mlp,
        test_meta_mlp,
    ) = meta_proc.fit_transform(train_df, val_df, test_df)

    # Pack results
    results = {
        # Sequence Data (MLP)
        "train_hist_seq": train_seq,
        "test_hist_seq": test_seq,
        # Consistency Scalars (RF)
        "train_cons_title": train_cons_title,
        "train_cons_body": train_cons_body,
        "test_cons_title": test_cons_title,
        "test_cons_body": test_cons_body,
        # Top-K (RF)
        "train_topk": train_topk,
        "test_topk": test_topk,
        # Metadata
        "train_meta_rf": train_meta_rf,
        "test_meta_rf": test_meta_rf,
        "train_meta_mlp": train_meta_mlp,
        "test_meta_mlp": test_meta_mlp,
    }

    if val_df is not None:
        results.update(
            {
                "val_hist_seq": val_seq,
                "val_cons_title": val_cons_title,
                "val_cons_body": val_cons_body,
                "val_topk": val_topk,
                "val_meta_rf": val_meta_rf,
                "val_meta_mlp": val_meta_mlp,
            }
        )

    # Save
    print(f"Saving meta features to {cache_file}...")
    np.savez(cache_file, **results)

    return results
