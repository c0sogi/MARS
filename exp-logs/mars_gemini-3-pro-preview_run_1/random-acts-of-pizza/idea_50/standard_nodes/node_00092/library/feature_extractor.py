import os
import ast
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from library.config import Config
from library.text_engine import TextProcessor


class FeatureEngineer:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.scaler_mlp = StandardScaler()
        self.scaler_rf = MinMaxScaler()

        # Numerical columns to use
        self.meta_cols = [
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

    def _parse_subreddits(self, df):
        """Parses the stringified list of subreddits."""

        def safe_eval(x):
            try:
                if pd.isna(x):
                    return []
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return []

        return df["requester_subreddits_at_request"].apply(safe_eval).tolist()

    def _compute_ratios(self, df):
        """Computes derived ratio features."""
        # Upvote ratio
        up = df["requester_upvotes_plus_downvotes_at_request"]
        diff = df["requester_upvotes_minus_downvotes_at_request"]
        # up = pos + neg, diff = pos - neg => 2*pos = up + diff => pos = (up+diff)/2
        # ratio = pos / (up + epsilon)
        # Actually simpler: (up + diff) / 2 is total upvotes.
        # But standard ratio is usually upvotes / (upvotes + downvotes) = upvotes / total_votes
        # total_votes = up
        # upvotes = (up + diff) / 2

        # Avoid division by zero
        total_votes = up.replace(0, 1)
        upvotes = (up + diff) / 2
        ratio = upvotes / total_votes

        # Interaction: Activity Ratio (RAOP vs Total)
        total_posts = df["requester_number_of_posts_at_request"].replace(0, 1)
        raop_posts = df["requester_number_of_posts_on_raop_at_request"]
        activity_ratio = raop_posts / total_posts

        return pd.DataFrame(
            {"upvote_ratio": ratio, "raop_activity_ratio": activity_ratio}
        )

    def _get_subreddit_features(self, train_subs, val_subs, test_subs):
        """
        Generates Top-K binary features, History Centroids, and History Sequences.
        """
        # 1. Identify Top-K Subreddits from Train
        from collections import Counter

        all_train_subs = [sub for user_list in train_subs for sub in user_list]
        counts = Counter(all_train_subs)
        top_k = [sub for sub, _ in counts.most_common(Config.TOP_K_SUBREDDITS)]
        top_k_map = {sub: i for i, sub in enumerate(top_k)}

        # 2. Embed Unique Subreddits
        unique_subs = set(all_train_subs)
        for subs in val_subs + test_subs:
            unique_subs.update(subs)
        unique_subs_list = sorted(list(unique_subs))

        # Use TextProcessor to embed subreddits (treat names as text)
        # We use a specific cache name for this
        sub_embeddings = self.text_processor.get_sbert_embeddings(
            unique_subs_list, cache_name="unique_subreddits"
        )
        sub_emb_map = {sub: emb for sub, emb in zip(unique_subs_list, sub_embeddings)}

        # Helper to process a single split
        def process_split(subs_list):
            n = len(subs_list)
            # Binary Top-K
            binary_matrix = np.zeros((n, Config.TOP_K_SUBREDDITS), dtype=np.float32)

            # Centroids & Sequences
            centroids = np.zeros((n, Config.TEXT_EMBED_DIM), dtype=np.float32)
            # Max sequence length for history (e.g., 20)
            max_seq_len = 20
            sequences = np.zeros(
                (n, max_seq_len, Config.TEXT_EMBED_DIM), dtype=np.float32
            )
            masks = np.zeros(
                (n, max_seq_len), dtype=np.float32
            )  # 1 for valid, 0 for pad

            for i, user_subs in enumerate(subs_list):
                # Binary
                for sub in user_subs:
                    if sub in top_k_map:
                        binary_matrix[i, top_k_map[sub]] = 1.0

                # Embeddings
                user_embs = [
                    sub_emb_map[sub] for sub in user_subs if sub in sub_emb_map
                ]

                if user_embs:
                    # Centroid
                    centroids[i] = np.mean(user_embs, axis=0)

                    # Sequence (truncate or pad)
                    seq_len = min(len(user_embs), max_seq_len)
                    sequences[i, :seq_len, :] = np.array(user_embs[:seq_len])
                    masks[i, :seq_len] = 1.0
                else:
                    # Default: Zero vector (already init)
                    pass

            return binary_matrix, centroids, sequences, masks

        train_feats = process_split(train_subs)
        val_feats = process_split(val_subs)
        test_feats = process_split(test_subs)

        return train_feats, val_feats, test_feats

    def fit_transform(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main pipeline to generate features for RF and MLP.
        """
        # Cache paths
        cache_dir = Config.WORKING_DIR
        rf_train_path = os.path.join(cache_dir, "rf_features_train.npz")
        rf_val_path = os.path.join(cache_dir, "rf_features_val.npz")
        rf_test_path = os.path.join(cache_dir, "rf_features_test.npz")

        mlp_train_path = os.path.join(cache_dir, "mlp_features_train.npz")
        mlp_val_path = os.path.join(cache_dir, "mlp_features_val.npz")
        mlp_test_path = os.path.join(cache_dir, "mlp_features_test.npz")

        # Check cache
        if load_cached_data:
            if all(
                os.path.exists(p)
                for p in [
                    rf_train_path,
                    rf_val_path,
                    rf_test_path,
                    mlp_train_path,
                    mlp_val_path,
                    mlp_test_path,
                ]
            ):
                # Load sparse matrices to check dimensions
                rf_train = sparse.load_npz(rf_train_path)
                rf_val = sparse.load_npz(rf_val_path)
                rf_test = sparse.load_npz(rf_test_path)

                # Validate dimensions against current dataframes
                if (
                    rf_train.shape[0] == len(train_df)
                    and rf_val.shape[0] == len(val_df)
                    and rf_test.shape[0] == len(test_df)
                ):
                    print("Loading features from cache...")

                    def load_mlp(path):
                        data = np.load(path)
                        return {
                            k: torch.tensor(data[k], dtype=torch.float32)
                            for k in data.files
                        }

                    return (rf_train, rf_val, rf_test), (
                        load_mlp(mlp_train_path),
                        load_mlp(mlp_val_path),
                        load_mlp(mlp_test_path),
                    )
                else:
                    print(
                        f"Cache dimension mismatch (Train: {rf_train.shape[0]} vs {len(train_df)}). Regenerating features..."
                    )
                    # Cite debug_lesson_7: Propagate validation failure to disable loading of potentially stale intermediate caches
                    load_cached_data = False

        print("Generating features from scratch...")

        # 1. Text Features (SBERT & TF-IDF & VADER)
        # SBERT
        train_title_emb = self.text_processor.get_sbert_embeddings(
            train_df["request_title"], "train_title", load_cached_data
        )
        val_title_emb = self.text_processor.get_sbert_embeddings(
            val_df["request_title"], "val_title", load_cached_data
        )
        test_title_emb = self.text_processor.get_sbert_embeddings(
            test_df["request_title"], "test_title", load_cached_data
        )

        train_body_emb = self.text_processor.get_sbert_embeddings(
            train_df["request_text_edit_aware"], "train_body", load_cached_data
        )
        val_body_emb = self.text_processor.get_sbert_embeddings(
            val_df["request_text_edit_aware"], "val_body", load_cached_data
        )
        test_body_emb = self.text_processor.get_sbert_embeddings(
            test_df["request_text_edit_aware"], "test_body", load_cached_data
        )

        # TF-IDF (Concatenate Title + Body)
        train_text = (
            train_df["request_title"].fillna("")
            + " "
            + train_df["request_text_edit_aware"].fillna("")
        )
        val_text = (
            val_df["request_title"].fillna("")
            + " "
            + val_df["request_text_edit_aware"].fillna("")
        )
        test_text = (
            test_df["request_title"].fillna("")
            + " "
            + test_df["request_text_edit_aware"].fillna("")
        )

        train_tfidf, val_tfidf, test_tfidf = self.text_processor.get_tfidf_features(
            train_text, val_text, test_text, load_cached_data
        )

        # VADER
        train_vader_title = self.text_processor.get_vader_sentiment(
            train_df["request_title"], "train_title", load_cached_data
        )
        val_vader_title = self.text_processor.get_vader_sentiment(
            val_df["request_title"], "val_title", load_cached_data
        )
        test_vader_title = self.text_processor.get_vader_sentiment(
            test_df["request_title"], "test_title", load_cached_data
        )

        train_vader_body = self.text_processor.get_vader_sentiment(
            train_df["request_text_edit_aware"], "train_body", load_cached_data
        )
        val_vader_body = self.text_processor.get_vader_sentiment(
            val_df["request_text_edit_aware"], "val_body", load_cached_data
        )
        test_vader_body = self.text_processor.get_vader_sentiment(
            test_df["request_text_edit_aware"], "test_body", load_cached_data
        )

        # 2. Metadata Processing
        # Extract and Impute
        def process_meta(df):
            meta = df[self.meta_cols].copy()
            # Simple median imputation
            for col in self.meta_cols:
                meta[col] = meta[col].fillna(meta[col].median())

            # Ratios
            ratios = self._compute_ratios(df)
            meta = pd.concat([meta, ratios], axis=1)
            return meta.fillna(0)

        train_meta_raw = process_meta(train_df)
        val_meta_raw = process_meta(val_df)
        test_meta_raw = process_meta(test_df)

        # Scaling for RF (MinMax)
        self.scaler_rf.fit(train_meta_raw)
        train_meta_rf = self.scaler_rf.transform(train_meta_raw)
        val_meta_rf = self.scaler_rf.transform(val_meta_raw)
        test_meta_rf = self.scaler_rf.transform(test_meta_raw)

        # Scaling for MLP (Arcsinh + Standard)
        # Apply arcsinh first to handle skew
        train_meta_arcsinh = np.arcsinh(train_meta_raw)
        val_meta_arcsinh = np.arcsinh(val_meta_raw)
        test_meta_arcsinh = np.arcsinh(test_meta_raw)

        self.scaler_mlp.fit(train_meta_arcsinh)
        train_meta_mlp = self.scaler_mlp.transform(train_meta_arcsinh)
        val_meta_mlp = self.scaler_mlp.transform(val_meta_arcsinh)
        test_meta_mlp = self.scaler_mlp.transform(test_meta_arcsinh)

        # 3. Subreddit Features
        train_subs = self._parse_subreddits(train_df)
        val_subs = self._parse_subreddits(val_df)
        test_subs = self._parse_subreddits(test_df)

        (
            (train_topk, train_centroid, train_seq, train_mask),
            (val_topk, val_centroid, val_seq, val_mask),
            (test_topk, test_centroid, test_seq, test_mask),
        ) = self._get_subreddit_features(train_subs, val_subs, test_subs)

        # 4. Coherence & Dissonance Features
        def compute_coherence(title_emb, body_emb, centroid, meta_df):
            # Cosine Similarities
            # Reshape for pairwise (N, D)
            sim_title_hist = np.sum(
                title_emb * centroid, axis=1
            )  # Normalized in TextProcessor, so dot product is cosine
            sim_body_hist = np.sum(body_emb * centroid, axis=1)
            sim_title_body = np.sum(title_emb * body_emb, axis=1)

            # Dissonance
            dissonance = np.abs(sim_title_hist - sim_body_hist)

            # Interaction Terms
            # I1 = Dissonance * log(1 + Account Age)
            acc_age = meta_df["requester_account_age_in_days_at_request"].values
            i1 = dissonance * np.log1p(acc_age)

            # I2 = Internal Consistency * Upvote Ratio
            up_ratio = meta_df["upvote_ratio"].values
            i2 = sim_title_body * up_ratio

            return np.stack(
                [sim_title_hist, sim_body_hist, sim_title_body, dissonance, i1, i2],
                axis=1,
            )

        train_coherence = compute_coherence(
            train_title_emb, train_body_emb, train_centroid, train_meta_raw
        )
        val_coherence = compute_coherence(
            val_title_emb, val_body_emb, val_centroid, val_meta_raw
        )
        test_coherence = compute_coherence(
            test_title_emb, test_body_emb, test_centroid, test_meta_raw
        )

        # 5. Assemble RF Features
        # Concat: TFIDF + Meta(RF) + TopK + Coherence
        def assemble_rf(tfidf, meta, topk, coherence):
            # Convert dense arrays to sparse for efficient hstack
            dense_feats = np.hstack([meta, topk, coherence])
            sparse_dense = sparse.csr_matrix(dense_feats)
            return sparse.hstack([tfidf, sparse_dense])

        rf_train_final = assemble_rf(
            train_tfidf, train_meta_rf, train_topk, train_coherence
        )
        rf_val_final = assemble_rf(val_tfidf, val_meta_rf, val_topk, val_coherence)
        rf_test_final = assemble_rf(test_tfidf, test_meta_rf, test_topk, test_coherence)

        # 6. Assemble MLP Features
        # Control Features: Meta(MLP) + TopK + Vader(Title) + Vader(Body)
        def assemble_mlp_control(meta, topk, vader_t, vader_b):
            return np.hstack([meta, topk, vader_t, vader_b])

        train_control = assemble_mlp_control(
            train_meta_mlp, train_topk, train_vader_title, train_vader_body
        )
        val_control = assemble_mlp_control(
            val_meta_mlp, val_topk, val_vader_title, val_vader_body
        )
        test_control = assemble_mlp_control(
            test_meta_mlp, test_topk, test_vader_title, test_vader_body
        )

        mlp_train_final = {
            "title_emb": train_title_emb,
            "body_emb": train_body_emb,
            "history_emb": train_seq,
            "history_mask": train_mask,
            "global_centroid": train_centroid,
            "control_features": train_control,
        }

        mlp_val_final = {
            "title_emb": val_title_emb,
            "body_emb": val_body_emb,
            "history_emb": val_seq,
            "history_mask": val_mask,
            "global_centroid": val_centroid,
            "control_features": val_control,
        }

        mlp_test_final = {
            "title_emb": test_title_emb,
            "body_emb": test_body_emb,
            "history_emb": test_seq,
            "history_mask": test_mask,
            "global_centroid": test_centroid,
            "control_features": test_control,
        }

        # Save to Cache
        print("Saving features to cache...")
        sparse.save_npz(rf_train_path, rf_train_final)
        sparse.save_npz(rf_val_path, rf_val_final)
        sparse.save_npz(rf_test_path, rf_test_final)

        np.savez(mlp_train_path, **mlp_train_final)
        np.savez(mlp_val_path, **mlp_val_final)
        np.savez(mlp_test_path, **mlp_test_final)

        # Convert MLP dicts to tensors for return
        def to_tensors(d):
            return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}

        return (rf_train_final, rf_val_final, rf_test_final), (
            to_tensors(mlp_train_final),
            to_tensors(mlp_val_final),
            to_tensors(mlp_test_final),
        )
