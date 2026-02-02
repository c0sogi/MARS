import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from library.config import Config
from library.utils import save_numpy, load_numpy, seed_everything, print_metric
from library.data_loader import load_data


class FeatureEngineer:
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.rf_cache_path = os.path.join(self.cache_dir, "rf_features.npz")
        self.mlp_cache_path = os.path.join(self.cache_dir, "mlp_features.npz")

        # SBERT model is loaded only when needed to save resources if cached
        self.sbert_model = None

    def _load_sbert(self):
        if self.sbert_model is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
            # Use CPU or GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=device
            )

    def _encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._load_sbert()
        # Ensure texts are strings
        texts = [str(t) if pd.notnull(t) else "" for t in texts]
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity later
        )
        return embeddings

    def _process_subreddits(
        self, train_subs: pd.Series, val_subs: pd.Series, test_subs: pd.Series
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Encodes all unique subreddits found in the dataset.
        Returns a dictionary mapping subreddit name to embedding.
        """
        self._load_sbert()

        # Collect all unique subreddits
        all_subs = set()
        for series in [train_subs, val_subs, test_subs]:
            for sub_list in series:
                if isinstance(sub_list, list):
                    all_subs.update(sub_list)

        unique_subs_list = sorted(list(all_subs))
        print(f"Encoding {len(unique_subs_list)} unique subreddits...")

        # Encode in batches
        sub_embeddings = self._encode_text(unique_subs_list, batch_size=256)

        sub_map = {sub: emb for sub, emb in zip(unique_subs_list, sub_embeddings)}
        return sub_map, unique_subs_list

    def _compute_history_features(
        self, df: pd.DataFrame, sub_map: Dict[str, np.ndarray], max_seq_len: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes history centroid and sequence embeddings for MLP.
        Returns:
            centroids: (N, 384)
            sequences: (N, max_seq_len, 384)
            masks: (N, max_seq_len) - 1 for real, 0 for padding
        """
        n_samples = len(df)
        emb_dim = Config.EMBEDDING_DIM

        centroids = np.zeros((n_samples, emb_dim), dtype=np.float32)
        sequences = np.zeros((n_samples, max_seq_len, emb_dim), dtype=np.float32)
        masks = np.zeros((n_samples, max_seq_len), dtype=np.float32)

        sub_col = Config.SUBREDDIT_LIST_COL

        for i, sub_list in enumerate(df[sub_col]):
            if not isinstance(sub_list, list) or len(sub_list) == 0:
                continue

            # Get embeddings for user's subreddits
            # We take up to max_seq_len (assuming recent ones are relevant, or just first N)
            # The list order in JSON is typically arbitrary or frequency based, we take first N.
            current_embs = []
            for sub in sub_list:
                if sub in sub_map:
                    current_embs.append(sub_map[sub])

            if not current_embs:
                continue

            # Centroid
            current_embs_arr = np.array(current_embs)
            centroids[i] = np.mean(current_embs_arr, axis=0)

            # Sequence & Mask
            n_items = min(len(current_embs), max_seq_len)
            sequences[i, :n_items, :] = current_embs_arr[:n_items]
            masks[i, :n_items] = 1.0

        return centroids, sequences, masks

    def _compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Computes row-wise cosine similarity between two arrays of shape (N, D).
        Assumes vectors are already normalized (SBERT encode does this with normalize_embeddings=True).
        """
        # Dot product of normalized vectors is cosine similarity
        return np.sum(a * b, axis=1)

    def process_data(
        self, load_cached_data: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main execution method.
        """
        seed_everything()

        # 1. Check Cache
        if load_cached_data:
            if os.path.exists(self.rf_cache_path) and os.path.exists(
                self.mlp_cache_path
            ):
                print("Loading engineered features from cache...")
                rf_data = load_numpy(self.rf_cache_path)
                mlp_data = load_numpy(self.mlp_cache_path)
                # Convert NpzFile to dict for easier handling if needed, or return as is
                return dict(rf_data), dict(mlp_data)
            else:
                print("Feature cache not found. Generating features...")
        else:
            print("Ignoring cache. Generating features...")

        # 2. Load Cleaned Data
        train_df, val_df, test_df = load_data(load_cached_data=True)

        # Prepare Target
        y_train = train_df[Config.TARGET_COL].astype(int).values
        y_val = val_df[Config.TARGET_COL].astype(int).values
        # Test has no target for prediction, but we handle structure consistently

        # ==========================================
        # SBERT Embeddings & Consistency (Shared)
        # ==========================================
        print("Generating SBERT embeddings...")

        # Title & Body
        train_title_emb = self._encode_text(train_df[Config.TEXT_TITLE_COL].tolist())
        val_title_emb = self._encode_text(val_df[Config.TEXT_TITLE_COL].tolist())
        test_title_emb = self._encode_text(test_df[Config.TEXT_TITLE_COL].tolist())

        train_body_emb = self._encode_text(train_df[Config.TEXT_BODY_COL].tolist())
        val_body_emb = self._encode_text(val_df[Config.TEXT_BODY_COL].tolist())
        test_body_emb = self._encode_text(test_df[Config.TEXT_BODY_COL].tolist())

        # Subreddits / History
        sub_map, _ = self._process_subreddits(
            train_df[Config.SUBREDDIT_LIST_COL],
            val_df[Config.SUBREDDIT_LIST_COL],
            test_df[Config.SUBREDDIT_LIST_COL],
        )

        print("Computing history features...")
        train_cent, train_seq, train_mask = self._compute_history_features(
            train_df, sub_map
        )
        val_cent, val_seq, val_mask = self._compute_history_features(val_df, sub_map)
        test_cent, test_seq, test_mask = self._compute_history_features(
            test_df, sub_map
        )

        # Triple-View Consistency Scalars
        print("Computing consistency scalars...")
        # 1. Internal: Title vs Body
        train_cons_tb = self._compute_cosine_similarity(train_title_emb, train_body_emb)
        val_cons_tb = self._compute_cosine_similarity(val_title_emb, val_body_emb)
        test_cons_tb = self._compute_cosine_similarity(test_title_emb, test_body_emb)

        # 2. External: Title vs History
        train_cons_th = self._compute_cosine_similarity(train_title_emb, train_cent)
        val_cons_th = self._compute_cosine_similarity(val_title_emb, val_cent)
        test_cons_th = self._compute_cosine_similarity(test_title_emb, test_cent)

        # 3. External: Body vs History
        train_cons_bh = self._compute_cosine_similarity(train_body_emb, train_cent)
        val_cons_bh = self._compute_cosine_similarity(val_body_emb, val_cent)
        test_cons_bh = self._compute_cosine_similarity(test_body_emb, test_cent)

        # Stack consistency features (N, 3)
        train_consistency = np.stack(
            [train_cons_tb, train_cons_th, train_cons_bh], axis=1
        )
        val_consistency = np.stack([val_cons_tb, val_cons_th, val_cons_bh], axis=1)
        test_consistency = np.stack([test_cons_tb, test_cons_th, test_cons_bh], axis=1)

        # ==========================================
        # Metadata & Top-K (Shared)
        # ==========================================
        print("Processing metadata and Top-K...")

        # 1. Top-K Subreddits
        # Count frequency in Train
        all_train_subs = [
            s for sub_list in train_df[Config.SUBREDDIT_LIST_COL] for s in sub_list
        ]
        top_k_subs = (
            pd.Series(all_train_subs)
            .value_counts()
            .head(Config.TOP_K_SUBREDDITS)
            .index.tolist()
        )

        def get_top_k_features(df):
            matrix = np.zeros((len(df), len(top_k_subs)), dtype=np.float32)
            for i, sub_list in enumerate(df[Config.SUBREDDIT_LIST_COL]):
                if isinstance(sub_list, list):
                    s_set = set(sub_list)
                    for j, sub in enumerate(top_k_subs):
                        if sub in s_set:
                            matrix[i, j] = 1.0
            return matrix

        train_topk = get_top_k_features(train_df)
        val_topk = get_top_k_features(val_df)
        test_topk = get_top_k_features(test_df)

        # 2. Numerical Metadata
        # Arcsinh transform
        num_cols = Config.NUMERIC_COLS
        train_meta_raw = np.arcsinh(train_df[num_cols].fillna(0).values)
        val_meta_raw = np.arcsinh(val_df[num_cols].fillna(0).values)
        test_meta_raw = np.arcsinh(test_df[num_cols].fillna(0).values)

        # Scale
        scaler = StandardScaler()
        train_meta_scaled = scaler.fit_transform(train_meta_raw)
        val_meta_scaled = scaler.transform(val_meta_raw)
        test_meta_scaled = scaler.transform(test_meta_raw)

        # ==========================================
        # Stream A: Random Forest Assembly
        # ==========================================
        print("Assembling Random Forest features...")

        # 1. TF-IDF
        # Combine Title + Body
        train_text = (
            train_df[Config.TEXT_TITLE_COL].fillna("")
            + " "
            + train_df[Config.TEXT_BODY_COL].fillna("")
        )
        val_text = (
            val_df[Config.TEXT_TITLE_COL].fillna("")
            + " "
            + val_df[Config.TEXT_BODY_COL].fillna("")
        )
        test_text = (
            test_df[Config.TEXT_TITLE_COL].fillna("")
            + " "
            + test_df[Config.TEXT_BODY_COL].fillna("")
        )

        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", sublinear_tf=True
        )
        train_tfidf = tfidf.fit_transform(train_text).toarray().astype(np.float32)
        val_tfidf = tfidf.transform(val_text).toarray().astype(np.float32)
        test_tfidf = tfidf.transform(test_text).toarray().astype(np.float32)

        # 2. Interaction Features (Credibility x Consistency)
        # Credibility Proxies:
        # C1: Log Account Age (already in scaled metadata, index 0)
        # C2: Upvote Ratio (Derived)
        # C3: Log Num Posts (already in scaled metadata, index 4)

        # Helper to get raw values for ratio calculation to avoid negative scaled values issues in logic if any
        # But we can use the scaled values for interaction multiplication.

        # Let's compute explicit Upvote Ratio: (Up - Down) / (Up + Down + 1)
        def get_upvote_ratio(df):
            up_minus = df["requester_upvotes_minus_downvotes_at_request"].fillna(0)
            up_plus = df["requester_upvotes_plus_downvotes_at_request"].fillna(0)
            # ratio roughly: (up - down) / (up + down)
            # Avoid div by zero
            return (up_minus / (up_plus + 1.0)).values.reshape(-1, 1)

        train_ur = get_upvote_ratio(train_df)
        val_ur = get_upvote_ratio(val_df)
        test_ur = get_upvote_ratio(test_df)

        # Extract scaled credibility metrics
        # Indices in Config.NUMERIC_COLS:
        # 0: requester_account_age_in_days_at_request
        # 4: requester_number_of_posts_at_request

        def make_interactions(meta_scaled, upvote_ratio, consistency):
            c1 = meta_scaled[:, 0:1]  # Account Age
            c2 = upvote_ratio  # Upvote Ratio
            c3 = meta_scaled[:, 4:5]  # Num Posts

            credibility = np.hstack([c1, c2, c3])  # (N, 3)

            # Cross product: 3 Cred * 3 Cons = 9 features
            interactions = []
            for i in range(3):  # Cred
                for j in range(3):  # Cons
                    interactions.append(credibility[:, i] * consistency[:, j])

            return np.stack(interactions, axis=1)

        train_inter = make_interactions(train_meta_scaled, train_ur, train_consistency)
        val_inter = make_interactions(val_meta_scaled, val_ur, val_consistency)
        test_inter = make_interactions(test_meta_scaled, test_ur, test_consistency)

        # Concatenate RF Features
        # [TF-IDF (5000), Metadata (9), Top-K (50), Consistency (3), Interactions (9)]
        X_train_rf = np.hstack(
            [train_tfidf, train_meta_scaled, train_topk, train_consistency, train_inter]
        )
        X_val_rf = np.hstack(
            [val_tfidf, val_meta_scaled, val_topk, val_consistency, val_inter]
        )
        X_test_rf = np.hstack(
            [test_tfidf, test_meta_scaled, test_topk, test_consistency, test_inter]
        )

        rf_data = {
            "X_train": X_train_rf,
            "y_train": y_train,
            "X_val": X_val_rf,
            "y_val": y_val,
            "X_test": X_test_rf,
        }

        # ==========================================
        # Stream B: MLP Assembly
        # ==========================================
        print("Assembling MLP features...")

        # MLP needs structured inputs. We'll store them in dicts.
        # Metadata for MLP includes: Scaled Numeric + Top-K + Consistency Scalars
        # We concatenate these into a single dense vector for the "Metadata Branch"

        train_mlp_meta = np.hstack([train_meta_scaled, train_topk, train_consistency])
        val_mlp_meta = np.hstack([val_meta_scaled, val_topk, val_consistency])
        test_mlp_meta = np.hstack([test_meta_scaled, test_topk, test_consistency])

        mlp_data = {
            "train": {
                "title_emb": train_title_emb,
                "body_emb": train_body_emb,
                "history_seq": train_seq,
                "history_mask": train_mask,
                "history_cent": train_cent,
                "metadata": train_mlp_meta,
                "y": y_train,
            },
            "val": {
                "title_emb": val_title_emb,
                "body_emb": val_body_emb,
                "history_seq": val_seq,
                "history_mask": val_mask,
                "history_cent": val_cent,
                "metadata": val_mlp_meta,
                "y": y_val,
            },
            "test": {
                "title_emb": test_title_emb,
                "body_emb": test_body_emb,
                "history_seq": test_seq,
                "history_mask": test_mask,
                "history_cent": test_cent,
                "metadata": test_mlp_meta,
            },
        }

        # 3. Save to Cache
        print("Saving features to cache...")
        save_numpy(rf_data, self.rf_cache_path)
        save_numpy(mlp_data, self.mlp_cache_path)

        return rf_data, mlp_data
