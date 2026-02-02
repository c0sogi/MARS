import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from library.config import Config
from library.utils import (
    save_npy,
    load_npy,
    save_pickle,
    load_pickle,
    set_seed,
    ensure_dir,
)


class FeatureEngine:
    """
    Orchestrates feature engineering for both Random Forest (Stream A) and MLP (Stream B).
    Handles caching, SBERT embedding, TF-IDF, and metadata transformation.
    """

    def __init__(self):
        self.sbert_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.top_k_subreddits = None
        self.subreddit_embeddings_map = {}

    def _load_sbert(self):
        if self.sbert_model is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}")
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _get_sbert_embeddings(self, texts, batch_size=Config.SBERT_BATCH_SIZE):
        """Generates embeddings for a list of texts."""
        self._load_sbert()
        # Ensure texts are strings and handle NaNs
        texts = [str(t) if pd.notnull(t) else "" for t in texts]
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def _compute_tfidf(self, df_train, df_val, df_test):
        """Computes TF-IDF features for RF."""
        print("Computing TF-IDF features...")

        # Concatenate title and body for TF-IDF
        def combine_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).tolist()

        train_text = combine_text(df_train)
        val_text = combine_text(df_val)
        test_text = combine_text(df_test)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            dtype=np.float32,
        )

        X_train = self.tfidf_vectorizer.fit_transform(train_text).toarray()
        X_val = self.tfidf_vectorizer.transform(val_text).toarray()
        X_test = self.tfidf_vectorizer.transform(test_text).toarray()

        return X_train, X_val, X_test

    def _process_metadata(self, df_train, df_val, df_test):
        """
        Processes numerical metadata: Arcsinh transform + Standardization.
        Returns processed arrays and list of feature names.
        """
        print("Processing metadata...")

        # Select numerical columns (excluding IDs, text, and target)
        exclude = [
            "request_id",
            "requester_received_pizza",
            "request_text",
            "request_title",
            "request_text_edit_aware",
            "requester_subreddits_at_request",
            "source_file",
            "giver_username_if_known",
            "requester_username",
            "requester_user_flair",
            "post_was_edited",  # bool
        ]

        num_cols = [
            c
            for c in df_train.columns
            if df_train[c].dtype in [np.float64, np.int64] and c not in exclude
        ]

        # Filter to ensure consistency with test set (Cite debug_lesson_2)
        test_cols_set = set(df_test.columns)
        num_cols = [c for c in num_cols if c in test_cols_set]

        # Extract raw data
        X_train = df_train[num_cols].fillna(0).values
        X_val = df_val[num_cols].fillna(0).values
        X_test = df_test[num_cols].fillna(0).values

        # Arcsinh transformation (handles skewness and zeros better than log)
        X_train = np.arcsinh(X_train)
        X_val = np.arcsinh(X_val)
        X_test = np.arcsinh(X_test)

        # Standardization
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test, num_cols

    def _process_subreddits_top_k(self, df_train, df_val, df_test):
        """
        Selects top K subreddits based on Mutual Information and creates binary indicators.
        """
        print(
            f"Selecting Top-{Config.TOP_K_SUBREDDIT_INDICATORS} predictive subreddits..."
        )

        # Convert list of subreddits to space-separated string for CountVectorizer
        def to_string(sub_list):
            return " ".join(sub_list) if isinstance(sub_list, list) else ""

        train_subs = df_train["requester_subreddits_at_request"].apply(to_string)
        val_subs = df_val["requester_subreddits_at_request"].apply(to_string)
        test_subs = df_test["requester_subreddits_at_request"].apply(to_string)

        # Initial Vectorizer to get counts
        cv = CountVectorizer(min_df=5, binary=True)
        X_train_counts = cv.fit_transform(train_subs)

        # Compute Mutual Information
        y_train = df_train["requester_received_pizza"].astype(int).values
        mi_scores = mutual_info_classif(
            X_train_counts,
            y_train,
            discrete_features=True,
            random_state=Config.RANDOM_SEED,
        )

        # Select Top K
        top_indices = np.argsort(mi_scores)[-Config.TOP_K_SUBREDDIT_INDICATORS :]
        vocab = np.array(cv.get_feature_names_out())
        self.top_k_subreddits = vocab[top_indices]

        # Create final binary features for selected subreddits
        # We can use a new CountVectorizer with fixed vocabulary
        cv_final = CountVectorizer(vocabulary=self.top_k_subreddits, binary=True)

        X_train = cv_final.fit_transform(train_subs).toarray()
        X_val = cv_final.transform(val_subs).toarray()
        X_test = cv_final.transform(test_subs).toarray()

        return X_train, X_val, X_test

    def _embed_all_unique_subreddits(self, df_list):
        """
        Identifies all unique subreddits across datasets and computes their SBERT embeddings.
        """
        print("Embedding unique subreddits...")
        unique_subs = set()
        for df in df_list:
            for subs in df["requester_subreddits_at_request"]:
                if isinstance(subs, list):
                    unique_subs.update(subs)

        unique_subs_list = sorted(list(unique_subs))
        if not unique_subs_list:
            return

        embeddings = self._get_sbert_embeddings(unique_subs_list, batch_size=256)
        self.subreddit_embeddings_map = {
            sub: emb for sub, emb in zip(unique_subs_list, embeddings)
        }

    def _compute_history_features(self, df, title_embs, body_embs):
        """
        Computes Peak-Relevance, Global Alignment, and Sequence History for MLP.
        """
        peak_sim_title = []
        peak_sim_body = []
        centroid_sim_title = []
        centroid_sim_body = []

        # For MLP: Sequence of history embeddings (padded)
        # We need a fixed max length. Let's use 50 (same as Top K logic roughly)
        MAX_SEQ_LEN = 50
        history_sequences = []
        history_masks = []  # 1 for real, 0 for pad

        # Pre-fetch embedding dim
        emb_dim = Config.MLP_INPUT_EMBEDDING_DIM

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing History"):
            subs = row["requester_subreddits_at_request"]
            if not isinstance(subs, list):
                subs = []

            # Filter subs that have embeddings (should be all, but safety check)
            sub_embs = [
                self.subreddit_embeddings_map[s]
                for s in subs
                if s in self.subreddit_embeddings_map
            ]

            if not sub_embs:
                # No history
                peak_sim_title.append(0.0)
                peak_sim_body.append(0.0)
                centroid_sim_title.append(0.0)
                centroid_sim_body.append(0.0)

                history_sequences.append(
                    np.zeros((MAX_SEQ_LEN, emb_dim), dtype=np.float32)
                )
                history_masks.append(np.zeros(MAX_SEQ_LEN, dtype=np.float32))
                continue

            sub_embs_arr = np.array(sub_embs)  # (N_subs, Dim)

            # --- RF Features ---
            # Peak Similarity
            # Title
            t_sims = cosine_similarity(title_embs[idx].reshape(1, -1), sub_embs_arr)
            peak_sim_title.append(np.max(t_sims))

            # Body
            b_sims = cosine_similarity(body_embs[idx].reshape(1, -1), sub_embs_arr)
            peak_sim_body.append(np.max(b_sims))

            # Centroid Similarity
            centroid = np.mean(sub_embs_arr, axis=0).reshape(1, -1)
            centroid_sim_title.append(
                cosine_similarity(title_embs[idx].reshape(1, -1), centroid)[0][0]
            )
            centroid_sim_body.append(
                cosine_similarity(body_embs[idx].reshape(1, -1), centroid)[0][0]
            )

            # --- MLP Features ---
            # Truncate or Pad
            seq_len = min(len(sub_embs), MAX_SEQ_LEN)
            # Take most recent? The list order isn't guaranteed chronological in JSON,
            # but usually it's just a set. We take first N.
            seq_embs = np.zeros((MAX_SEQ_LEN, emb_dim), dtype=np.float32)
            seq_embs[:seq_len] = sub_embs_arr[:seq_len]

            mask = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
            mask[:seq_len] = 1.0

            history_sequences.append(seq_embs)
            history_masks.append(mask)

        # Stack
        rf_history_feats = np.column_stack(
            [peak_sim_title, peak_sim_body, centroid_sim_title, centroid_sim_body]
        ).astype(np.float32)

        mlp_history_seq = np.stack(history_sequences)
        mlp_history_mask = np.stack(history_masks)

        return rf_history_feats, mlp_history_seq, mlp_history_mask

    def process_data(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Main execution method.
        Checks cache, otherwise runs pipeline.
        Returns:
            rf_data: dict with keys 'train', 'val', 'test', each containing 'X' and 'y' (if avail)
            mlp_data: dict with keys 'train', 'val', 'test', each containing dict of tensors
        """
        cache_files = {
            "rf_train": "rf_data_train.npz",
            "rf_val": "rf_data_val.npz",
            "rf_test": "rf_data_test.npz",
            "mlp_train": "nn_data_train.npz",
            "mlp_val": "nn_data_val.npz",
            "mlp_test": "nn_data_test.npz",
        }

        # Check cache
        if load_cached_data:
            all_exist = all(
                [
                    os.path.exists(os.path.join(Config.CACHE_DIR, f))
                    for f in cache_files.values()
                ]
            )
            if all_exist:
                print("Loading features from cache...")
                rf_data = {}
                mlp_data = {}

                for split in ["train", "val", "test"]:
                    # Load RF
                    rf_loaded = load_npy(cache_files[f"rf_{split}"])
                    rf_data[split] = {k: rf_loaded[k] for k in rf_loaded.files}

                    # Load MLP
                    mlp_loaded = load_npy(cache_files[f"mlp_{split}"])
                    mlp_data[split] = {k: mlp_loaded[k] for k in mlp_loaded.files}

                return rf_data, mlp_data
            else:
                print("Cache incomplete or missing. Regenerating features...")

        # --- 1. SBERT Embeddings (Title & Body) ---
        print("Generating SBERT embeddings for Title and Body...")
        train_title_emb = self._get_sbert_embeddings(df_train["request_title"])
        val_title_emb = self._get_sbert_embeddings(df_val["request_title"])
        test_title_emb = self._get_sbert_embeddings(df_test["request_title"])

        train_body_emb = self._get_sbert_embeddings(df_train["request_text_edit_aware"])
        val_body_emb = self._get_sbert_embeddings(df_val["request_text_edit_aware"])
        test_body_emb = self._get_sbert_embeddings(df_test["request_text_edit_aware"])

        # --- 2. Subreddit Embeddings & History Features ---
        self._embed_all_unique_subreddits([df_train, df_val, df_test])

        train_hist_rf, train_hist_seq, train_hist_mask = self._compute_history_features(
            df_train, train_title_emb, train_body_emb
        )
        val_hist_rf, val_hist_seq, val_hist_mask = self._compute_history_features(
            df_val, val_title_emb, val_body_emb
        )
        test_hist_rf, test_hist_seq, test_hist_mask = self._compute_history_features(
            df_test, test_title_emb, test_body_emb
        )

        # --- 3. Top-K Subreddit Indicators (RF) ---
        train_topk, val_topk, test_topk = self._process_subreddits_top_k(
            df_train, df_val, df_test
        )

        # --- 4. Metadata (RF & MLP) ---
        train_meta, val_meta, test_meta, _ = self._process_metadata(
            df_train, df_val, df_test
        )

        # --- 5. TF-IDF (RF) ---
        train_tfidf, val_tfidf, test_tfidf = self._compute_tfidf(
            df_train, df_val, df_test
        )

        # --- 6. Assemble Outputs ---

        # RF Assembly: Concatenate [TF-IDF, Metadata, Top-K, History Scalars]
        # Note: TF-IDF is large, others are small.
        def assemble_rf(tfidf, meta, topk, hist):
            return np.hstack([tfidf, meta, topk, hist]).astype(np.float32)

        X_train_rf = assemble_rf(train_tfidf, train_meta, train_topk, train_hist_rf)
        X_val_rf = assemble_rf(val_tfidf, val_meta, val_topk, val_hist_rf)
        X_test_rf = assemble_rf(test_tfidf, test_meta, test_topk, test_hist_rf)

        # Targets
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        # Test target not used for prediction, but structure consistency

        rf_data = {
            "train": {"X": X_train_rf, "y": y_train},
            "val": {"X": X_val_rf, "y": y_val},
            "test": {"X": X_test_rf},
        }

        # MLP Assembly: Dictionary of tensors
        # Structure: title_emb, body_emb, history_seq, history_mask, meta
        mlp_data = {
            "train": {
                "title_emb": train_title_emb,
                "body_emb": train_body_emb,
                "history_seq": train_hist_seq,
                "history_mask": train_hist_mask,
                "meta": train_meta,
                "y": y_train,
            },
            "val": {
                "title_emb": val_title_emb,
                "body_emb": val_body_emb,
                "history_seq": val_hist_seq,
                "history_mask": val_hist_mask,
                "meta": val_meta,
                "y": y_val,
            },
            "test": {
                "title_emb": test_title_emb,
                "body_emb": test_body_emb,
                "history_seq": test_hist_seq,
                "history_mask": test_hist_mask,
                "meta": test_meta,
            },
        }

        # --- 7. Save to Cache ---
        print("Saving features to cache...")
        save_npy(rf_data["train"], cache_files["rf_train"])
        save_npy(rf_data["val"], cache_files["rf_val"])
        save_npy(rf_data["test"], cache_files["rf_test"])

        save_npy(mlp_data["train"], cache_files["mlp_train"])
        save_npy(mlp_data["val"], cache_files["mlp_val"])
        save_npy(mlp_data["test"], cache_files["mlp_test"])

        return rf_data, mlp_data
