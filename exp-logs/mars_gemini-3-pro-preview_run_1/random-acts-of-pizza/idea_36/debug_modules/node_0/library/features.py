import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from library.config import Config
from library.utils import save_file, load_file, set_seed, ensure_dir


class FeatureGenerator:
    """
    Handles feature engineering for the Hybrid Ensemble.
    Generates:
    1. Stream A (RF): TF-IDF, Metadata, Top-K Subreddits, Dispersion Metrics.
    2. Stream B (MLP): SBERT Embeddings (Title, Body), History Sequences, Centroids, Metadata.
    """

    def __init__(self):
        self.sbert = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.imputer = None
        self.top_k_subreddits = None
        self.subreddit_embedding_map = {}

        # Cache paths
        self.rf_features_path = os.path.join(Config.WORKING_DIR, "rf_features.npz")
        self.mlp_features_path = os.path.join(Config.WORKING_DIR, "mlp_features.npz")
        self.meta_state_path = os.path.join(Config.WORKING_DIR, "feature_gen_state.npz")

    def _load_sbert(self):
        if self.sbert is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}...")
            self.sbert = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _get_unique_subreddits(self, dfs):
        """Extracts unique subreddits from all dataframes."""
        unique_subs = set()
        for df in dfs:
            if "requester_subreddits_at_request" in df.columns:
                for sub_list in df["requester_subreddits_at_request"]:
                    if isinstance(sub_list, list):
                        unique_subs.update(sub_list)
        return sorted(list(unique_subs))

    def _embed_subreddits(self, subreddits):
        """Embeds a list of subreddits using SBERT."""
        self._load_sbert()
        print(f"Embedding {len(subreddits)} unique subreddits...")
        # Batch encoding
        embeddings = self.sbert.encode(
            subreddits, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        return dict(zip(subreddits, embeddings))

    def _compute_dispersion_profile(self, df, title_embs, body_embs):
        """
        Computes Consistency Profile: Centroids, Dispersion, Divergence, Z-Scores.
        """
        n_samples = len(df)
        emb_dim = Config.EMBEDDING_DIM

        # Outputs
        centroids = np.zeros((n_samples, emb_dim), dtype=np.float32)
        dispersions = np.zeros((n_samples, 1), dtype=np.float32)
        divergences = np.zeros((n_samples, 1), dtype=np.float32)
        z_scores = np.zeros((n_samples, 1), dtype=np.float32)

        # History sequences for MLP (padded)
        max_len = Config.TOP_K_SUBREDDITS
        history_seqs = np.zeros((n_samples, max_len, emb_dim), dtype=np.float32)
        history_masks = np.zeros(
            (n_samples, max_len), dtype=np.float32
        )  # 1 for valid, 0 for pad

        sub_col = "requester_subreddits_at_request"

        # Request embedding (Mean of Title + Body for divergence calc)
        request_embs = (title_embs + body_embs) / 2.0

        print("Computing dispersion profiles...")
        for i, row in tqdm(df.iterrows(), total=n_samples, desc="Dispersion"):
            subs = row[sub_col] if isinstance(row[sub_col], list) else []

            # Filter subs present in embedding map
            valid_subs = [s for s in subs if s in self.subreddit_embedding_map]

            if not valid_subs:
                # No history: Centroid is 0, Dispersion 0, Divergence is norm of request
                # Z-score 0 (neutral)
                divergences[i] = np.linalg.norm(request_embs[i])
                continue

            # Get embeddings
            sub_embs = np.array([self.subreddit_embedding_map[s] for s in valid_subs])

            # Truncate to max_len for MLP sequence
            seq_len = min(len(sub_embs), max_len)
            history_seqs[i, :seq_len, :] = sub_embs[:seq_len]
            history_masks[i, :seq_len] = 1.0

            # Centroid (Mean of all history)
            centroid = np.mean(sub_embs, axis=0)
            centroids[i] = centroid

            # Distances from centroid to history items
            dists = np.linalg.norm(sub_embs - centroid, axis=1)

            # Dispersion (Std Dev)
            dispersion = np.std(dists)
            dispersions[i] = dispersion

            # Request Divergence
            req_div = np.linalg.norm(request_embs[i] - centroid)
            divergences[i] = req_div

            # Z-Score
            # (Distance of Request - Mean Distance of History) / (Dispersion + epsilon)
            mean_hist_dist = np.mean(dists)
            z_score = (req_div - mean_hist_dist) / (dispersion + 1e-6)
            z_scores[i] = z_score

        return {
            "centroids": centroids,
            "dispersions": dispersions,
            "divergences": divergences,
            "z_scores": z_scores,
            "history_seqs": history_seqs,
            "history_masks": history_masks,
        }

    def _process_metadata(self, df, fit=False):
        """
        Extracts and transforms numerical metadata.
        """
        # Define numerical columns
        num_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_number_of_subreddits_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",
        ]

        # Select columns, handle missing
        X_num = df[num_cols].copy()

        # Simple imputation
        if fit:
            self.imputer = SimpleImputer(strategy="median")
            X_num = self.imputer.fit_transform(X_num)
        else:
            X_num = self.imputer.transform(X_num)

        # Arcsinh transform (handle skew/negatives better than log)
        if Config.USE_ARCSINH_TRANSFORM:
            X_num = np.arcsinh(X_num)

        # Scaling
        if fit:
            self.scaler = StandardScaler()
            X_num = self.scaler.fit_transform(X_num)
        else:
            X_num = self.scaler.transform(X_num)

        return X_num.astype(np.float32)

    def _generate_top_k_features(self, df, fit=False):
        """
        Generates binary flags for top K subreddits.
        """
        subs = df["requester_subreddits_at_request"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        if fit:
            # Count frequency
            all_subs = [s for sublist in subs for s in sublist]
            counts = pd.Series(all_subs).value_counts()
            self.top_k_subreddits = counts.head(Config.TOP_K_SUBREDDITS).index.tolist()

        # Manual one-hot encoding for fixed top-k
        X_top_k = np.zeros((len(df), len(self.top_k_subreddits)), dtype=np.float32)

        for i, sub_list in enumerate(subs):
            sub_set = set(sub_list)
            for j, target_sub in enumerate(self.top_k_subreddits):
                if target_sub in sub_set:
                    X_top_k[i, j] = 1.0

        return X_top_k

    def process_data(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Main pipeline to generate features.
        """
        # Check cache
        if (
            load_cached_data
            and os.path.exists(self.rf_features_path)
            and os.path.exists(self.mlp_features_path)
        ):
            print("Loading features from cache...")
            rf_data = load_file(self.rf_features_path)
            mlp_data = load_file(self.mlp_features_path)
            return rf_data, mlp_data

        print("Starting Feature Generation...")
        self._load_sbert()

        # 1. Embed Unique Subreddits
        unique_subs = self._get_unique_subreddits([df_train, df_val, df_test])
        self.subreddit_embedding_map = self._embed_subreddits(unique_subs)

        # 2. Embed Request Texts (Title & Body)
        # Combine for TF-IDF, keep separate for SBERT
        all_dfs = [df_train, df_val, df_test]
        titles = []
        bodies = []
        full_texts = []

        for df in all_dfs:
            t = df["request_title"].fillna("").astype(str).tolist()
            b = df["request_text_edit_aware"].fillna("").astype(str).tolist()
            titles.extend(t)
            bodies.extend(b)
            full_texts.extend([ti + " " + bo for ti, bo in zip(t, b)])

        # SBERT Encoding
        print("Embedding Titles...")
        title_embs = self.sbert.encode(
            titles, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        print("Embedding Bodies...")
        body_embs = self.sbert.encode(
            bodies, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )

        # Split back
        n_train = len(df_train)
        n_val = len(df_val)
        n_test = len(df_test)

        splits = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, n_train + n_val + n_test),
        }

        # 3. TF-IDF (Stream A)
        print("Generating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_VOCAB_SIZE,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )
        # Fit on train only to prevent leakage
        train_texts = full_texts[:n_train]
        self.tfidf_vectorizer.fit(train_texts)
        tfidf_matrix = self.tfidf_vectorizer.transform(full_texts)

        # 4. Dispersion & Centroids
        dispersion_data = {}
        for name, (start, end) in splits.items():
            df_slice = all_dfs[["train", "val", "test"].index(name)]
            t_slice = title_embs[start:end]
            b_slice = body_embs[start:end]
            dispersion_data[name] = self._compute_dispersion_profile(
                df_slice, t_slice, b_slice
            )

        # 5. Metadata & Top-K
        print("Processing Metadata...")
        meta_data = {}
        top_k_data = {}

        # Train
        meta_data["train"] = self._process_metadata(df_train, fit=True)
        top_k_data["train"] = self._generate_top_k_features(df_train, fit=True)

        # Val & Test
        meta_data["val"] = self._process_metadata(df_val, fit=False)
        top_k_data["val"] = self._generate_top_k_features(df_val, fit=False)

        meta_data["test"] = self._process_metadata(df_test, fit=False)
        top_k_data["test"] = self._generate_top_k_features(df_test, fit=False)

        # 6. Assemble RF Features (Stream A)
        # X = [TFIDF, Meta, Top-K, Dispersion, Divergence, Z-Score]
        rf_outputs = {}
        for name in ["train", "val", "test"]:
            idx_start, idx_end = splits[name]

            # TF-IDF (Sparse) -> Dense for concatenation (vocab is 5000, manageable for 4k rows)
            # If memory is tight, we would keep sparse, but here 4000x5000 float32 is ~80MB. Safe.
            tfidf_part = tfidf_matrix[idx_start:idx_end].toarray().astype(np.float32)

            meta_part = meta_data[name]
            top_k_part = top_k_data[name]

            disp = dispersion_data[name]
            disp_feats = np.hstack(
                [disp["dispersions"], disp["divergences"], disp["z_scores"]]
            )

            X = np.hstack([tfidf_part, meta_part, top_k_part, disp_feats])

            # Get Targets
            y = None
            if name != "test":
                df_curr = all_dfs[["train", "val", "test"].index(name)]
                if "requester_received_pizza" in df_curr.columns:
                    y = df_curr["requester_received_pizza"].astype(int).values

            rf_outputs[f"X_{name}"] = X
            if y is not None:
                rf_outputs[f"y_{name}"] = y

        # 7. Assemble MLP Features (Stream B)
        mlp_outputs = {}
        for name in ["train", "val", "test"]:
            idx_start, idx_end = splits[name]
            disp = dispersion_data[name]

            data_dict = {
                "title_emb": title_embs[idx_start:idx_end],
                "body_emb": body_embs[idx_start:idx_end],
                "history_seqs": disp["history_seqs"],
                "history_masks": disp["history_masks"],
                "centroids": disp["centroids"],
                "metadata": meta_data[name],  # Using same scaled metadata
                "alignment_feats": np.hstack(
                    [disp["divergences"], disp["z_scores"]]
                ),  # Extra scalars for MLP
            }

            # Add labels
            if name != "test":
                df_curr = all_dfs[["train", "val", "test"].index(name)]
                if "requester_received_pizza" in df_curr.columns:
                    data_dict["labels"] = (
                        df_curr["requester_received_pizza"]
                        .astype(int)
                        .values.astype(np.float32)
                    )

            # Prefix keys with split name for storage
            for k, v in data_dict.items():
                mlp_outputs[f"{name}_{k}"] = v

        # Save
        print("Saving features to cache...")
        save_file(rf_outputs, self.rf_features_path)
        save_file(mlp_outputs, self.mlp_features_path)

        return rf_outputs, mlp_outputs
