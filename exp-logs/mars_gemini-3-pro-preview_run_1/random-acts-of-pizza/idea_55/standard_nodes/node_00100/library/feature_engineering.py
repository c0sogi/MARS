import os
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import set_seed
from library.data_loader import load_raw_data, get_feature_intersection


class FeaturePipeline:
    def __init__(self):
        self.sbert_model = None
        self.tfidf_vectorizer = None
        self.scaler = None
        self.imputer = None
        self.top_k_subreddits = None

        # Define specific interaction pairs for RF
        # (Consistency Scalar, Credibility Metric)
        self.interaction_pairs = [
            ("topic_consistency", "requester_account_age_in_days_at_request"),
            ("narrative_consistency", "requester_account_age_in_days_at_request"),
            ("topic_consistency", "requester_upvotes_minus_downvotes_at_request"),
            ("narrative_consistency", "requester_upvotes_minus_downvotes_at_request"),
        ]

    def _load_sbert(self):
        if self.sbert_model is None:
            print(f"Loading SBERT model: {Config.SBERT_MODEL_NAME}")
            self.sbert_model = SentenceTransformer(
                Config.SBERT_MODEL_NAME, device=Config.DEVICE
            )

    def _get_embeddings(self, texts, cache_name):
        """
        Generates or loads SBERT embeddings for a list of texts.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"sbert_{cache_name}.npy")

        if os.path.exists(cache_path):
            return np.load(cache_path)

        self._load_sbert()
        # Encode in batches
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        np.save(cache_path, embeddings)
        return embeddings

    def _process_history(self, df, split_name):
        """
        Processes user history:
        1. Parses subreddit lists.
        2. Computes/retrieves embeddings for unique subreddits.
        3. Generates Centroids (for RF/MLP) and Sequences (for MLP).
        """
        cache_path_centroid = os.path.join(
            Config.CACHE_DIR, f"hist_centroid_{split_name}.npy"
        )
        cache_path_seq = os.path.join(Config.CACHE_DIR, f"hist_seq_{split_name}.npy")
        cache_path_mask = os.path.join(Config.CACHE_DIR, f"hist_mask_{split_name}.npy")

        if (
            os.path.exists(cache_path_centroid)
            and os.path.exists(cache_path_seq)
            and os.path.exists(cache_path_mask)
        ):
            return (
                np.load(cache_path_centroid),
                np.load(cache_path_seq),
                np.load(cache_path_mask),
            )

        # Parse stringified lists
        col_name = Config.LIST_COLS["subreddits"]
        # Handle potential NaNs or non-string types by ensuring string format first
        raw_lists = (
            df[col_name]
            .fillna("[]")
            .astype(str)
            .apply(lambda x: ast.literal_eval(x) if x.startswith("[") else [])
            .tolist()
        )

        # Identify unique subreddits to minimize embedding calls
        unique_subs = set(sub for sub_list in raw_lists for sub in sub_list)
        unique_subs_list = sorted(list(unique_subs))
        sub_to_idx = {sub: i for i, sub in enumerate(unique_subs_list)}

        # Embed unique subreddits
        # We use a unique cache key based on the split to avoid collisions if vocab differs,
        # though ideally a global vocab is better. For simplicity, we embed per split's needs
        # or we could try to load a global map. Here we embed what's needed.
        # To save time, check if we have a global vocab cache, if not create one.
        # For this implementation, we'll just embed the current split's unique subs.
        sub_embeddings = self._get_embeddings(
            unique_subs_list, f"unique_subs_{split_name}"
        )

        # Prepare outputs
        centroids = []
        sequences = []
        masks = []

        max_len = Config.TOP_K_SUBREDDITS  # Use Top-K as sequence length limit
        emb_dim = Config.SBERT_EMBEDDING_DIM

        for sub_list in raw_lists:
            if not sub_list:
                # No history
                centroids.append(np.zeros(emb_dim))
                sequences.append(np.zeros((max_len, emb_dim)))
                masks.append(np.zeros(max_len))
                continue

            # Get indices
            idxs = [sub_to_idx[s] for s in sub_list if s in sub_to_idx]
            if not idxs:
                centroids.append(np.zeros(emb_dim))
                sequences.append(np.zeros((max_len, emb_dim)))
                masks.append(np.zeros(max_len))
                continue

            # Get embeddings
            embs = sub_embeddings[idxs]

            # Centroid
            centroids.append(np.mean(embs, axis=0))

            # Sequence (Truncate/Pad)
            seq_len = min(len(embs), max_len)
            padded_seq = np.zeros((max_len, emb_dim))
            padded_seq[:seq_len] = embs[:seq_len]
            sequences.append(padded_seq)

            # Mask (1 for real token, 0 for padding)
            # Note: MLP logic might expect additive mask (-inf), but standard PyTorch
            # attention often takes 0/1 boolean masks. The prompt mentions "Explicit Additive Masking".
            # We will provide 0/1 here, and the model can convert to additive mask.
            mask = np.zeros(max_len)
            mask[:seq_len] = 1.0
            masks.append(mask)

        centroids = np.array(centroids, dtype=np.float32)
        sequences = np.array(sequences, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        np.save(cache_path_centroid, centroids)
        np.save(cache_path_seq, sequences)
        np.save(cache_path_mask, masks)

        return centroids, sequences, masks

    def _compute_consistency(self, title_embs, body_embs, centroids):
        """
        Computes Cosine Similarity between:
        1. Title and History Centroid
        2. Body and History Centroid
        """
        # Cosine similarity returns (N, N), we want diagonal (N,)
        # Efficient way: sum(A * B, axis=1) since vectors are normalized by SBERT
        # However, centroids might not be normalized.

        # Normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        centroids_norm = centroids / norms

        # SBERT embeddings are already normalized
        topic_consistency = np.sum(title_embs * centroids_norm, axis=1)
        narrative_consistency = np.sum(body_embs * centroids_norm, axis=1)

        return topic_consistency, narrative_consistency

    def _get_tfidf(self, df_train, df_val, df_test):
        """
        Computes TF-IDF features. Fits on Train, transforms all.
        """
        print("Generating TF-IDF features...")
        # Concatenate title and body
        train_text = (
            df_train[Config.TEXT_COLS["title"]]
            + " "
            + df_train[Config.TEXT_COLS["body"]]
        )
        val_text = (
            df_val[Config.TEXT_COLS["title"]] + " " + df_val[Config.TEXT_COLS["body"]]
        )
        test_text = (
            df_test[Config.TEXT_COLS["title"]] + " " + df_test[Config.TEXT_COLS["body"]]
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            stop_words="english",
            dtype=np.float32,
        )

        X_train = self.tfidf_vectorizer.fit_transform(train_text).toarray()
        X_val = self.tfidf_vectorizer.transform(val_text).toarray()
        X_test = self.tfidf_vectorizer.transform(test_text).toarray()

        return X_train, X_val, X_test

    def _get_top_k_subreddits(self, df_train, df_val, df_test):
        """
        Generates binary indicators for top K subreddits found in training data.
        """
        print("Generating Top-K Subreddit indicators...")
        col_name = Config.LIST_COLS["subreddits"]

        # Flatten train subreddits to find top K
        all_subs = []
        for x in df_train[col_name].fillna("[]").astype(str):
            if x.startswith("["):
                all_subs.extend(ast.literal_eval(x))

        from collections import Counter

        counts = Counter(all_subs)
        top_k = [sub for sub, _ in counts.most_common(Config.TOP_K_SUBREDDITS)]
        self.top_k_subreddits = top_k

        def transform(df):
            matrix = np.zeros((len(df), len(top_k)), dtype=np.float32)
            raw_lists = (
                df[col_name]
                .fillna("[]")
                .astype(str)
                .apply(lambda x: ast.literal_eval(x) if x.startswith("[") else [])
                .tolist()
            )

            for i, sub_list in enumerate(raw_lists):
                s_set = set(sub_list)
                for j, target in enumerate(top_k):
                    if target in s_set:
                        matrix[i, j] = 1.0
            return matrix

        return transform(df_train), transform(df_val), transform(df_test)

    def _process_metadata_rf(self, df, topic_cons, narr_cons):
        """
        Prepares metadata for Random Forest:
        1. Selects numeric columns.
        2. Imputes NaNs.
        3. Adds Consistency Scalars.
        4. Adds Interaction Features.
        """
        # Select numeric cols
        meta_cols = Config.NUMERICAL_COLS
        X_meta = df[meta_cols].values.astype(np.float32)

        # Impute (Fit on self if training, but here we do simple median per batch or fit once)
        # Ideally fit on train, transform others. We'll handle this in the main flow.
        return X_meta

    def process_data(self, load_cached_data=True):
        """
        Main execution method.
        """
        set_seed()
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Check if final outputs exist
        files = [
            "X_rf_train.npy",
            "X_rf_val.npy",
            "X_rf_test.npy",
            "X_mlp_train.npz",
            "X_mlp_val.npz",
            "X_mlp_test.npz",
            "y_train.npy",
            "y_val.npy",
        ]

        all_exist = all(
            os.path.exists(os.path.join(Config.CACHE_DIR, f)) for f in files
        )

        if load_cached_data and all_exist:
            print("Loading engineered features from cache...")
            X_rf_train = np.load(os.path.join(Config.CACHE_DIR, "X_rf_train.npy"))
            X_rf_val = np.load(os.path.join(Config.CACHE_DIR, "X_rf_val.npy"))
            X_rf_test = np.load(os.path.join(Config.CACHE_DIR, "X_rf_test.npy"))

            y_train = np.load(os.path.join(Config.CACHE_DIR, "y_train.npy"))
            y_val = np.load(os.path.join(Config.CACHE_DIR, "y_val.npy"))

            def load_mlp(split):
                data = np.load(os.path.join(Config.CACHE_DIR, f"X_mlp_{split}.npz"))
                return {k: torch.tensor(data[k]) for k in data.files}

            X_mlp_train = load_mlp("train")
            X_mlp_val = load_mlp("val")
            X_mlp_test = load_mlp("test")

            return (
                X_rf_train,
                X_rf_val,
                X_rf_test,
                X_mlp_train,
                X_mlp_val,
                X_mlp_test,
                y_train,
                y_val,
            )

        # --- Fresh Processing ---
        print("Starting fresh feature engineering...")

        # Load Raw Data
        df_train = load_raw_data("train", load_cached_data)
        df_val = load_raw_data("val", load_cached_data)
        df_test = load_raw_data("test", load_cached_data)

        # Extract Targets
        y_train = df_train[Config.TARGET_COL].astype(int).values
        y_val = df_val[Config.TARGET_COL].astype(int).values

        # 1. SBERT Embeddings (Title & Body)
        print("Generating SBERT Embeddings...")
        train_title_emb = self._get_embeddings(
            df_train[Config.TEXT_COLS["title"]].tolist(), "train_title"
        )
        val_title_emb = self._get_embeddings(
            df_val[Config.TEXT_COLS["title"]].tolist(), "val_title"
        )
        test_title_emb = self._get_embeddings(
            df_test[Config.TEXT_COLS["title"]].tolist(), "test_title"
        )

        train_body_emb = self._get_embeddings(
            df_train[Config.TEXT_COLS["body"]].tolist(), "train_body"
        )
        val_body_emb = self._get_embeddings(
            df_val[Config.TEXT_COLS["body"]].tolist(), "val_body"
        )
        test_body_emb = self._get_embeddings(
            df_test[Config.TEXT_COLS["body"]].tolist(), "test_body"
        )

        # 2. History Processing (Centroids & Sequences)
        print("Processing User History...")
        train_hist_c, train_hist_s, train_hist_m = self._process_history(
            df_train, "train"
        )
        val_hist_c, val_hist_s, val_hist_m = self._process_history(df_val, "val")
        test_hist_c, test_hist_s, test_hist_m = self._process_history(df_test, "test")

        # 3. Consistency Scalars
        train_topic_cons, train_narr_cons = self._compute_consistency(
            train_title_emb, train_body_emb, train_hist_c
        )
        val_topic_cons, val_narr_cons = self._compute_consistency(
            val_title_emb, val_body_emb, val_hist_c
        )
        test_topic_cons, test_narr_cons = self._compute_consistency(
            test_title_emb, test_body_emb, test_hist_c
        )

        # 4. RF Specifics: TF-IDF & Top-K
        rf_tfidf_train, rf_tfidf_val, rf_tfidf_test = self._get_tfidf(
            df_train, df_val, df_test
        )
        rf_topk_train, rf_topk_val, rf_topk_test = self._get_top_k_subreddits(
            df_train, df_val, df_test
        )

        # 5. Metadata Processing (RF)
        print("Processing RF Metadata...")
        self.imputer = SimpleImputer(strategy="median")

        # Extract raw numeric
        raw_meta_train = df_train[Config.NUMERICAL_COLS].values
        raw_meta_val = df_val[Config.NUMERICAL_COLS].values
        raw_meta_test = df_test[Config.NUMERICAL_COLS].values

        # Impute
        raw_meta_train = self.imputer.fit_transform(raw_meta_train)
        raw_meta_val = self.imputer.transform(raw_meta_val)
        raw_meta_test = self.imputer.transform(raw_meta_test)

        def build_rf_features(raw_meta, topic_cons, narr_cons, df_source):
            # Base: Raw Meta + Consistency
            base = np.column_stack([raw_meta, topic_cons, narr_cons])

            # Interactions
            interactions = []
            for cons_name, cred_col in self.interaction_pairs:
                # Get scalar vector
                if cons_name == "topic_consistency":
                    scalar = topic_cons
                else:
                    scalar = narr_cons

                # Get credibility vector (imputed via raw_meta index)
                # Need to map column name to index in NUMERICAL_COLS
                col_idx = Config.NUMERICAL_COLS.index(cred_col)
                cred_vec = raw_meta[:, col_idx]

                # Interaction: Scalar * log(Cred + 1) or just Scalar * Cred
                # The prompt suggests: Topic_Consistency * log(Account_Age)
                if "age" in cred_col:
                    inter_vec = scalar * np.log1p(cred_vec)
                else:
                    inter_vec = scalar * cred_vec
                interactions.append(inter_vec)

            interactions = np.column_stack(interactions)
            return np.column_stack([base, interactions])

        rf_meta_train = build_rf_features(
            raw_meta_train, train_topic_cons, train_narr_cons, df_train
        )
        rf_meta_val = build_rf_features(
            raw_meta_val, val_topic_cons, val_narr_cons, df_val
        )
        rf_meta_test = build_rf_features(
            raw_meta_test, test_topic_cons, test_narr_cons, df_test
        )

        # Assemble RF
        X_rf_train = np.column_stack(
            [rf_tfidf_train, rf_meta_train, rf_topk_train]
        ).astype(np.float32)
        X_rf_val = np.column_stack([rf_tfidf_val, rf_meta_val, rf_topk_val]).astype(
            np.float32
        )
        X_rf_test = np.column_stack([rf_tfidf_test, rf_meta_test, rf_topk_test]).astype(
            np.float32
        )

        # 6. Metadata Processing (MLP)
        print("Processing MLP Metadata...")
        # Arcsinh transform
        mlp_meta_train = np.arcsinh(raw_meta_train)
        mlp_meta_val = np.arcsinh(raw_meta_val)
        mlp_meta_test = np.arcsinh(raw_meta_test)

        # Scale
        self.scaler = StandardScaler()
        mlp_meta_train = self.scaler.fit_transform(mlp_meta_train)
        mlp_meta_val = self.scaler.transform(mlp_meta_val)
        mlp_meta_test = self.scaler.transform(mlp_meta_test)

        # Add Consistency Scalars to MLP Metadata (as explicit features)
        mlp_meta_train = np.column_stack(
            [mlp_meta_train, train_topic_cons, train_narr_cons]
        )
        mlp_meta_val = np.column_stack([mlp_meta_val, val_topic_cons, val_narr_cons])
        mlp_meta_test = np.column_stack(
            [mlp_meta_test, test_topic_cons, test_narr_cons]
        )

        # Assemble MLP Dicts (Save as npz then convert to tensor on load)
        def save_mlp(split, t_emb, b_emb, h_seq, h_mask, h_cent, meta):
            path = os.path.join(Config.CACHE_DIR, f"X_mlp_{split}.npz")
            np.savez(
                path,
                title_emb=t_emb,
                body_emb=b_emb,
                history_emb=h_seq,
                history_mask=h_mask,
                persona_centroid=h_cent,
                metadata=meta,
            )
            return {
                "title_emb": torch.tensor(t_emb),
                "body_emb": torch.tensor(b_emb),
                "history_emb": torch.tensor(h_seq),
                "history_mask": torch.tensor(h_mask),
                "persona_centroid": torch.tensor(h_cent),
                "metadata": torch.tensor(meta),
            }

        X_mlp_train = save_mlp(
            "train",
            train_title_emb,
            train_body_emb,
            train_hist_s,
            train_hist_m,
            train_hist_c,
            mlp_meta_train,
        )
        X_mlp_val = save_mlp(
            "val",
            val_title_emb,
            val_body_emb,
            val_hist_s,
            val_hist_m,
            val_hist_c,
            mlp_meta_val,
        )
        X_mlp_test = save_mlp(
            "test",
            test_title_emb,
            test_body_emb,
            test_hist_s,
            test_hist_m,
            test_hist_c,
            mlp_meta_test,
        )

        # Save RF and Targets
        np.save(os.path.join(Config.CACHE_DIR, "X_rf_train.npy"), X_rf_train)
        np.save(os.path.join(Config.CACHE_DIR, "X_rf_val.npy"), X_rf_val)
        np.save(os.path.join(Config.CACHE_DIR, "X_rf_test.npy"), X_rf_test)
        np.save(os.path.join(Config.CACHE_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(Config.CACHE_DIR, "y_val.npy"), y_val)

        print("Feature engineering complete and cached.")
        return (
            X_rf_train,
            X_rf_val,
            X_rf_test,
            X_mlp_train,
            X_mlp_val,
            X_mlp_test,
            y_train,
            y_val,
        )
