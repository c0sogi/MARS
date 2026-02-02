import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer
from library import config, utils


class FeatureEngineer:
    """
    Implements the feature engineering pipeline for the Hybrid Ensemble.
    Generates distinct feature sets for:
    1. Stream A (Random Forest): Interaction-Projected Top-K Features (Sparse TF-IDF + Interactions).
    2. Stream B (MLP): Topology-Aware Semantic Features (Dense SBERT + Scaled Metadata).
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.sbert_model_name = config.SBERT_MODEL_NAME
        self.top_k = config.TOP_K_SUBREDDITS
        self.tfidf_max_features = config.TFIDF_MAX_FEATURES
        self.random_state = config.RANDOM_STATE

        # Define numerical columns to use from metadata
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

    def _get_cache_paths(self):
        """Returns paths for cached feature files."""
        return {
            "rf_train": os.path.join(self.cache_dir, "rf_features_train.npz"),
            "rf_val": os.path.join(self.cache_dir, "rf_features_val.npz"),
            "rf_test": os.path.join(self.cache_dir, "rf_features_test.npz"),
            "mlp_train": os.path.join(self.cache_dir, "mlp_features_train.npz"),
            "mlp_val": os.path.join(self.cache_dir, "mlp_features_val.npz"),
            "mlp_test": os.path.join(self.cache_dir, "mlp_features_test.npz"),
            "y_train": os.path.join(self.cache_dir, "y_train.npy"),
            "y_val": os.path.join(self.cache_dir, "y_val.npy"),
        }

    def _compute_sbert_embeddings(self, model, texts):
        """Computes SBERT embeddings for a list of texts."""
        # Handle empty or non-string inputs gracefully
        cleaned_texts = [t if isinstance(t, str) and len(t) > 0 else " " for t in texts]
        embeddings = model.encode(
            cleaned_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings

    def _compute_history_centroids(self, model, subreddits_series):
        """
        Computes the centroid of subreddit embeddings for each user.
        """
        # 1. Identify all unique subreddits across the series
        all_subreddits = set()
        for sub_list in subreddits_series:
            if isinstance(sub_list, list):
                all_subreddits.update(sub_list)

        unique_subs_list = sorted(list(all_subreddits))
        if not unique_subs_list:
            return np.zeros(
                (len(subreddits_series), model.get_sentence_embedding_dimension())
            )

        # 2. Encode all unique subreddits
        sub_embeddings = model.encode(
            unique_subs_list,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        sub_map = {sub: emb for sub, emb in zip(unique_subs_list, sub_embeddings)}

        # 3. Compute centroids
        centroids = []
        embedding_dim = model.get_sentence_embedding_dimension()

        for sub_list in subreddits_series:
            if isinstance(sub_list, list) and len(sub_list) > 0:
                # Average embedding of user's subreddits
                user_sub_embs = [sub_map[s] for s in sub_list if s in sub_map]
                if user_sub_embs:
                    centroids.append(np.mean(user_sub_embs, axis=0))
                else:
                    centroids.append(np.zeros(embedding_dim))
            else:
                centroids.append(np.zeros(embedding_dim))

        return np.vstack(centroids)

    def _get_top_k_binary(self, train_subs, target_subs_series):
        """
        Generates binary indicators for the top K most frequent subreddits found in training data.
        """
        # Count frequencies in training
        from collections import Counter

        counts = Counter()
        for sub_list in train_subs:
            if isinstance(sub_list, list):
                counts.update(sub_list)

        top_k_subs = [sub for sub, _ in counts.most_common(self.top_k)]

        # Create binary matrix
        matrix = np.zeros((len(target_subs_series), self.top_k), dtype=np.float32)
        for i, sub_list in enumerate(target_subs_series):
            if isinstance(sub_list, list):
                s_set = set(sub_list)
                for j, top_sub in enumerate(top_k_subs):
                    if top_sub in s_set:
                        matrix[i, j] = 1.0
        return matrix

    def process_data(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main pipeline execution. Checks cache, otherwise computes features.
        """
        paths = self._get_cache_paths()

        # Check if cache exists
        cache_exists = all(os.path.exists(p) for p in paths.values())

        if load_cached_data and cache_exists:
            print("Loading features from cache...")
            try:
                X_train_rf = sp.load_npz(paths["rf_train"])
                X_val_rf = sp.load_npz(paths["rf_val"])
                X_test_rf = sp.load_npz(paths["rf_test"])

                X_train_mlp = np.load(paths["mlp_train"])["arr_0"]
                X_val_mlp = np.load(paths["mlp_val"])["arr_0"]
                X_test_mlp = np.load(paths["mlp_test"])["arr_0"]

                y_train = np.load(paths["y_train"])
                y_val = np.load(paths["y_val"])

                return (X_train_rf, y_train, X_val_rf, y_val, X_test_rf), (
                    X_train_mlp,
                    y_train,
                    X_val_mlp,
                    y_val,
                    X_test_mlp,
                )
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing features.")

        print("Computing features from scratch...")

        # --- Preprocessing ---
        # Fill missing text
        for df in [train_df, val_df, test_df]:
            df["request_title"] = df["request_title"].fillna("")
            df["request_text_edit_aware"] = df["request_text_edit_aware"].fillna("")
            # Create combined text for TF-IDF
            df["combined_text"] = (
                df["request_title"] + " " + df["request_text_edit_aware"]
            )

        # --- 1. SBERT & Semantic Features (Shared/MLP) ---
        print("Encoding semantic features (SBERT)...")
        sbert = SentenceTransformer(self.sbert_model_name)

        # Encode Title and Body
        # We process all together to batch efficiently, then split, or loop per df. Looping is safer for memory.
        def get_semantic_vectors(df):
            title_emb = self._compute_sbert_embeddings(
                sbert, df["request_title"].tolist()
            )
            body_emb = self._compute_sbert_embeddings(
                sbert, df["request_text_edit_aware"].tolist()
            )
            centroid_emb = self._compute_history_centroids(
                sbert, df["requester_subreddits_at_request"]
            )
            return title_emb, body_emb, centroid_emb

        train_sem = get_semantic_vectors(train_df)
        val_sem = get_semantic_vectors(val_df)
        test_sem = get_semantic_vectors(test_df)

        # Compute Consistency Scalars (Cosine Similarity)
        # paired_cosine_distances returns 1 - cos_sim. So Sim = 1 - dist.
        def get_consistency(sem_tuple):
            t, b, c = sem_tuple
            # Title-History Consistency
            sim_title = 1 - paired_cosine_distances(t, c)
            # Body-History Consistency
            sim_body = 1 - paired_cosine_distances(b, c)
            return sim_title.reshape(-1, 1), sim_body.reshape(-1, 1)

        train_cons = get_consistency(train_sem)
        val_cons = get_consistency(val_sem)
        test_cons = get_consistency(test_sem)

        # --- 2. Top-K Community Indicators ---
        print(f"Generating Top-{self.top_k} community indicators...")
        train_subs = train_df["requester_subreddits_at_request"].tolist()

        X_train_topk = self._get_top_k_binary(
            train_subs, train_df["requester_subreddits_at_request"]
        )
        X_val_topk = self._get_top_k_binary(
            train_subs, val_df["requester_subreddits_at_request"]
        )
        X_test_topk = self._get_top_k_binary(
            train_subs, test_df["requester_subreddits_at_request"]
        )

        # --- 3. Numerical Metadata (MLP vs RF) ---
        print("Processing numerical metadata...")
        # Extract raw
        X_train_meta_raw = train_df[self.meta_cols].values
        X_val_meta_raw = val_df[self.meta_cols].values
        X_test_meta_raw = test_df[self.meta_cols].values

        # MLP: Arcsinh + StandardScaler
        scaler = StandardScaler()
        X_train_meta_mlp = scaler.fit_transform(
            np.arcsinh(np.nan_to_num(X_train_meta_raw))
        )
        X_val_meta_mlp = scaler.transform(np.arcsinh(np.nan_to_num(X_val_meta_raw)))
        X_test_meta_mlp = scaler.transform(np.arcsinh(np.nan_to_num(X_test_meta_raw)))

        # RF: Simple Imputation (Median)
        imputer = SimpleImputer(strategy="median")
        X_train_meta_rf = imputer.fit_transform(X_train_meta_raw)
        X_val_meta_rf = imputer.transform(X_val_meta_raw)
        X_test_meta_rf = imputer.transform(X_test_meta_raw)

        # --- 4. Explicit Interaction Features (RF Only) ---
        print("Generating interaction features for RF...")

        def get_interactions(meta_rf, cons_tuple, df):
            # Indices in meta_cols:
            # 0: account_age
            # 8: upvotes_plus_downvotes (sum)
            # 7: upvotes_minus_downvotes (diff)

            age = meta_rf[:, 0]
            sum_votes = meta_rf[:, 8]
            diff_votes = meta_rf[:, 7]

            # Upvotes = (Sum + Diff) / 2. Avoid div by zero in ratio.
            upvotes = (sum_votes + diff_votes) / 2
            # Ratio: Up / (Sum + epsilon)
            upvote_ratio = upvotes / (sum_votes + 1e-5)

            sim_title, sim_body = cons_tuple

            # I1: Topic Consistency * log(1 + Age)
            i1 = sim_title.flatten() * np.log1p(age)

            # I2: Narrative Consistency * Upvote Ratio
            i2 = sim_body.flatten() * upvote_ratio

            return np.stack([i1, i2], axis=1)

        X_train_inter = get_interactions(X_train_meta_rf, train_cons, train_df)
        X_val_inter = get_interactions(X_val_meta_rf, val_cons, val_df)
        X_test_inter = get_interactions(X_test_meta_rf, test_cons, test_df)

        # --- 5. TF-IDF (RF Only) ---
        print("Fitting TF-IDF...")
        tfidf = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words="english",
        )
        X_train_tfidf = tfidf.fit_transform(train_df["combined_text"])
        X_val_tfidf = tfidf.transform(val_df["combined_text"])
        X_test_tfidf = tfidf.transform(test_df["combined_text"])

        # --- 6. Assembly ---
        print("Assembling final feature matrices...")

        # Stream A: RF (Sparse)
        # [TFIDF, Meta_RF, TopK, Interactions, Consistency]
        # Consistency scalars are also useful raw for RF
        X_train_rf = sp.hstack(
            [
                X_train_tfidf,
                X_train_meta_rf,
                X_train_topk,
                X_train_inter,
                train_cons[0],
                train_cons[1],
            ]
        ).tocsr()

        X_val_rf = sp.hstack(
            [
                X_val_tfidf,
                X_val_meta_rf,
                X_val_topk,
                X_val_inter,
                val_cons[0],
                val_cons[1],
            ]
        ).tocsr()

        X_test_rf = sp.hstack(
            [
                X_test_tfidf,
                X_test_meta_rf,
                X_test_topk,
                X_test_inter,
                test_cons[0],
                test_cons[1],
            ]
        ).tocsr()

        # Stream B: MLP (Dense)
        # [SBERT_Title, SBERT_Body, Centroid, Consistency, Meta_MLP, TopK]
        X_train_mlp = np.hstack(
            [
                train_sem[0],
                train_sem[1],
                train_sem[2],
                train_cons[0],
                train_cons[1],
                X_train_meta_mlp,
                X_train_topk,
            ]
        ).astype(np.float32)

        X_val_mlp = np.hstack(
            [
                val_sem[0],
                val_sem[1],
                val_sem[2],
                val_cons[0],
                val_cons[1],
                X_val_meta_mlp,
                X_val_topk,
            ]
        ).astype(np.float32)

        X_test_mlp = np.hstack(
            [
                test_sem[0],
                test_sem[1],
                test_sem[2],
                test_cons[0],
                test_cons[1],
                X_test_meta_mlp,
                X_test_topk,
            ]
        ).astype(np.float32)

        # Targets
        y_train = train_df["requester_received_pizza"].astype(int).values
        y_val = val_df["requester_received_pizza"].astype(int).values

        # --- 7. Caching ---
        print("Saving features to cache...")
        sp.save_npz(paths["rf_train"], X_train_rf)
        sp.save_npz(paths["rf_val"], X_val_rf)
        sp.save_npz(paths["rf_test"], X_test_rf)

        np.savez(paths["mlp_train"], X_train_mlp)
        np.savez(paths["mlp_val"], X_val_mlp)
        np.savez(paths["mlp_test"], X_test_mlp)

        np.save(paths["y_train"], y_train)
        np.save(paths["y_val"], y_val)

        return (X_train_rf, y_train, X_val_rf, y_val, X_test_rf), (
            X_train_mlp,
            y_train,
            X_val_mlp,
            y_val,
            X_test_mlp,
        )
