import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter

from library.config import Config
from library.data_loader import DataLoader
from library.text_processing import SBERTHandler, TFIDFHandler


class FeatureEngineer:
    """
    Constructs complex feature sets for the Hybrid Ensemble.
    Stream A (RF): Interaction-Augmented Features (Metadata, Top-K, Interactions, TF-IDF).
    Stream B (MLP): Semantic & History Features (Embeddings, Sequences, Centroids, Transformed Metadata).
    """

    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        Config.ensure_dirs()

        # Define cache paths for the engineered feature sets
        self.cache_files = {
            "train": {
                "rf": os.path.join(self.cache_dir, "rf_features_train.npz"),
                "mlp": os.path.join(self.cache_dir, "mlp_features_train.npz"),
            },
            "val": {
                "rf": os.path.join(self.cache_dir, "rf_features_val.npz"),
                "mlp": os.path.join(self.cache_dir, "mlp_features_val.npz"),
            },
            "test": {
                "rf": os.path.join(self.cache_dir, "rf_features_test.npz"),
                "mlp": os.path.join(self.cache_dir, "mlp_features_test.npz"),
            },
        }

        self.top_k = Config.TOP_K_SUBREDDITS

    def create_features(self, load_cached_data=True):
        """
        Orchestrates the generation of all features.
        Returns a nested dictionary with keys 'train', 'val', 'test'.
        """
        # 1. Check Cache
        if load_cached_data and self._check_cache():
            print("Loading engineered features from cache...")
            return self._load_cache()

        print("Generating features from scratch...")

        # 2. Load Data
        dl = DataLoader()
        train_df, val_df, test_df = dl.load_dataset(load_cached_data=load_cached_data)

        # 3. Text Processing (Delegated to Handlers)
        sbert_handler = SBERTHandler()
        sbert_feats = sbert_handler.process_data(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        tfidf_handler = TFIDFHandler()
        tfidf_feats = tfidf_handler.process_data(
            train_df, val_df, test_df, load_cached_data=load_cached_data
        )

        # 4. Base Metadata Engineering (Numerical + Ratios)
        raw_num_cols = [
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

        def get_numeric_matrix(df):
            return df[raw_num_cols].values.astype(np.float32)

        X_num_train = get_numeric_matrix(train_df)
        X_num_val = get_numeric_matrix(val_df)
        X_num_test = get_numeric_matrix(test_df)

        # Impute Missing Values (Median)
        imputer = SimpleImputer(strategy="median")
        X_num_train = imputer.fit_transform(X_num_train)
        X_num_val = imputer.transform(X_num_val)
        X_num_test = imputer.transform(X_num_test)

        # Convert back to DataFrame for column-wise engineering
        def to_df(data):
            return pd.DataFrame(data, columns=raw_num_cols)

        df_num_train = to_df(X_num_train)
        df_num_val = to_df(X_num_val)
        df_num_test = to_df(X_num_test)

        def engineer_base_metadata(df):
            # Log transforms (log1p to handle 0)
            df["log_account_age"] = np.log1p(
                df["requester_account_age_in_days_at_request"]
            )
            df["log_num_posts"] = np.log1p(df["requester_number_of_posts_at_request"])
            df["log_num_comments"] = np.log1p(
                df["requester_number_of_comments_at_request"]
            )
            df["log_num_subs"] = np.log1p(
                df["requester_number_of_subreddits_at_request"]
            )

            # Upvote Ratio Calculation
            # plus = up + down, minus = up - down => up = (plus + minus) / 2
            up = (
                df["requester_upvotes_plus_downvotes_at_request"]
                + df["requester_upvotes_minus_downvotes_at_request"]
            ) / 2
            total = df["requester_upvotes_plus_downvotes_at_request"]
            # Avoid division by zero
            ratio = np.divide(up, total, out=np.zeros_like(up), where=total != 0)
            df["upvote_ratio"] = ratio

            return df

        df_eng_train = engineer_base_metadata(df_num_train)
        df_eng_val = engineer_base_metadata(df_num_val)
        df_eng_test = engineer_base_metadata(df_num_test)

        # 5. Top-K Community Indicators
        # Identify Top-K from Train set only
        all_subs = []
        for subs in train_df["requester_subreddits_at_request"]:
            if isinstance(subs, list):
                all_subs.extend(subs)

        top_k_subs = [s for s, c in Counter(all_subs).most_common(self.top_k)]

        def get_top_k_features(df_source):
            n = len(df_source)
            feats = np.zeros((n, self.top_k), dtype=np.float32)
            for i, subs in enumerate(df_source["requester_subreddits_at_request"]):
                if isinstance(subs, list):
                    s_set = set(subs)
                    for j, k_sub in enumerate(top_k_subs):
                        if k_sub in s_set:
                            feats[i, j] = 1.0
            return feats

        top_k_train = get_top_k_features(train_df)
        top_k_val = get_top_k_features(val_df)
        top_k_test = get_top_k_features(test_df)

        # 6. Consistency Scalars (Triple-View)
        def compute_cosine_similarity(a, b):
            # Normalize vectors to unit length
            norm_a = np.linalg.norm(a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(b, axis=1, keepdims=True)
            # Avoid div by zero
            norm_a[norm_a == 0] = 1e-9
            norm_b[norm_b == 0] = 1e-9
            return np.sum((a / norm_a) * (b / norm_b), axis=1, keepdims=True)

        def get_consistency_features(split_name):
            feats = sbert_feats[split_name]
            title = feats["title"]
            body = feats["body"]
            centroid = feats["centroid"]

            sim_title_body = compute_cosine_similarity(title, body)
            sim_title_hist = compute_cosine_similarity(title, centroid)
            sim_body_hist = compute_cosine_similarity(body, centroid)

            # Stack horizontally: (N, 3)
            return np.hstack([sim_title_body, sim_title_hist, sim_body_hist])

        cons_train = get_consistency_features("train")
        cons_val = get_consistency_features("val")
        cons_test = get_consistency_features("test")

        # 7. Interaction Features (Specifically for RF)
        # Drivers: log_account_age, upvote_ratio, log_num_posts
        # Consistency: The 3 scalars computed above

        def get_interactions(df_eng, cons_feats):
            drivers = df_eng[
                ["log_account_age", "upvote_ratio", "log_num_posts"]
            ].values
            interactions = []
            # Compute Cross-Products
            for d_idx in range(drivers.shape[1]):
                for c_idx in range(cons_feats.shape[1]):
                    inter = (
                        drivers[:, d_idx : d_idx + 1] * cons_feats[:, c_idx : c_idx + 1]
                    )
                    interactions.append(inter)
            return np.hstack(interactions)

        inter_train = get_interactions(df_eng_train, cons_train)
        inter_val = get_interactions(df_eng_val, cons_val)
        inter_test = get_interactions(df_eng_test, cons_test)

        # 8. Assemble RF Dense Features
        # Concatenate: [Metadata, Top-K, Consistency, Interactions]
        def assemble_rf_dense(df_eng, top_k, cons, inter):
            return np.hstack([df_eng.values, top_k, cons, inter]).astype(np.float32)

        rf_dense_train = assemble_rf_dense(
            df_eng_train, top_k_train, cons_train, inter_train
        )
        rf_dense_val = assemble_rf_dense(df_eng_val, top_k_val, cons_val, inter_val)
        rf_dense_test = assemble_rf_dense(
            df_eng_test, top_k_test, cons_test, inter_test
        )

        # 9. Assemble MLP Metadata
        # Apply StandardScaler to the engineered metadata.
        # Note: We use df_eng which contains log-transformed features, suitable for MLP.
        scaler = StandardScaler()
        mlp_meta_base_train = scaler.fit_transform(df_eng_train.values)
        mlp_meta_base_val = scaler.transform(df_eng_val.values)
        mlp_meta_base_test = scaler.transform(df_eng_test.values)

        # Concatenate: [Scaled Metadata, Top-K, Consistency]
        # Top-K and Consistency are already in appropriate ranges (0/1, -1/1)
        def assemble_mlp_meta(base, top_k, cons):
            return np.hstack([base, top_k, cons]).astype(np.float32)

        mlp_meta_train = assemble_mlp_meta(mlp_meta_base_train, top_k_train, cons_train)
        mlp_meta_val = assemble_mlp_meta(mlp_meta_base_val, top_k_val, cons_val)
        mlp_meta_test = assemble_mlp_meta(mlp_meta_base_test, top_k_test, cons_test)

        # 10. Package and Save Results
        results = {}
        for split, rf_dense, mlp_meta in zip(
            ["train", "val", "test"],
            [rf_dense_train, rf_dense_val, rf_dense_test],
            [mlp_meta_train, mlp_meta_val, mlp_meta_test],
        ):
            results[split] = {
                "rf": {"dense": rf_dense, "tfidf": tfidf_feats[split]},
                "mlp": {
                    "title": sbert_feats[split]["title"],
                    "body": sbert_feats[split]["body"],
                    "history": sbert_feats[split]["history"],
                    "history_mask": sbert_feats[split]["history_mask"],
                    "centroid": sbert_feats[split]["centroid"],
                    "metadata": mlp_meta,
                },
            }

        self._save_cache(results)
        return results

    def _check_cache(self):
        """Checks if all required cache files exist."""
        for split in ["train", "val", "test"]:
            if not os.path.exists(self.cache_files[split]["rf"]):
                return False
            if not os.path.exists(self.cache_files[split]["mlp"]):
                return False
        return True

    def _save_cache(self, results):
        """Saves dense features to NPZ files."""
        for split in ["train", "val", "test"]:
            # Save RF Dense features
            np.savez_compressed(
                self.cache_files[split]["rf"], dense=results[split]["rf"]["dense"]
            )
            # Save MLP features
            np.savez_compressed(
                self.cache_files[split]["mlp"],
                title=results[split]["mlp"]["title"],
                body=results[split]["mlp"]["body"],
                history=results[split]["mlp"]["history"],
                history_mask=results[split]["mlp"]["history_mask"],
                centroid=results[split]["mlp"]["centroid"],
                metadata=results[split]["mlp"]["metadata"],
            )

    def _load_cache(self):
        """Loads features from NPZ files and re-fetches TF-IDF from its handler."""
        # Load TF-IDF (Sparse) via Handler (it manages its own caching)
        tfidf_handler = TFIDFHandler()
        # We pass None for DFs because we expect the cache to exist
        tfidf_feats = tfidf_handler.process_data(
            None, None, None, load_cached_data=True
        )

        results = {}
        for split in ["train", "val", "test"]:
            rf_data = np.load(self.cache_files[split]["rf"])
            mlp_data = np.load(self.cache_files[split]["mlp"])

            results[split] = {
                "rf": {"dense": rf_data["dense"], "tfidf": tfidf_feats[split]},
                "mlp": {
                    "title": mlp_data["title"],
                    "body": mlp_data["body"],
                    "history": mlp_data["history"],
                    "history_mask": mlp_data["history_mask"],
                    "centroid": mlp_data["centroid"],
                    "metadata": mlp_data["metadata"],
                },
            }
        return results
