import os
import gc
import ast
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy import sparse

from library.config import Config
from library.data_loader import get_common_columns


class FeatureEngineer:
    """
    Handles all feature engineering steps: metadata extraction, text embeddings,
    target encoding, and data preparation for RF and MLP models.
    """

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            stop_words="english",
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.scaler = StandardScaler()
        self.sbert_model = None

    def _get_sbert_model(self):
        """Lazy loader for SBERT model to avoid memory overhead if not needed."""
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(Config.SBERT_MODEL_NAME)
            self.sbert_model.to(Config.DEVICE)
        return self.sbert_model

    def _parse_subreddits(self, df):
        """Parses the stringified list of subreddits safely."""
        col = "requester_subreddits_at_request"
        if col not in df.columns:
            return pd.Series([[]] * len(df), index=df.index)

        # Check if first element is string, if so, parse
        first_val = df[col].iloc[0] if not df.empty else []
        if isinstance(first_val, str):
            return df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else []
            )
        return df[col]

    def extract_metadata(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Extracts numerical metadata, ratios, and text stats.
        Implements caching using Parquet.
        """
        cache_path_train = os.path.join(
            Config.WORKING_DIR, "metadata_features_train.parquet"
        )
        cache_path_val = os.path.join(
            Config.WORKING_DIR, "metadata_features_val.parquet"
        )
        cache_path_test = os.path.join(
            Config.WORKING_DIR, "metadata_features_test.parquet"
        )

        if (
            load_cached_data
            and os.path.exists(cache_path_train)
            and os.path.exists(cache_path_val)
            and os.path.exists(cache_path_test)
        ):
            return (
                pd.read_parquet(cache_path_train),
                pd.read_parquet(cache_path_val),
                pd.read_parquet(cache_path_test),
            )

        def process_df(df):
            # 1. Select safe common numeric columns
            safe_cols = get_common_columns(train_df, test_df)
            numeric_cols = (
                df[safe_cols].select_dtypes(include=[np.number]).columns.tolist()
            )
            meta = df[numeric_cols].copy()

            # 2. Impute NaNs with median
            meta = meta.fillna(meta.median())

            # 3. Engineered Ratios
            up = df.get("requester_upvotes_plus_downvotes_at_request", 0)
            diff = df.get("requester_upvotes_minus_downvotes_at_request", 0)
            # Avoid division by zero
            meta["upvote_ratio"] = diff / (up + 1e-5)

            # 4. Text Stats
            txt = df["request_text_edit_aware"].fillna("").astype(str)
            title = df["request_title"].fillna("").astype(str)

            meta["text_len_char"] = txt.apply(len)
            meta["text_len_word"] = txt.apply(lambda x: len(x.split()))
            meta["title_len_char"] = title.apply(len)

            def get_caps_ratio(s):
                if len(s) == 0:
                    return 0.0
                return sum(1 for c in s if c.isupper()) / len(s)

            meta["text_caps_ratio"] = txt.apply(get_caps_ratio)
            meta["title_caps_ratio"] = title.apply(get_caps_ratio)

            return meta

        meta_train = process_df(train_df)
        meta_val = process_df(val_df)
        meta_test = process_df(test_df)

        # Save cache
        meta_train.to_parquet(cache_path_train)
        meta_val.to_parquet(cache_path_val)
        meta_test.to_parquet(cache_path_test)

        return meta_train, meta_val, meta_test

    def generate_tfidf(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Generates TF-IDF features on concatenated title and body.
        """
        cache_file = os.path.join(Config.WORKING_DIR, "tfidf_features.npz")

        if load_cached_data and os.path.exists(cache_file):
            data = np.load(cache_file)
            return data["train"].item(), data["val"].item(), data["test"].item()

        def get_text(df):
            return (
                df["request_title"].fillna("")
                + " "
                + df["request_text_edit_aware"].fillna("")
            ).astype(str)

        train_text = get_text(train_df)
        val_text = get_text(val_df)
        test_text = get_text(test_df)

        # Fit on Train, Transform all
        X_train = self.tfidf_vectorizer.fit_transform(train_text)
        X_val = self.tfidf_vectorizer.transform(val_text)
        X_test = self.tfidf_vectorizer.transform(test_text)

        # Save
        np.savez(cache_file, train=X_train, val=X_val, test=X_test)

        return X_train, X_val, X_test

    def generate_sbert_embeddings(
        self, train_df, val_df, test_df, load_cached_data=True
    ):
        """
        Generates SBERT embeddings for requests and user history.
        Returns:
            request_embs: Tuple of (train, val, test) arrays (N, 384)
            history_embs: Tuple of (train, val, test) arrays (N, Max_Len, 384)
        """
        cache_req = os.path.join(Config.WORKING_DIR, "sbert_request.npz")
        cache_hist = os.path.join(Config.WORKING_DIR, "sbert_history.npz")

        if (
            load_cached_data
            and os.path.exists(cache_req)
            and os.path.exists(cache_hist)
        ):
            req_data = np.load(cache_req)
            hist_data = np.load(cache_hist)
            return (
                (req_data["train"], req_data["val"], req_data["test"]),
                (hist_data["train"], hist_data["val"], hist_data["test"]),
            )

        model = self._get_sbert_model()

        # 1. Request Embeddings
        def encode_requests(df):
            texts = (
                (
                    df["request_title"].fillna("")
                    + " "
                    + df["request_text_edit_aware"].fillna("")
                )
                .astype(str)
                .tolist()
            )
            return model.encode(
                texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )

        req_train = encode_requests(train_df)
        req_val = encode_requests(val_df)
        req_test = encode_requests(test_df)

        # 2. History Embeddings
        # Extract all unique subreddits across datasets
        all_subreddits = set()
        for df in [train_df, val_df, test_df]:
            subs_series = self._parse_subreddits(df)
            for subs in subs_series:
                all_subreddits.update(subs)

        sorted_subs = sorted(list(all_subreddits))
        sub_to_idx = {sub: i for i, sub in enumerate(sorted_subs)}

        # Encode unique subreddits
        if not sorted_subs:
            sub_embeddings = np.zeros((1, Config.SBERT_EMBEDDING_DIM), dtype=np.float32)
        else:
            sub_embeddings = model.encode(
                sorted_subs,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        def encode_history(df):
            subs_series = self._parse_subreddits(df)
            batch_size = len(df)
            # Determine max length for padding
            lens = [len(s) for s in subs_series]
            max_len = max(lens) if lens else 0
            if max_len == 0:
                max_len = 1

            # Create padded array
            padded_emb = np.zeros(
                (batch_size, max_len, Config.SBERT_EMBEDDING_DIM), dtype=np.float32
            )

            for i, subs in enumerate(subs_series):
                if not subs:
                    continue
                indices = [sub_to_idx[s] for s in subs if s in sub_to_idx]
                if not indices:
                    continue
                indices = indices[:max_len]
                padded_emb[i, : len(indices), :] = sub_embeddings[indices]

            return padded_emb

        hist_train = encode_history(train_df)
        hist_val = encode_history(val_df)
        hist_test = encode_history(test_df)

        # Save
        np.savez(cache_req, train=req_train, val=req_val, test=req_test)
        np.savez(cache_hist, train=hist_train, val=hist_val, test=hist_test)

        return (req_train, req_val, req_test), (hist_train, hist_val, hist_test)

    def compute_target_encoding(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Computes Aggregated Target Encoding for subreddits.
        Uses K-Fold for Train to prevent leakage.
        """
        cache_path_train = os.path.join(Config.WORKING_DIR, "te_features_train.parquet")
        cache_path_val = os.path.join(Config.WORKING_DIR, "te_features_val.parquet")
        cache_path_test = os.path.join(Config.WORKING_DIR, "te_features_test.parquet")

        if (
            load_cached_data
            and os.path.exists(cache_path_train)
            and os.path.exists(cache_path_val)
            and os.path.exists(cache_path_test)
        ):
            return (
                pd.read_parquet(cache_path_train),
                pd.read_parquet(cache_path_val),
                pd.read_parquet(cache_path_test),
            )

        # Helper to explode
        def get_exploded(df):
            df_temp = df[["request_id"]].copy()
            df_temp["subs"] = self._parse_subreddits(df)
            return df_temp.explode("subs")

        # 1. Train (K-Fold)
        train_te_features = pd.DataFrame(index=train_df.index)
        # Initialize columns
        for col in ["te_mean", "te_max", "te_min"]:
            train_te_features[col] = np.nan

        kf = KFold(
            n_splits=Config.TARGET_ENCODING_FOLDS,
            shuffle=True,
            random_state=Config.RANDOM_SEED,
        )
        global_mean = train_df["requester_received_pizza"].mean()
        alpha = 10  # Smoothing factor

        for train_idx, holdout_idx in kf.split(train_df):
            fold_train = train_df.iloc[train_idx]
            fold_holdout = train_df.iloc[holdout_idx]

            exploded = get_exploded(fold_train)
            exploded["target"] = fold_train.loc[
                exploded.index, "requester_received_pizza"
            ].values

            sub_stats = exploded.groupby("subs")["target"].agg(["mean", "count"])
            sub_stats["smoothed_rate"] = (
                sub_stats["mean"] * sub_stats["count"] + global_mean * alpha
            ) / (sub_stats["count"] + alpha)
            mapper = sub_stats["smoothed_rate"].to_dict()

            holdout_exploded = get_exploded(fold_holdout)
            holdout_exploded["rate"] = (
                holdout_exploded["subs"].map(mapper).fillna(global_mean)
            )

            agg = holdout_exploded.groupby(holdout_exploded.index)["rate"].agg(
                ["mean", "max", "min"]
            )

            train_te_features.loc[holdout_idx, "te_mean"] = agg["mean"]
            train_te_features.loc[holdout_idx, "te_max"] = agg["max"]
            train_te_features.loc[holdout_idx, "te_min"] = agg["min"]

        train_te_features = train_te_features.fillna(global_mean)

        # 2. Val/Test (Global Map)
        exploded_full = get_exploded(train_df)
        exploded_full["target"] = train_df.loc[
            exploded_full.index, "requester_received_pizza"
        ].values
        sub_stats_full = exploded_full.groupby("subs")["target"].agg(["mean", "count"])
        sub_stats_full["smoothed_rate"] = (
            sub_stats_full["mean"] * sub_stats_full["count"] + global_mean * alpha
        ) / (sub_stats_full["count"] + alpha)
        global_mapper = sub_stats_full["smoothed_rate"].to_dict()

        def apply_mapping(df):
            ex = get_exploded(df)
            ex["rate"] = ex["subs"].map(global_mapper).fillna(global_mean)
            agg = ex.groupby(ex.index)["rate"].agg(["mean", "max", "min"])
            agg = agg.reindex(df.index, fill_value=global_mean)
            return agg.rename(
                columns={"mean": "te_mean", "max": "te_max", "min": "te_min"}
            )

        val_te_features = apply_mapping(val_df)
        test_te_features = apply_mapping(test_df)

        # Save
        train_te_features.to_parquet(cache_path_train)
        val_te_features.to_parquet(cache_path_val)
        test_te_features.to_parquet(cache_path_test)

        return train_te_features, val_te_features, test_te_features

    def compute_consistency_score(self, req_embs, hist_embs):
        """
        Computes Cosine Similarity between Request Embedding and History Centroid.
        """
        # Sum over sequence length (axis 1)
        sum_hist = np.sum(hist_embs, axis=1)  # (N, 384)

        # Count non-zero vectors
        is_non_zero = np.any(hist_embs != 0, axis=2)  # (N, Max_Len)
        count = np.sum(is_non_zero, axis=1, keepdims=True)  # (N, 1)
        count[count == 0] = 1  # Avoid div by zero

        centroids = sum_hist / count  # (N, 384)

        # Cosine Similarity
        norm_req = np.linalg.norm(req_embs, axis=1)
        norm_cen = np.linalg.norm(centroids, axis=1)

        norm_req[norm_req == 0] = 1e-9
        norm_cen[norm_cen == 0] = 1e-9

        dot = np.sum(req_embs * centroids, axis=1)
        scores = dot / (norm_req * norm_cen)

        return pd.DataFrame({"consistency_score": scores})

    def prepare_rf_inputs(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Assembles inputs for Random Forest: Metadata + Target Encoding + Consistency + TF-IDF.
        """
        # 1. Metadata
        meta_train, meta_val, meta_test = self.extract_metadata(
            train_df, val_df, test_df, load_cached_data
        )

        # 2. Target Encoding
        te_train, te_val, te_test = self.compute_target_encoding(
            train_df, val_df, test_df, load_cached_data
        )

        # 3. TF-IDF
        tfidf_train, tfidf_val, tfidf_test = self.generate_tfidf(
            train_df, val_df, test_df, load_cached_data
        )

        # 4. Consistency Score
        (req_tr, req_val, req_te), (hist_tr, hist_val, hist_te) = (
            self.generate_sbert_embeddings(train_df, val_df, test_df, load_cached_data)
        )
        cons_train = self.compute_consistency_score(req_tr, hist_tr)
        cons_val = self.compute_consistency_score(req_val, hist_val)
        cons_test = self.compute_consistency_score(req_te, hist_te)

        # Combine Dense Features
        def combine_dense(meta, te, cons):
            return pd.concat([meta, te, cons], axis=1).values.astype(np.float32)

        dense_train = combine_dense(meta_train, te_train, cons_train)
        dense_val = combine_dense(meta_val, te_val, cons_val)
        dense_test = combine_dense(meta_test, te_test, cons_test)

        # Combine with Sparse TF-IDF
        X_train = sparse.hstack([dense_train, tfidf_train])
        X_val = sparse.hstack([dense_val, tfidf_val])
        X_test = sparse.hstack([dense_test, tfidf_test])

        # Targets
        y_train = train_df["requester_received_pizza"].astype(int).values
        y_val = val_df["requester_received_pizza"].astype(int).values

        return (X_train, y_train), (X_val, y_val), X_test

    def prepare_mlp_inputs(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Assembles inputs for MLP: Scaled Metadata + Raw Embeddings.
        """
        # 1. Metadata
        meta_train, meta_val, meta_test = self.extract_metadata(
            train_df, val_df, test_df, load_cached_data
        )

        # Apply Arcsinh + Standard Scaling
        meta_train = np.arcsinh(meta_train)
        meta_val = np.arcsinh(meta_val)
        meta_test = np.arcsinh(meta_test)

        meta_train = self.scaler.fit_transform(meta_train)
        meta_val = self.scaler.transform(meta_val)
        meta_test = self.scaler.transform(meta_test)

        # 2. SBERT Embeddings
        (req_tr, req_val, req_te), (hist_tr, hist_val, hist_te) = (
            self.generate_sbert_embeddings(train_df, val_df, test_df, load_cached_data)
        )

        # Targets
        y_train = train_df["requester_received_pizza"].astype(int).values
        y_val = val_df["requester_received_pizza"].astype(int).values

        # Pack into dicts
        train_data = {
            "metadata": meta_train.astype(np.float32),
            "request_emb": req_tr.astype(np.float32),
            "history_emb": hist_tr.astype(np.float32),
            "y": y_train,
        }

        val_data = {
            "metadata": meta_val.astype(np.float32),
            "request_emb": req_val.astype(np.float32),
            "history_emb": hist_val.astype(np.float32),
            "y": y_val,
        }

        test_data = {
            "metadata": meta_test.astype(np.float32),
            "request_emb": req_te.astype(np.float32),
            "history_emb": hist_te.astype(np.float32),
        }

        return train_data, val_data, test_data
