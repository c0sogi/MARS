import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from library.config import Config, get_sbert_embeddings, parse_list_col


class FeatureEngineer:
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.top_k = Config.TOP_K_SUBREDDITS
        self.sbert_model = Config.SBERT_MODEL_NAME
        self.text_col = Config.TEXT_COL
        self.title_col = Config.TITLE_COL

        # Base metadata columns to use
        self.meta_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "requester_upvotes_plus_downvotes_at_request",  # Added for ratio calc
        ]

    def _fill_text_nans(self, df):
        df[self.text_col] = df[self.text_col].fillna("")
        df[self.title_col] = df[self.title_col].fillna("")
        return df

    def process_metadata(self, df_train, df_val, df_test):
        """
        Generates raw metadata with ratios for RF, and scaled metadata for MLP.
        """
        dfs = [df_train, df_val, df_test]

        # 1. Basic Imputation & Feature Engineering
        for df in dfs:
            # Fill NaNs in numeric cols
            for col in self.meta_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            # Engineered Ratios
            # Upvote Ratio: (Up - Down) / (Up + Down) -> approximate via provided columns
            # We have minus_downvotes (net) and plus_downvotes (total)
            # Avoid division by zero
            df["upvote_ratio"] = df["requester_upvotes_minus_downvotes_at_request"] / (
                df["requester_upvotes_plus_downvotes_at_request"] + 1e-5
            )

            # Interaction Rate: Comments / Posts
            df["interaction_ratio"] = df["requester_number_of_comments_at_request"] / (
                df["requester_number_of_posts_at_request"] + 1e-5
            )

            # Text Meta-Features
            df["text_len_char"] = df[self.text_col].astype(str).apply(len)
            df["title_len_char"] = df[self.title_col].astype(str).apply(len)
            df["text_len_word"] = (
                df[self.text_col].astype(str).apply(lambda x: len(x.split()))
            )

        # Select columns for RF (Raw + Engineered)
        # We exclude the 'plus_downvotes' as it's highly collinear with 'minus' and we have the ratio now
        rf_cols = [
            "requester_account_age_in_days_at_request",
            "requester_days_since_first_post_on_raop_at_request",
            "requester_number_of_comments_at_request",
            "requester_number_of_posts_at_request",
            "requester_upvotes_minus_downvotes_at_request",
            "upvote_ratio",
            "interaction_ratio",
            "text_len_char",
            "title_len_char",
            "text_len_word",
        ]

        X_meta_rf_train = df_train[rf_cols].values.astype(np.float32)
        X_meta_rf_val = df_val[rf_cols].values.astype(np.float32)
        X_meta_rf_test = df_test[rf_cols].values.astype(np.float32)

        # 2. Scaling for MLP (Arcsinh + Standard)
        # We use the same columns as RF for consistency in information, but transformed
        scaler = StandardScaler()

        # Apply Arcsinh to handle heavy tails before scaling
        X_meta_mlp_train = np.arcsinh(X_meta_rf_train)
        X_meta_mlp_val = np.arcsinh(X_meta_rf_val)
        X_meta_mlp_test = np.arcsinh(X_meta_rf_test)

        scaler.fit(X_meta_mlp_train)
        X_meta_mlp_train = scaler.transform(X_meta_mlp_train)
        X_meta_mlp_val = scaler.transform(X_meta_mlp_val)
        X_meta_mlp_test = scaler.transform(X_meta_mlp_test)

        return (X_meta_rf_train, X_meta_rf_val, X_meta_rf_test), (
            X_meta_mlp_train,
            X_meta_mlp_val,
            X_meta_mlp_test,
        )

    def process_community_top_k(self, df_train, df_val, df_test):
        """
        Generates binary indicators for top-K subreddits found in train.
        """
        # Parse subreddits if not already lists
        for df in [df_train, df_val, df_test]:
            if isinstance(df["requester_subreddits_at_request"].iloc[0], str):
                df["subreddits"] = df["requester_subreddits_at_request"].apply(
                    parse_list_col
                )
            else:
                df["subreddits"] = df["requester_subreddits_at_request"]

        # Identify Top K in Train
        all_subs = [sub for subs in df_train["subreddits"] for sub in subs]
        if all_subs:
            top_subs = (
                pd.Series(all_subs).value_counts().head(self.top_k).index.tolist()
            )
        else:
            top_subs = []

        # Generate Flags
        def get_flags(df):
            flags = np.zeros((len(df), len(top_subs)), dtype=np.int32)
            for i, row_subs in enumerate(df["subreddits"]):
                row_set = set(row_subs)
                for j, target_sub in enumerate(top_subs):
                    if target_sub in row_set:
                        flags[i, j] = 1
            return flags

        X_topk_train = get_flags(df_train)
        X_topk_val = get_flags(df_val)
        X_topk_test = get_flags(df_test)

        return X_topk_train, X_topk_val, X_topk_test

    def process_text_tfidf(self, df_train, df_val, df_test):
        """
        Generates TF-IDF features for RF.
        """
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

        # Combine title + body
        train_text = df_train[self.title_col] + " " + df_train[self.text_col]
        val_text = df_val[self.title_col] + " " + df_val[self.text_col]
        test_text = df_test[self.title_col] + " " + df_test[self.text_col]

        X_tfidf_train = tfidf.fit_transform(train_text).toarray().astype(np.float32)
        X_tfidf_val = tfidf.transform(val_text).toarray().astype(np.float32)
        X_tfidf_test = tfidf.transform(test_text).toarray().astype(np.float32)

        return X_tfidf_train, X_tfidf_val, X_tfidf_test

    def process_text_sbert(self, df_train, df_val, df_test, load_cached_data):
        """
        Generates SBERT embeddings for Title, Body, and History (Sequence).
        """
        # 1. Title Embeddings
        title_emb_train = get_sbert_embeddings(
            df_train[self.title_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "title_train.npy"),
            load_cached_data,
        )
        title_emb_val = get_sbert_embeddings(
            df_val[self.title_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "title_val.npy"),
            load_cached_data,
        )
        title_emb_test = get_sbert_embeddings(
            df_test[self.title_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "title_test.npy"),
            load_cached_data,
        )

        # 2. Body Embeddings
        body_emb_train = get_sbert_embeddings(
            df_train[self.text_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "body_train.npy"),
            load_cached_data,
        )
        body_emb_val = get_sbert_embeddings(
            df_val[self.text_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "body_val.npy"),
            load_cached_data,
        )
        body_emb_test = get_sbert_embeddings(
            df_test[self.text_col].tolist(),
            self.sbert_model,
            os.path.join(self.cache_dir, "body_test.npy"),
            load_cached_data,
        )

        # 3. History Sequence Processing
        # Collect all unique subreddits to build vocabulary
        all_subs_lists = (
            df_train["subreddits"].tolist()
            + df_val["subreddits"].tolist()
            + df_test["subreddits"].tolist()
        )
        unique_subs = set(s for subs in all_subs_lists for s in subs)
        unique_subs = sorted(list(unique_subs))

        if not unique_subs:
            unique_subs = ["placeholder"]

        # Embed unique subreddits
        sub_embeddings = get_sbert_embeddings(
            unique_subs,
            self.sbert_model,
            os.path.join(self.cache_dir, "sub_embeddings.npy"),
            load_cached_data,
        )

        # Add padding embedding (index 0) -> zeros
        sub_embeddings_pad = np.vstack(
            [np.zeros((1, sub_embeddings.shape[1])), sub_embeddings]
        )

        # Mapping: Subreddit -> Index (1-based, 0 is padding)
        sub_to_idx = {sub: i + 1 for i, sub in enumerate(unique_subs)}

        def encode_history(subs_list, max_len=20):
            indices = [sub_to_idx.get(s, 0) for s in subs_list[:max_len]]
            if len(indices) < max_len:
                indices += [0] * (max_len - len(indices))
            return indices

        hist_idx_train = np.array([encode_history(s) for s in df_train["subreddits"]])
        hist_idx_val = np.array([encode_history(s) for s in df_val["subreddits"]])
        hist_idx_test = np.array([encode_history(s) for s in df_test["subreddits"]])

        return (
            (title_emb_train, title_emb_val, title_emb_test),
            (body_emb_train, body_emb_val, body_emb_test),
            (hist_idx_train, hist_idx_val, hist_idx_test),
            sub_embeddings_pad,
        )

    def run(self, load_cached_data=True):
        """
        Orchestrates the feature engineering process with caching.
        """
        # Cache paths for aggregated datasets
        rf_cache_path = os.path.join(self.cache_dir, "rf_data.npz")
        mlp_cache_path = os.path.join(self.cache_dir, "mlp_data.npz")
        ids_cache_path = os.path.join(self.cache_dir, "ids.npy")

        # Check if main cache files exist
        if (
            load_cached_data
            and os.path.exists(rf_cache_path)
            and os.path.exists(mlp_cache_path)
        ):
            print("Loading features from cache...")
            rf_data = np.load(rf_cache_path)
            mlp_data = np.load(mlp_cache_path)
            ids = np.load(ids_cache_path, allow_pickle=True)

            return {
                "rf": (
                    rf_data["X_train"],
                    rf_data["y_train"],
                    rf_data["X_val"],
                    rf_data["y_val"],
                    rf_data["X_test"],
                ),
                "mlp": {
                    "train": (
                        mlp_data["title_train"],
                        mlp_data["body_train"],
                        mlp_data["hist_train"],
                        mlp_data["meta_train"],
                        mlp_data["y_train"],
                    ),
                    "val": (
                        mlp_data["title_val"],
                        mlp_data["body_val"],
                        mlp_data["hist_val"],
                        mlp_data["meta_val"],
                        mlp_data["y_val"],
                    ),
                    "test": (
                        mlp_data["title_test"],
                        mlp_data["body_test"],
                        mlp_data["hist_test"],
                        mlp_data["meta_test"],
                    ),
                    "sub_emb": mlp_data["sub_emb"],
                },
                "ids": ids,
            }

        print("Computing features from scratch...")

        # Load raw data
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df_test = pd.read_csv(Config.TEST_PATH)

        # Pre-cleaning
        df_train = self._fill_text_nans(df_train)
        df_val = self._fill_text_nans(df_val)
        df_test = self._fill_text_nans(df_test)

        # 1. Process Metadata (RF Raw + MLP Scaled)
        (meta_rf_tr, meta_rf_val, meta_rf_te), (
            meta_mlp_tr,
            meta_mlp_val,
            meta_mlp_te,
        ) = self.process_metadata(df_train, df_val, df_test)

        # 2. Process Community Top-K (RF)
        topk_tr, topk_val, topk_te = self.process_community_top_k(
            df_train, df_val, df_test
        )

        # 3. Process TF-IDF (RF)
        tfidf_tr, tfidf_val, tfidf_te = self.process_text_tfidf(
            df_train, df_val, df_test
        )

        # 4. Process SBERT (MLP)
        (
            (title_tr, title_val, title_te),
            (body_tr, body_val, body_te),
            (hist_tr, hist_val, hist_te),
            sub_emb,
        ) = self.process_text_sbert(df_train, df_val, df_test, load_cached_data)

        # 5. Assemble RF Data
        X_rf_train = np.hstack([tfidf_tr, topk_tr, meta_rf_tr])
        X_rf_val = np.hstack([tfidf_val, topk_val, meta_rf_val])
        X_rf_test = np.hstack([tfidf_te, topk_te, meta_rf_te])

        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values
        ids = df_test["request_id"].values

        # Save to Cache
        np.savez(
            rf_cache_path,
            X_train=X_rf_train,
            y_train=y_train,
            X_val=X_rf_val,
            y_val=y_val,
            X_test=X_rf_test,
        )

        np.savez(
            mlp_cache_path,
            title_train=title_tr,
            body_train=body_tr,
            hist_train=hist_tr,
            meta_train=meta_mlp_tr,
            y_train=y_train,
            title_val=title_val,
            body_val=body_val,
            hist_val=hist_val,
            meta_val=meta_mlp_val,
            y_val=y_val,
            title_test=title_te,
            body_test=body_te,
            hist_test=hist_te,
            meta_test=meta_mlp_te,
            sub_emb=sub_emb,
        )

        np.save(ids_cache_path, ids)

        return {
            "rf": (X_rf_train, y_train, X_rf_val, y_val, X_rf_test),
            "mlp": {
                "train": (title_tr, body_tr, hist_tr, meta_mlp_tr, y_train),
                "val": (title_val, body_val, hist_val, meta_mlp_val, y_val),
                "test": (title_te, body_te, hist_te, meta_mlp_te),
                "sub_emb": sub_emb,
            },
            "ids": ids,
        }
