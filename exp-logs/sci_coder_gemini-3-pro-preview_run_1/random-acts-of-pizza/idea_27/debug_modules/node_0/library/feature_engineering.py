import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch
from collections import Counter
from library import config, utils


class FeaturePipeline:
    def __init__(self):
        """
        Initializes the FeaturePipeline with cache paths.
        """
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rf_cache_path = os.path.join(self.cache_dir, "rf_data.npz")
        self.mlp_cache_path = os.path.join(self.cache_dir, "mlp_data.npz")

    def _process_metadata(self, df_train, df_val, df_test):
        """
        Processes numerical metadata: selection, ratio engineering, imputation, scaling.
        Returns:
            - Imputed metadata (for RF)
            - Scaled metadata (for MLP)
        """
        # 1. Identify candidate numeric columns (intersection of train and test)
        test_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()
        train_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        common_cols = [c for c in test_cols if c in train_cols]

        # 2. Exclude ID, Target, and known leakage/irrelevant columns
        exclude = [
            "requester_received_pizza",
            "request_id",
            "unix_timestamp_of_request",  # Redundant/Local time
        ]
        feature_cols = [c for c in common_cols if c not in exclude]

        # 3. Extract raw matrices
        X_train = df_train[feature_cols].copy()
        X_val = df_val[feature_cols].copy()
        X_test = df_test[feature_cols].copy()

        # 4. Feature Engineering: Ratios
        def create_ratio(df, num_col, den_col, name):
            if num_col in df.columns and den_col in df.columns:
                # Add small epsilon to denominator
                df[name] = df[num_col] / (df[den_col] + 1e-5)

        for df in [X_train, X_val, X_test]:
            # Upvote Ratio: u / (u + d)
            if (
                "requester_upvotes_plus_downvotes_at_request" in df.columns
                and "requester_upvotes_minus_downvotes_at_request" in df.columns
            ):
                # u = (sum + diff) / 2
                u = (
                    df["requester_upvotes_plus_downvotes_at_request"]
                    + df["requester_upvotes_minus_downvotes_at_request"]
                ) / 2
                df["upvote_ratio"] = u / (
                    df["requester_upvotes_plus_downvotes_at_request"] + 1e-5
                )

            # RAOP Activity Ratios
            create_ratio(
                df,
                "requester_number_of_comments_in_raop_at_request",
                "requester_number_of_comments_at_request",
                "raop_comment_ratio",
            )
            create_ratio(
                df,
                "requester_number_of_posts_on_raop_at_request",
                "requester_number_of_posts_at_request",
                "raop_post_ratio",
            )

        # 5. Text Meta-Features (Length, Caps)
        for X, original_df in zip(
            [X_train, X_val, X_test], [df_train, df_val, df_test]
        ):
            title = original_df["request_title"].fillna("").astype(str)
            body = original_df["request_text_edit_aware"].fillna("").astype(str)

            X["title_len_char"] = title.apply(len)
            X["body_len_char"] = body.apply(len)
            X["title_len_word"] = title.apply(lambda x: len(x.split()))
            X["body_len_word"] = body.apply(lambda x: len(x.split()))

            def caps_ratio(s):
                return sum(1 for c in s if c.isupper()) / (len(s) + 1e-5)

            X["title_caps_ratio"] = title.apply(caps_ratio)
            X["body_caps_ratio"] = body.apply(caps_ratio)

        # 6. Imputation (Median) - For RF
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        # 7. Scaling (Arcsinh + Standard) - For MLP
        X_train_arc = utils.arcsinh_scale(X_train_imp)
        X_val_arc = utils.arcsinh_scale(X_val_imp)
        X_test_arc = utils.arcsinh_scale(X_test_imp)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arc)
        X_val_scaled = scaler.transform(X_val_arc)
        X_test_scaled = scaler.transform(X_test_arc)

        return (
            X_train_imp,
            X_val_imp,
            X_test_imp,
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
        )

    def _process_tfidf(self, df_train, df_val, df_test):
        """
        Generates High-Fidelity TF-IDF features (Dense).
        """

        def get_text(df):
            t = df["request_title"].fillna("").astype(str)
            b = df["request_text_edit_aware"].fillna("").astype(str)
            return t + " " + b

        train_text = get_text(df_train)
        val_text = get_text(df_val)
        test_text = get_text(df_test)

        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_VOCAB_SIZE,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        # Return as dense float32 arrays
        X_train = vectorizer.fit_transform(train_text).toarray().astype(np.float32)
        X_val = vectorizer.transform(val_text).toarray().astype(np.float32)
        X_test = vectorizer.transform(test_text).toarray().astype(np.float32)

        return X_train, X_val, X_test

    def _process_top_k(self, df_train, df_val, df_test):
        """
        Generates binary indicators for top K subreddits.
        """
        # Flatten list of subreddits in train to count frequencies
        all_subs = []
        for subs in df_train["requester_subreddits_at_request"]:
            all_subs.extend(subs)

        counts = Counter(all_subs)
        top_k = [sub for sub, _ in counts.most_common(config.TOP_K_SUBREDDITS)]

        def get_indicators(df):
            matrix = np.zeros((len(df), len(top_k)), dtype=np.float32)
            for i, subs in enumerate(df["requester_subreddits_at_request"]):
                current_subs = set(subs)
                for j, target_sub in enumerate(top_k):
                    if target_sub in current_subs:
                        matrix[i, j] = 1.0
            return matrix

        X_train = get_indicators(df_train)
        X_val = get_indicators(df_val)
        X_test = get_indicators(df_test)

        return X_train, X_val, X_test

    def _process_embeddings(self, df_train, df_val, df_test):
        """
        Generates SBERT embeddings for Title, Body, and History Sequences.
        """
        model = SentenceTransformer(config.SBERT_MODEL_NAME, device=config.DEVICE)

        def embed_col(col_name):
            t_train = df_train[col_name].fillna("").astype(str).tolist()
            t_val = df_val[col_name].fillna("").astype(str).tolist()
            t_test = df_test[col_name].fillna("").astype(str).tolist()

            e_train = model.encode(
                t_train, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            e_val = model.encode(
                t_val, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            e_test = model.encode(
                t_test, batch_size=64, show_progress_bar=False, convert_to_numpy=True
            )
            return e_train, e_val, e_test

        # 1. Title & Body Embeddings
        title_train, title_val, title_test = embed_col("request_title")
        body_train, body_val, body_test = embed_col("request_text_edit_aware")

        # 2. History Sequences
        # Collect all unique subreddits to batch embed
        all_subs = set()
        for df in [df_train, df_val, df_test]:
            for subs in df["requester_subreddits_at_request"]:
                all_subs.update(subs)

        unique_subs = list(all_subs)
        if not unique_subs:
            unique_subs = [""]

        sub_embeddings = model.encode(
            unique_subs, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        sub_to_emb = {sub: emb for sub, emb in zip(unique_subs, sub_embeddings)}

        # Determine max sequence length (observed max)
        max_len = 0
        for df in [df_train, df_val, df_test]:
            m = df["requester_subreddits_at_request"].apply(len).max()
            if m > max_len:
                max_len = m

        embedding_dim = config.SBERT_EMBEDDING_DIM

        def create_seq_tensor(df):
            num_samples = len(df)
            # Shape: (N, L, D)
            seqs = np.zeros((num_samples, max_len, embedding_dim), dtype=np.float32)
            # Mask: (N, L) - 1 for valid, 0 for padding
            masks = np.zeros((num_samples, max_len), dtype=np.float32)

            for i, subs in enumerate(df["requester_subreddits_at_request"]):
                current_subs = subs[:max_len]
                for j, sub in enumerate(current_subs):
                    if sub in sub_to_emb:
                        seqs[i, j, :] = sub_to_emb[sub]
                        masks[i, j] = 1.0
            return seqs, masks

        hist_train, mask_train = create_seq_tensor(df_train)
        hist_val, mask_val = create_seq_tensor(df_val)
        hist_test, mask_test = create_seq_tensor(df_test)

        return (
            (title_train, title_val, title_test),
            (body_train, body_val, body_test),
            (hist_train, hist_val, hist_test),
            (mask_train, mask_val, mask_test),
        )

    def run(self, load_cached_data=True):
        """
        Main execution method.
        Checks cache, loads data, computes features, saves cache, returns data dicts.
        """
        # 1. Check Cache
        if (
            load_cached_data
            and os.path.exists(self.rf_cache_path)
            and os.path.exists(self.mlp_cache_path)
        ):
            print("Loading cached features...")
            try:
                rf_data = dict(np.load(self.rf_cache_path, allow_pickle=True))
                mlp_data = dict(np.load(self.mlp_cache_path, allow_pickle=True))
                return rf_data, mlp_data
            except Exception as e:
                print(f"Cache load failed: {e}. Recomputing...")

        print("Computing features from scratch...")

        # 2. Load Data
        df_train = utils.load_dataset(
            config.TRAIN_DATA_PATH, list_columns=["requester_subreddits_at_request"]
        )
        df_val = utils.load_dataset(
            config.VAL_DATA_PATH, list_columns=["requester_subreddits_at_request"]
        )
        df_test = utils.load_dataset(
            config.TEST_DATA_PATH, list_columns=["requester_subreddits_at_request"]
        )

        # Debugging sample
        if config.DATA_SAMPLE_SIZE:
            print(f"DEBUG: Subsampling to {config.DATA_SAMPLE_SIZE}")
            df_train = df_train.head(config.DATA_SAMPLE_SIZE)
            df_val = df_val.head(config.DATA_SAMPLE_SIZE)
            df_test = df_test.head(config.DATA_SAMPLE_SIZE)

        # Labels
        y_train = df_train["requester_received_pizza"].astype(int).values
        y_val = df_val["requester_received_pizza"].astype(int).values

        # 3. Compute Features
        print("Processing Metadata...")
        (
            meta_imp_train,
            meta_imp_val,
            meta_imp_test,
            meta_scl_train,
            meta_scl_val,
            meta_scl_test,
        ) = self._process_metadata(df_train, df_val, df_test)

        print("Processing TF-IDF...")
        tfidf_train, tfidf_val, tfidf_test = self._process_tfidf(
            df_train, df_val, df_test
        )

        print("Processing Top-K Subreddits...")
        topk_train, topk_val, topk_test = self._process_top_k(df_train, df_val, df_test)

        print("Processing SBERT Embeddings...")
        (
            (tit_tr, tit_val, tit_te),
            (bod_tr, bod_val, bod_te),
            (hist_tr, hist_val, hist_te),
            (mask_tr, mask_val, mask_te),
        ) = self._process_embeddings(df_train, df_val, df_test)

        # 4. Assemble & Save

        # RF Data: Concat [Meta_Imp, TFIDF, TopK]
        rf_X_train = np.hstack([meta_imp_train, tfidf_train, topk_train])
        rf_X_val = np.hstack([meta_imp_val, tfidf_val, topk_val])
        rf_X_test = np.hstack([meta_imp_test, tfidf_test, topk_test])

        rf_data = {
            "X_train": rf_X_train,
            "y_train": y_train,
            "X_val": rf_X_val,
            "y_val": y_val,
            "X_test": rf_X_test,
        }

        # MLP Data: Dictionary of components
        mlp_data = {
            "meta_train": meta_scl_train,
            "title_train": tit_tr,
            "body_train": bod_tr,
            "hist_train": hist_tr,
            "mask_train": mask_tr,
            "y_train": y_train,
            "meta_val": meta_scl_val,
            "title_val": tit_val,
            "body_val": bod_val,
            "hist_val": hist_val,
            "mask_val": mask_val,
            "y_val": y_val,
            "meta_test": meta_scl_test,
            "title_test": tit_te,
            "body_test": bod_te,
            "hist_test": hist_te,
            "mask_test": mask_te,
        }

        print("Saving to cache...")
        np.savez_compressed(self.rf_cache_path, **rf_data)
        np.savez_compressed(self.mlp_cache_path, **mlp_data)

        return rf_data, mlp_data
