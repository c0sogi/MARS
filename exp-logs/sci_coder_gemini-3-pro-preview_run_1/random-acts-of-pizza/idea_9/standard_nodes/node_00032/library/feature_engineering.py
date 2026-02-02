import os
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import set_seed


class SemanticProcessor:
    """
    Handles generation of semantic embeddings using SBERT for:
    1. Request Text (Sentence Embeddings)
    2. Subreddit History (Sequence of Embeddings)
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.model_name = Config.SBERT_MODEL_NAME
        self.model = None

    def _load_model(self):
        """Lazy loading of the SBERT model."""
        if self.model is None:
            print(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def process_data(self, df_train, df_val, df_test, load_cached_data=True):
        """
        Generates or loads semantic features for all splits.

        Returns:
            tuple: (train_text, train_subs, val_text, val_subs, test_text, test_subs)
            - text: (N, 384) numpy array
            - subs: (N, MAX_SEQ_LEN, 384) numpy array
        """
        set_seed()
        cache_path = os.path.join(Config.WORKING_DIR, "semantic_features.npz")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached semantic features from {cache_path}")
            try:
                data = np.load(cache_path)
                return (
                    data["train_text"],
                    data["train_subs"],
                    data["val_text"],
                    data["val_subs"],
                    data["test_text"],
                    data["test_subs"],
                )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute Features
        self._load_model()
        print("Generating semantic features (this may take a while)...")

        train_text, train_subs = self._process_split(df_train)
        val_text, val_subs = self._process_split(df_val)
        test_text, test_subs = self._process_split(df_test)

        # 3. Save to Cache
        print(f"Saving semantic features to {cache_path}")
        np.savez(
            cache_path,
            train_text=train_text,
            train_subs=train_subs,
            val_text=val_text,
            val_subs=val_subs,
            test_text=test_text,
            test_subs=test_subs,
        )

        return train_text, train_subs, val_text, val_subs, test_text, test_subs

    def _process_split(self, df):
        """
        Processes a single dataframe split.
        """
        # A. Request Text Embeddings
        # Fill NaNs with empty string
        texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=Config.BATCH_SIZE,
        )

        # B. Subreddit History Sequence Embeddings
        # 1. Identify all unique subreddits in this split to batch encode
        all_subreddits = set()
        for sub_list in df[Config.SUBREDDIT_LIST_COL]:
            # sub_list is already a list (parsed by DataLoader)
            all_subreddits.update(sub_list)

        sub_emb_map = {}
        if all_subreddits:
            unique_subs_list = list(all_subreddits)
            unique_embs = self.model.encode(
                unique_subs_list,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=Config.BATCH_SIZE,
            )
            sub_emb_map = {sub: emb for sub, emb in zip(unique_subs_list, unique_embs)}

        # 2. Construct Padded Tensors
        N = len(df)
        max_len = Config.MAX_SUBREDDIT_SEQ_LEN
        emb_dim = Config.SBERT_EMBEDDING_DIM

        # Initialize with zeros (padding)
        sub_tensor = np.zeros((N, max_len, emb_dim), dtype=np.float32)

        for i, sub_list in enumerate(df[Config.SUBREDDIT_LIST_COL]):
            # Truncate to max_len
            current_subs = sub_list[:max_len]
            for j, sub in enumerate(current_subs):
                if sub in sub_emb_map:
                    sub_tensor[i, j, :] = sub_emb_map[sub]

        return text_embeddings, sub_tensor


class TabularProcessor:
    """
    Handles Feature Engineering for:
    1. Stream A (Random Forest): TF-IDF + Imputed Full-Spectrum Metadata
    2. Stream B (MLP): Arcsinh-Transformed & Scaled Full-Spectrum Metadata
    """

    def process_data(self, df_train, df_val, df_test, safe_cols, load_cached_data=True):
        """
        Generates or loads tabular features.

        Args:
            safe_cols (list): List of safe numerical column names to use as base.

        Returns:
            tuple: (train_tfidf, train_meta_a, train_meta_b, ...)
        """
        set_seed()
        cache_path = os.path.join(Config.WORKING_DIR, "tabular_features.npz")

        # 1. Try Loading from Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached tabular features from {cache_path}")
            try:
                data = np.load(cache_path)
                return (
                    data["train_tfidf"],
                    data["train_meta_a"],
                    data["train_meta_b"],
                    data["val_tfidf"],
                    data["val_meta_a"],
                    data["val_meta_b"],
                    data["test_tfidf"],
                    data["test_meta_a"],
                    data["test_meta_b"],
                )
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print("Generating tabular features...")

        # --- Part 1: TF-IDF (Stream A) ---
        tfidf = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            ngram_range=Config.TFIDF_NGRAM_RANGE,
            stop_words="english",
            dtype=np.float32,
        )

        # Prepare text (fill NaNs)
        train_txt = df_train[Config.TEXT_COL].fillna("").astype(str)
        val_txt = df_val[Config.TEXT_COL].fillna("").astype(str)
        test_txt = df_test[Config.TEXT_COL].fillna("").astype(str)

        # Fit on Train, Transform All
        train_tfidf = tfidf.fit_transform(train_txt).toarray()
        val_tfidf = tfidf.transform(val_txt).toarray()
        test_tfidf = tfidf.transform(test_txt).toarray()

        # --- Part 2: Metadata Engineering (Stream A & B) ---

        def engineer_features(df):
            # Start with safe numeric columns
            meta = df[safe_cols].copy()

            # A. Text Meta-Features
            txt = df[Config.TEXT_COL].fillna("").astype(str)
            meta["text_char_count"] = txt.apply(len)
            meta["text_word_count"] = txt.apply(lambda x: len(x.split()))
            # Caps Ratio: (uppercase chars) / (total chars + 1)
            meta["text_caps_ratio"] = txt.apply(
                lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
            )

            # B. Ratio Engineering
            # Columns: 'requester_upvotes_plus_downvotes_at_request', 'requester_upvotes_minus_downvotes_at_request'
            # Sum = U + D, Diff = U - D
            # U = (Sum + Diff) / 2
            sum_votes = df.get(
                "requester_upvotes_plus_downvotes_at_request",
                pd.Series(0, index=df.index),
            ).fillna(0)
            diff_votes = df.get(
                "requester_upvotes_minus_downvotes_at_request",
                pd.Series(0, index=df.index),
            ).fillna(0)

            upvotes = (sum_votes + diff_votes) / 2
            # Ratio = Upvotes / Total Votes (add epsilon)
            meta["upvote_ratio"] = upvotes / (sum_votes + 1e-5)

            # C. RAOP Activity Ratios
            # Helps distinguish community regulars from drive-by requesters
            raop_comments = df.get(
                "requester_number_of_comments_in_raop_at_request",
                pd.Series(0, index=df.index),
            ).fillna(0)
            total_comments = df.get(
                "requester_number_of_comments_at_request", pd.Series(0, index=df.index)
            ).fillna(0)
            # Add 1.0 to denominator to smooth and avoid div/0
            meta["raop_comment_ratio"] = raop_comments / (total_comments + 1.0)

            raop_posts = df.get(
                "requester_number_of_posts_on_raop_at_request",
                pd.Series(0, index=df.index),
            ).fillna(0)
            total_posts = df.get(
                "requester_number_of_posts_at_request", pd.Series(0, index=df.index)
            ).fillna(0)
            meta["raop_post_ratio"] = raop_posts / (total_posts + 1.0)

            return meta

        # Generate raw engineered dataframes
        meta_train_raw = engineer_features(df_train)
        meta_val_raw = engineer_features(df_val)
        meta_test_raw = engineer_features(df_test)

        # --- Part 3: Stream A Processing (Imputation Only) ---
        # Random Forest handles scale well, but needs no NaNs
        imputer_a = SimpleImputer(strategy="median")
        train_meta_a = imputer_a.fit_transform(meta_train_raw).astype(np.float32)
        val_meta_a = imputer_a.transform(meta_val_raw).astype(np.float32)
        test_meta_a = imputer_a.transform(meta_test_raw).astype(np.float32)

        # --- Part 4: Stream B Processing (Arcsinh + Scaling) ---
        # MLP requires normalized data. Arcsinh handles heavy tails common in social media metrics.

        # 1. Impute (Reuse imputer_a logic or new one)
        # We reuse the imputed data from Stream A as the base to ensure no NaNs before math ops
        train_meta_b_base = train_meta_a.copy()
        val_meta_b_base = val_meta_a.copy()
        test_meta_b_base = test_meta_a.copy()

        # 2. Arcsinh Transformation
        train_meta_b_arc = np.arcsinh(train_meta_b_base)
        val_meta_b_arc = np.arcsinh(val_meta_b_base)
        test_meta_b_arc = np.arcsinh(test_meta_b_base)

        # 3. Standard Scaling
        scaler = StandardScaler()
        train_meta_b = scaler.fit_transform(train_meta_b_arc).astype(np.float32)
        val_meta_b = scaler.transform(val_meta_b_arc).astype(np.float32)
        test_meta_b = scaler.transform(test_meta_b_arc).astype(np.float32)

        # --- Save to Cache ---
        print(f"Saving tabular features to {cache_path}")
        np.savez(
            cache_path,
            train_tfidf=train_tfidf,
            train_meta_a=train_meta_a,
            train_meta_b=train_meta_b,
            val_tfidf=val_tfidf,
            val_meta_a=val_meta_a,
            val_meta_b=val_meta_b,
            test_tfidf=test_tfidf,
            test_meta_a=test_meta_a,
            test_meta_b=test_meta_b,
        )

        return (
            train_tfidf,
            train_meta_a,
            train_meta_b,
            val_tfidf,
            val_meta_a,
            val_meta_b,
            test_tfidf,
            test_meta_a,
            test_meta_b,
        )
