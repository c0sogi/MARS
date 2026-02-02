import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config


class FeaturePipeline:
    """
    Orchestrates the generation of four distinct feature views:
    1. Lexical (Sparse Text + Metadata)
    2. Behavioral (Sparse History + Metadata)
    3. Semantic (Dense Embeddings + Metadata)
    4. Contextual (Metadata Only)
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.lexical_vectorizer = TfidfVectorizer(**Config.LEXICAL_VECTORIZER_PARAMS)
        self.behavioral_vectorizer = TfidfVectorizer(
            **Config.BEHAVIORAL_VECTORIZER_PARAMS
        )
        self.sbert_model = None  # Lazy loading

        # Define base numerical columns to use from the dataframe
        self.numeric_cols = [
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

    def _extract_global_metadata(self, train_df, val_df, test_df):
        """
        Generates the dense Global Metadata Vector.
        Includes user stats, temporal features, and text complexity features.
        """

        def process_single_df(df):
            # 1. Select base numerical columns
            # Ensure columns exist, fill missing with NaN for imputer
            data = df[self.numeric_cols].copy()
            for col in self.numeric_cols:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            # 2. Temporal Features
            if "unix_timestamp_of_request" in df.columns:
                timestamps = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
                data["request_hour"] = timestamps.dt.hour
                data["request_dow"] = timestamps.dt.dayofweek
            else:
                data["request_hour"] = 0
                data["request_dow"] = 0

            # 3. Text Complexity Features
            text_col = Config.TEXT_COL
            texts = df[text_col].astype(str).fillna("")
            data["text_len_char"] = texts.apply(len)
            data["text_len_word"] = texts.apply(lambda x: len(x.split()))

            return data

        # Extract raw features
        train_meta = process_single_df(train_df)
        val_meta = process_single_df(val_df)
        test_meta = process_single_df(test_df)

        # Impute
        train_meta_imputed = self.imputer.fit_transform(train_meta)
        val_meta_imputed = self.imputer.transform(val_meta)
        test_meta_imputed = self.imputer.transform(test_meta)

        # Scale
        train_meta_scaled = self.scaler.fit_transform(train_meta_imputed)
        val_meta_scaled = self.scaler.transform(val_meta_imputed)
        test_meta_scaled = self.scaler.transform(test_meta_imputed)

        return train_meta_scaled, val_meta_scaled, test_meta_scaled

    def _vectorize_text(self, train_df, val_df, test_df):
        """
        Generates TF-IDF matrix for the request text.
        """
        text_col = Config.TEXT_COL
        train_text = train_df[text_col].astype(str).fillna("")
        val_text = val_df[text_col].astype(str).fillna("")
        test_text = test_df[text_col].astype(str).fillna("")

        X_train = self.lexical_vectorizer.fit_transform(train_text)
        X_val = self.lexical_vectorizer.transform(val_text)
        X_test = self.lexical_vectorizer.transform(test_text)

        return X_train, X_val, X_test

    def _vectorize_behavior(self, train_df, val_df, test_df):
        """
        Generates TF-IDF matrix for the subreddit history.
        """
        col = "requester_subreddits_at_request"

        def process_subreddits(df):
            if col not in df.columns:
                return pd.Series([""] * len(df))

            # Handle lists or strings
            return df[col].apply(
                lambda x: (
                    " ".join(x)
                    if isinstance(x, (list, np.ndarray))
                    else str(x) if pd.notnull(x) else ""
                )
            )

        train_subs = process_subreddits(train_df)
        val_subs = process_subreddits(val_df)
        test_subs = process_subreddits(test_df)

        X_train = self.behavioral_vectorizer.fit_transform(train_subs)
        X_val = self.behavioral_vectorizer.transform(val_subs)
        X_test = self.behavioral_vectorizer.transform(test_subs)

        return X_train, X_val, X_test

    def _generate_embeddings(self, train_df, val_df, test_df):
        """
        Generates dense SBERT embeddings.
        """
        if self.sbert_model is None:
            self.sbert_model = SentenceTransformer(Config.SBERT_MODEL_NAME)

        text_col = Config.TEXT_COL

        # Helper to encode
        def encode(df):
            texts = df[text_col].astype(str).fillna("").tolist()
            return self.sbert_model.encode(
                texts,
                batch_size=Config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        X_train = encode(train_df)
        X_val = encode(val_df)
        X_test = encode(test_df)

        return X_train, X_val, X_test

    def create_views(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main entry point. Generates or loads all feature views.

        Returns:
            tuple: (train_features, val_features, test_features)
            Each element is a dictionary with keys: 'lexical', 'behavioral', 'semantic', 'contextual'
        """
        # Define cache file paths
        cache_files = {
            "train": {
                "lexical": "X_train_lexical.npz",
                "behavioral": "X_train_behavioral.npz",
                "semantic": "X_train_semantic.npy",
                "contextual": "X_train_contextual.npy",
            },
            "val": {
                "lexical": "X_val_lexical.npz",
                "behavioral": "X_val_behavioral.npz",
                "semantic": "X_val_semantic.npy",
                "contextual": "X_val_contextual.npy",
            },
            "test": {
                "lexical": "X_test_lexical.npz",
                "behavioral": "X_test_behavioral.npz",
                "semantic": "X_test_semantic.npy",
                "contextual": "X_test_contextual.npy",
            },
        }

        # Check if cache exists
        cache_exists = True
        for split in cache_files:
            for view, filename in cache_files[split].items():
                if not os.path.exists(os.path.join(Config.WORKING_DIR, filename)):
                    cache_exists = False
                    break

        # Load from cache if requested and valid
        if load_cached_data and cache_exists:
            print("Loading features from cache...")
            try:
                features = {"train": {}, "val": {}, "test": {}}
                for split in ["train", "val", "test"]:
                    for view, filename in cache_files[split].items():
                        path = os.path.join(Config.WORKING_DIR, filename)
                        if filename.endswith(".npz"):
                            features[split][view] = sparse.load_npz(path)
                        else:
                            features[split][view] = np.load(path)
                return features["train"], features["val"], features["test"]
            except Exception as e:
                print(f"Cache load failed ({e}). Recomputing...")
        else:
            print("Computing features from scratch...")

        # --- Compute Features ---

        # 1. Global Metadata (Contextual View Base)
        print("Generating Global Metadata...")
        meta_train, meta_val, meta_test = self._extract_global_metadata(
            train_df, val_df, test_df
        )

        # 2. Lexical View (TF-IDF)
        print("Generating Lexical View...")
        lex_train_tfidf, lex_val_tfidf, lex_test_tfidf = self._vectorize_text(
            train_df, val_df, test_df
        )

        # 3. Behavioral View (Subreddit TF-IDF)
        print("Generating Behavioral View...")
        beh_train_tfidf, beh_val_tfidf, beh_test_tfidf = self._vectorize_behavior(
            train_df, val_df, test_df
        )

        # 4. Semantic View (Embeddings)
        print("Generating Semantic View...")
        sem_train_emb, sem_val_emb, sem_test_emb = self._generate_embeddings(
            train_df, val_df, test_df
        )

        # --- Assemble Views ---

        # Helper to stack sparse/dense
        def stack_sparse(m1, m2):
            return sparse.hstack([m1, sparse.csr_matrix(m2)])

        def stack_dense(m1, m2):
            return np.hstack([m1, m2])

        # Train
        X_train_lexical = stack_sparse(lex_train_tfidf, meta_train)
        X_train_behavioral = stack_sparse(beh_train_tfidf, meta_train)
        X_train_semantic = stack_dense(sem_train_emb, meta_train)
        X_train_contextual = meta_train

        # Val
        X_val_lexical = stack_sparse(lex_val_tfidf, meta_val)
        X_val_behavioral = stack_sparse(beh_val_tfidf, meta_val)
        X_val_semantic = stack_dense(sem_val_emb, meta_val)
        X_val_contextual = meta_val

        # Test
        X_test_lexical = stack_sparse(lex_test_tfidf, meta_test)
        X_test_behavioral = stack_sparse(beh_test_tfidf, meta_test)
        X_test_semantic = stack_dense(sem_test_emb, meta_test)
        X_test_contextual = meta_test

        # --- Save to Cache ---
        print("Saving features to cache...")

        def save_set(split_name, lex, beh, sem, ctx):
            base = Config.WORKING_DIR
            sparse.save_npz(os.path.join(base, cache_files[split_name]["lexical"]), lex)
            sparse.save_npz(
                os.path.join(base, cache_files[split_name]["behavioral"]), beh
            )
            np.save(os.path.join(base, cache_files[split_name]["semantic"]), sem)
            np.save(os.path.join(base, cache_files[split_name]["contextual"]), ctx)

        save_set(
            "train",
            X_train_lexical,
            X_train_behavioral,
            X_train_semantic,
            X_train_contextual,
        )
        save_set(
            "val", X_val_lexical, X_val_behavioral, X_val_semantic, X_val_contextual
        )
        save_set(
            "test",
            X_test_lexical,
            X_test_behavioral,
            X_test_semantic,
            X_test_contextual,
        )

        # Return dictionaries
        train_feats = {
            "lexical": X_train_lexical,
            "behavioral": X_train_behavioral,
            "semantic": X_train_semantic,
            "contextual": X_train_contextual,
        }
        val_feats = {
            "lexical": X_val_lexical,
            "behavioral": X_val_behavioral,
            "semantic": X_val_semantic,
            "contextual": X_val_contextual,
        }
        test_feats = {
            "lexical": X_test_lexical,
            "behavioral": X_test_behavioral,
            "semantic": X_test_semantic,
            "contextual": X_test_contextual,
        }

        return train_feats, val_feats, test_feats
