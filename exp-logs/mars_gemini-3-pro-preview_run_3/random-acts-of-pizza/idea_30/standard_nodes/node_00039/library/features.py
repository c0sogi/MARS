import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import log, set_seed


class FeatureEngineer:
    """
    Handles feature engineering for the Pent-View Stacking architecture.
    Generates Lexical, Behavioral, Semantic, and Contextual feature views.
    """

    def __init__(self):
        # Lexical Branch Vectorizer (Text)
        self.tfidf_text = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            min_df=Config.TFIDF_MIN_DF,
            sublinear_tf=Config.TFIDF_SUBLINEAR,
            stop_words="english",
            ngram_range=(1, 2),
        )

        # Behavioral Branch Vectorizer (History)
        self.tfidf_history = TfidfVectorizer(
            max_features=Config.TFIDF_MAX_FEATURES,
            min_df=Config.TFIDF_MIN_DF,
            sublinear_tf=Config.TFIDF_SUBLINEAR,
            stop_words="english",
            ngram_range=(1, 1),  # Concepts/Subreddits are unigrams
        )

        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.embedding_model = None  # Lazy load

    def _load_embedding_model(self):
        """Lazy loads the Sentence Transformer model."""
        if self.embedding_model is None:
            log(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)

    def _generate_metadata(self, df, is_train=False):
        """
        Extracts numerical and temporal features.
        Returns unscaled numpy array.
        """
        # 1. Select Numerical Columns (Positive Selection)
        meta_df = df[Config.NUMERICAL_COLS].copy()

        # 2. Temporal Features
        if "unix_timestamp_of_request" in df.columns:
            dt = pd.to_datetime(df["unix_timestamp_of_request"], unit="s")
            meta_df["hour"] = dt.dt.hour
            meta_df["dayofweek"] = dt.dt.dayofweek
        else:
            meta_df["hour"] = 0
            meta_df["dayofweek"] = 0

        # Convert to float32
        X_meta = meta_df.values.astype(np.float32)

        # 3. Impute
        if is_train:
            X_meta = self.imputer.fit_transform(X_meta)
        else:
            X_meta = self.imputer.transform(X_meta)

        return X_meta

    def _process_text_lists(self, df):
        """Helper to extract text and join subreddit lists."""
        # Request Text
        texts = df["request_text"].fillna("").astype(str).tolist()

        # History (Subreddits)
        if "requester_subreddits_at_request" in df.columns:
            subreddits = (
                df["requester_subreddits_at_request"]
                .apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
                .fillna("")
                .tolist()
            )
        else:
            subreddits = [""] * len(df)

        return texts, subreddits

    def _compute_embeddings_and_interaction(self, texts, subreddits):
        """
        Generates dense embeddings and computes cross-modal interaction (Cosine Similarity).
        """
        self._load_embedding_model()

        # Encode with normalization for cosine similarity
        log("Encoding texts...")
        text_emb = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        log("Encoding subreddit history...")
        hist_emb = self.embedding_model.encode(
            subreddits,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Compute Cosine Similarity (Interaction)
        # Element-wise dot product since vectors are normalized
        interaction = np.sum(text_emb * hist_emb, axis=1, keepdims=True)

        return text_emb, hist_emb, interaction

    def generate_features(self, train_df, val_df, test_df, load_cached_data=True):
        """
        Main method to generate all feature views for Train, Val, and Test.
        Handles caching to disk.
        """
        cache_dir = Config.WORKING_DIR
        os.makedirs(cache_dir, exist_ok=True)

        # Define cache filenames
        files = {
            "train_lexical": "X_train_lexical.npz",
            "train_behavioral": "X_train_behavioral.npz",
            "train_semantic": "X_train_semantic.npy",
            "train_metadata": "X_train_metadata.npy",
            "val_lexical": "X_val_lexical.npz",
            "val_behavioral": "X_val_behavioral.npz",
            "val_semantic": "X_val_semantic.npy",
            "val_metadata": "X_val_metadata.npy",
            "test_lexical": "X_test_lexical.npz",
            "test_behavioral": "X_test_behavioral.npz",
            "test_semantic": "X_test_semantic.npy",
            "test_metadata": "X_test_metadata.npy",
        }

        # Check if all cache files exist
        all_exist = all(
            os.path.exists(os.path.join(cache_dir, f)) for f in files.values()
        )

        if load_cached_data and all_exist:
            log("Loading feature views from cache...")
            data = {}
            for key, fname in files.items():
                path = os.path.join(cache_dir, fname)
                if fname.endswith(".npz"):
                    data[key] = sp.load_npz(path)
                else:
                    data[key] = np.load(path)
        else:
            log("Generating features from scratch...")

            # 1. Prepare Raw Data
            train_texts, train_subs = self._process_text_lists(train_df)
            val_texts, val_subs = self._process_text_lists(val_df)
            test_texts, test_subs = self._process_text_lists(test_df)

            # 2. TF-IDF (Sparse)
            log("Generating TF-IDF features...")
            train_tfidf_text = self.tfidf_text.fit_transform(train_texts)
            val_tfidf_text = self.tfidf_text.transform(val_texts)
            test_tfidf_text = self.tfidf_text.transform(test_texts)

            train_tfidf_hist = self.tfidf_history.fit_transform(train_subs)
            val_tfidf_hist = self.tfidf_history.transform(val_subs)
            test_tfidf_hist = self.tfidf_history.transform(test_subs)

            # 3. Embeddings & Interaction (Dense)
            log("Generating Embeddings and Interaction features...")
            train_emb_text, _, train_interaction = (
                self._compute_embeddings_and_interaction(train_texts, train_subs)
            )
            val_emb_text, _, val_interaction = self._compute_embeddings_and_interaction(
                val_texts, val_subs
            )
            test_emb_text, _, test_interaction = (
                self._compute_embeddings_and_interaction(test_texts, test_subs)
            )

            # 4. Base Metadata (Dense)
            log("Generating Metadata features...")
            train_meta_base = self._generate_metadata(train_df, is_train=True)
            val_meta_base = self._generate_metadata(val_df, is_train=False)
            test_meta_base = self._generate_metadata(test_df, is_train=False)

            # 5. Construct Global Metadata (Base + Interaction)
            # Concatenate Base Metadata and Interaction Feature
            train_global_meta = np.hstack([train_meta_base, train_interaction])
            val_global_meta = np.hstack([val_meta_base, val_interaction])
            test_global_meta = np.hstack([test_meta_base, test_interaction])

            # Scale Global Metadata
            train_global_meta = self.scaler.fit_transform(train_global_meta)
            val_global_meta = self.scaler.transform(val_global_meta)
            test_global_meta = self.scaler.transform(test_global_meta)

            # 6. Construct Final Views
            log("Constructing final feature views...")

            def create_views(tfidf_text, tfidf_hist, emb_text, global_meta):
                # Lexical: Text TF-IDF + Global Meta (Sparse)
                lexical = sp.hstack([tfidf_text, sp.csr_matrix(global_meta)])

                # Behavioral: History TF-IDF + Global Meta (Sparse)
                behavioral = sp.hstack([tfidf_hist, sp.csr_matrix(global_meta)])

                # Semantic: Text Embedding + Global Meta (Dense)
                semantic = np.hstack([emb_text, global_meta])

                # Contextual: Global Meta (Dense)
                contextual = global_meta

                return lexical, behavioral, semantic, contextual

            X_train_lex, X_train_beh, X_train_sem, X_train_ctx = create_views(
                train_tfidf_text, train_tfidf_hist, train_emb_text, train_global_meta
            )

            X_val_lex, X_val_beh, X_val_sem, X_val_ctx = create_views(
                val_tfidf_text, val_tfidf_hist, val_emb_text, val_global_meta
            )

            X_test_lex, X_test_beh, X_test_sem, X_test_ctx = create_views(
                test_tfidf_text, test_tfidf_hist, test_emb_text, test_global_meta
            )

            # Store in dictionary
            data = {
                "train_lexical": X_train_lex,
                "train_behavioral": X_train_beh,
                "train_semantic": X_train_sem,
                "train_metadata": X_train_ctx,
                "val_lexical": X_val_lex,
                "val_behavioral": X_val_beh,
                "val_semantic": X_val_sem,
                "val_metadata": X_val_ctx,
                "test_lexical": X_test_lex,
                "test_behavioral": X_test_beh,
                "test_semantic": X_test_sem,
                "test_metadata": X_test_ctx,
            }

            # Save to Cache
            log("Saving features to cache...")
            for key, val in data.items():
                path = os.path.join(cache_dir, files[key])
                if sp.issparse(val):
                    sp.save_npz(path, val)
                else:
                    np.save(path, val)

        # Organize into return structure
        return {
            "train": {
                "lexical": data["train_lexical"],
                "behavioral": data["train_behavioral"],
                "semantic": data["train_semantic"],
                "contextual": data["train_metadata"],
            },
            "val": {
                "lexical": data["val_lexical"],
                "behavioral": data["val_behavioral"],
                "semantic": data["val_semantic"],
                "contextual": data["val_metadata"],
            },
            "test": {
                "lexical": data["test_lexical"],
                "behavioral": data["test_behavioral"],
                "semantic": data["test_semantic"],
                "contextual": data["test_metadata"],
            },
        }
