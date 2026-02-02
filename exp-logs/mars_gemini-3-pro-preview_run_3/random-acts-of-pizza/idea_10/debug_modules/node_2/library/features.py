import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
import torch
import joblib

from library.config import (
    TEXT_TFIDF_PARAMS,
    SUBREDDIT_TFIDF_PARAMS,
    SUBREDDIT_SVD_COMPONENTS,
    SBERT_MODEL_NAME,
    DROP_COLS,
    RETRIEVAL_SUFFIX,
    TEXT_COL,
    SUBREDDIT_COL,
    TARGET_COL,
    WORKING_DIR,
    SEED,
)
from library.utils import Timer, ensure_dir


class FeaturePipeline:
    def __init__(self, cache_dir=WORKING_DIR):
        """
        Initializes the feature engineering pipeline with necessary transformers.
        """
        self.cache_dir = cache_dir

        # Lexical View Transformers
        self.text_tfidf = TfidfVectorizer(**TEXT_TFIDF_PARAMS)

        # Behavioral View Transformers
        self.subreddit_tfidf = TfidfVectorizer(**SUBREDDIT_TFIDF_PARAMS)
        self.subreddit_svd = TruncatedSVD(
            n_components=SUBREDDIT_SVD_COMPONENTS, random_state=SEED
        )

        # Dense/Metadata Transformers
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # SBERT Model placeholder (lazy initialization to save resources if loading from cache)
        self.sbert_model = None

        # Internal state
        self.metadata_cols = []
        self.is_fitted = False

    def _init_sbert(self):
        """
        Initializes the SBERT model on the appropriate device.
        """
        if self.sbert_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

    def _get_metadata_cols(self, df):
        """
        Identifies valid numerical metadata columns, excluding target, IDs, and retrieval-time features.
        """
        candidates = df.select_dtypes(include=["number"]).columns.tolist()
        valid_cols = []
        for col in candidates:
            if col in DROP_COLS:
                continue
            if col == TARGET_COL:
                continue
            if col.endswith(RETRIEVAL_SUFFIX):
                continue
            valid_cols.append(col)
        return valid_cols

    def _extract_text_complexity(self, df):
        """
        Extracts structural text complexity features: char count, word count, sentence count, avg word length.
        """
        texts = df[TEXT_COL].fillna("").astype(str)

        # Vectorized string operations
        char_len = texts.apply(len).values.reshape(-1, 1)
        word_count = texts.apply(lambda x: len(x.split())).values.reshape(-1, 1)
        # Simple heuristic for sentence count
        sent_count = texts.apply(
            lambda x: x.count(".") + x.count("!") + x.count("?")
        ).values.reshape(-1, 1)

        # Avoid division by zero
        avg_word_len = np.divide(
            char_len,
            word_count,
            out=np.zeros_like(char_len, dtype=float),
            where=word_count != 0,
        )

        return np.hstack([char_len, word_count, sent_count, avg_word_len])

    def fit(self, df):
        """
        Fits all transformers on the training data.
        """
        with Timer("FeaturePipeline Fit"):
            # 1. Metadata fitting
            self.metadata_cols = self._get_metadata_cols(df)
            meta_data = df[self.metadata_cols].values
            self.imputer.fit(meta_data)
            # Impute to fit scaler
            meta_data_imputed = self.imputer.transform(meta_data)
            self.scaler.fit(meta_data_imputed)

            # 2. Text TF-IDF fitting
            texts = df[TEXT_COL].fillna("").astype(str)
            self.text_tfidf.fit(texts)

            # 3. Subreddit TF-IDF & SVD fitting
            # Convert list of subreddits to space-separated string for TF-IDF
            subreddits = (
                df[SUBREDDIT_COL]
                .apply(lambda x: " ".join(x) if isinstance(x, list) else "")
                .values
            )
            subreddit_matrix = self.subreddit_tfidf.fit_transform(subreddits)
            self.subreddit_svd.fit(subreddit_matrix)

            self.is_fitted = True
        return self

    def transform(self, df, name, load_cached_data=False):
        """
        Transforms the data into three views: Lexical, Behavioral, and Dense.
        Implements strict caching logic.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")

        # Define cache paths
        cache_files = {
            "lexical": os.path.join(self.cache_dir, f"X_{name}_lexical.npz"),
            "behavioral": os.path.join(self.cache_dir, f"X_{name}_behavioral.npz"),
            "dense": os.path.join(self.cache_dir, f"X_{name}_dense.npy"),
        }

        # 1. Check Cache
        if load_cached_data:
            if all(os.path.exists(p) for p in cache_files.values()):
                # print(f"Loading features for {name} from cache...")
                X_lexical = sp.load_npz(cache_files["lexical"])
                X_behavioral = sp.load_npz(cache_files["behavioral"])
                X_dense = np.load(cache_files["dense"])
                return {
                    "lexical": X_lexical,
                    "behavioral": X_behavioral,
                    "dense": X_dense,
                }

        # 2. Compute Features if not cached
        with Timer(f"Feature Extraction: {name}"):
            # Metadata Processing
            meta_data = df[self.metadata_cols].values
            meta_data_imputed = self.imputer.transform(meta_data)
            meta_data_scaled = self.scaler.transform(
                meta_data_imputed
            )  # Scaled for Dense View

            # Text TF-IDF
            texts = df[TEXT_COL].fillna("").astype(str)
            X_text_tfidf = self.text_tfidf.transform(texts)

            # Subreddit TF-IDF & SVD
            subreddits = (
                df[SUBREDDIT_COL]
                .apply(lambda x: " ".join(x) if isinstance(x, list) else "")
                .values
            )
            X_sub_tfidf = self.subreddit_tfidf.transform(subreddits)
            X_sub_svd = self.subreddit_svd.transform(X_sub_tfidf)

            # SBERT Embeddings
            self._init_sbert()
            # Encode in batches, silent execution
            X_sbert = self.sbert_model.encode(
                texts.tolist(),
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Text Complexity
            X_complexity = self._extract_text_complexity(df)

            # --- Construct Views ---

            # Lexical View (Sparse): Text TF-IDF + Metadata (Imputed)
            # We use imputed but unscaled metadata for tree-based sparse models (RF)
            X_lexical = sp.hstack(
                [X_text_tfidf, sp.csr_matrix(meta_data_imputed)]
            ).tocsr()

            # Behavioral View (Sparse): Subreddit TF-IDF + Metadata (Imputed)
            X_behavioral = sp.hstack(
                [X_sub_tfidf, sp.csr_matrix(meta_data_imputed)]
            ).tocsr()

            # Dense View (Dense): SBERT + Subreddit SVD + Complexity + Metadata (Scaled)
            # Used for XGBoost and Meta-Learner
            X_dense = np.hstack([X_sbert, X_sub_svd, X_complexity, meta_data_scaled])

            # 3. Save to Cache
            ensure_dir(cache_files["lexical"])
            sp.save_npz(cache_files["lexical"], X_lexical)
            sp.save_npz(cache_files["behavioral"], X_behavioral)
            np.save(cache_files["dense"], X_dense)

        return {"lexical": X_lexical, "behavioral": X_behavioral, "dense": X_dense}

    def save(self, path):
        """
        Saves the fitted pipeline object.
        Removes the SBERT model from the object before pickling to save space/time,
        as it can be re-initialized.
        """
        ensure_dir(path)
        sbert_ref = self.sbert_model
        self.sbert_model = None
        joblib.dump(self, path)
        self.sbert_model = sbert_ref

    @classmethod
    def load(cls, path):
        """
        Loads the pipeline object.
        """
        return joblib.load(path)
