import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os

from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TARGET_COL,
    TEXT_COL,
    SUBREDDIT_COL,
    EXCLUDE_COLS,
    ID_COL,
    TFIDF_PARAMS,
    EMBEDDING_MODEL,
    PCA_COMPONENTS,
    SEED,
)
from library.utils import save_to_cache, load_from_cache, set_seed


class DataProcessor:
    def __init__(self):
        set_seed(SEED)

    def load_raw_data(self):
        """Loads the stratified metadata parquet files."""
        train_df = pd.read_parquet(TRAIN_PATH)
        val_df = pd.read_parquet(VAL_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        return train_df, val_df, test_df

    def _get_numerical_features(self, df):
        """Selects numerical columns, excluding targets and leakage features."""
        # Identify numerical columns
        candidates = df.select_dtypes(include=["number"]).columns.tolist()

        # Filter out target and excluded columns
        final_cols = [
            c for c in candidates if c != TARGET_COL and c not in EXCLUDE_COLS
        ]
        return df[final_cols], final_cols

    def process_metadata(self, train_df, val_df, test_df):
        """
        Generates the Contextual View (Global Metadata Vector).
        Performs Median Imputation and Standard Scaling.
        """
        train_num, _ = self._get_numerical_features(train_df)
        val_num, _ = self._get_numerical_features(val_df)
        test_num, _ = self._get_numerical_features(test_df)

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        train_imputed = imputer.fit_transform(train_num)
        val_imputed = imputer.transform(val_num)
        test_imputed = imputer.transform(test_num)

        # Scale features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imputed)
        val_scaled = scaler.transform(val_imputed)
        test_scaled = scaler.transform(test_imputed)

        return train_scaled, val_scaled, test_scaled

    def process_text_tfidf(self, train_df, val_df, test_df):
        """
        Generates the Lexical View base (Sparse Text Features).
        Uses TF-IDF on the edit-aware request text.
        """
        # Handle NaNs and ensure string type
        train_text = train_df[TEXT_COL].fillna("").astype(str).tolist()
        val_text = val_df[TEXT_COL].fillna("").astype(str).tolist()
        test_text = test_df[TEXT_COL].fillna("").astype(str).tolist()

        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        train_tfidf = vectorizer.fit_transform(train_text)
        val_tfidf = vectorizer.transform(val_text)
        test_tfidf = vectorizer.transform(test_text)

        # Densify for easier concatenation and caching (dataset size permits this)
        return train_tfidf.toarray(), val_tfidf.toarray(), test_tfidf.toarray()

    def process_history_tfidf(self, train_df, val_df, test_df):
        """
        Generates the Behavioral View base (Sparse History Features).
        Treats the list of subreddits as a bag-of-words.
        """

        def parse_subreddits(series):
            # Join list into space-separated string
            return series.apply(
                lambda x: (
                    " ".join(x)
                    if isinstance(x, (list, np.ndarray))
                    else (str(x) if pd.notnull(x) else "")
                )
            )

        train_subs = parse_subreddits(train_df[SUBREDDIT_COL])
        val_subs = parse_subreddits(val_df[SUBREDDIT_COL])
        test_subs = parse_subreddits(test_df[SUBREDDIT_COL])

        # Use same TFIDF params for consistency in sparse representation
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        train_tfidf = vectorizer.fit_transform(train_subs)
        val_tfidf = vectorizer.transform(val_subs)
        test_tfidf = vectorizer.transform(test_subs)

        return train_tfidf.toarray(), val_tfidf.toarray(), test_tfidf.toarray()

    def process_embeddings(self, train_df, val_df, test_df):
        """
        Generates the Semantic View base (Dense Embeddings).
        Uses a pre-trained Sentence Transformer.
        """
        model = SentenceTransformer(EMBEDDING_MODEL)

        train_text = train_df[TEXT_COL].fillna("").astype(str).tolist()
        val_text = val_df[TEXT_COL].fillna("").astype(str).tolist()
        test_text = test_df[TEXT_COL].fillna("").astype(str).tolist()

        # Encode
        train_emb = model.encode(
            train_text, convert_to_numpy=True, show_progress_bar=False
        )
        val_emb = model.encode(val_text, convert_to_numpy=True, show_progress_bar=False)
        test_emb = model.encode(
            test_text, convert_to_numpy=True, show_progress_bar=False
        )

        return train_emb, val_emb, test_emb

    def process_manifold(self, train_emb, val_emb, test_emb):
        """
        Generates the Manifold View base (PCA-reduced Embeddings).
        Specific for the kNN learner to reduce dimensionality.
        """
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        train_pca = pca.fit_transform(train_emb)
        val_pca = pca.transform(val_emb)
        test_pca = pca.transform(test_emb)

        # Scale PCA outputs as required for kNN
        scaler = StandardScaler()
        train_pca = scaler.fit_transform(train_pca)
        val_pca = scaler.transform(val_pca)
        test_pca = scaler.transform(test_pca)

        return train_pca, val_pca, test_pca

    def process_data(self, load_cached_data=True):
        """
        Master pipeline to generate all feature views.
        Checks cache, computes if missing, saves to cache.
        """
        cache_filename = "processed_data_full"

        # 1. Check Cache
        if load_cached_data:
            cached = load_from_cache(cache_filename)
            if cached is not None:
                # Convert NpzFile to dict
                return dict(cached)

        # 2. Load Raw Data
        train_df, val_df, test_df = self.load_raw_data()

        # Extract Targets and IDs
        y_train = train_df[TARGET_COL].values
        y_val = val_df[TARGET_COL].values
        test_ids = test_df[ID_COL].values

        # 3. Feature Engineering

        # A. Contextual (Metadata)
        meta_train, meta_val, meta_test = self.process_metadata(
            train_df, val_df, test_df
        )

        # B. Lexical (Text TFIDF)
        lex_train_raw, lex_val_raw, lex_test_raw = self.process_text_tfidf(
            train_df, val_df, test_df
        )

        # C. Behavioral (History TFIDF)
        beh_train_raw, beh_val_raw, beh_test_raw = self.process_history_tfidf(
            train_df, val_df, test_df
        )

        # D. Semantic (Embeddings)
        sem_train_raw, sem_val_raw, sem_test_raw = self.process_embeddings(
            train_df, val_df, test_df
        )

        # E. Manifold (PCA of Embeddings)
        man_train_raw, man_val_raw, man_test_raw = self.process_manifold(
            sem_train_raw, sem_val_raw, sem_test_raw
        )

        # 4. View Construction (Concatenation)
        # We append Metadata (Contextual) to all other views to preserve cross-modal signals

        # Lexical View
        X_train_lexical = np.hstack([lex_train_raw, meta_train])
        X_val_lexical = np.hstack([lex_val_raw, meta_val])
        X_test_lexical = np.hstack([lex_test_raw, meta_test])

        # Behavioral View
        X_train_behavioral = np.hstack([beh_train_raw, meta_train])
        X_val_behavioral = np.hstack([beh_val_raw, meta_val])
        X_test_behavioral = np.hstack([beh_test_raw, meta_test])

        # Semantic View
        X_train_semantic = np.hstack([sem_train_raw, meta_train])
        X_val_semantic = np.hstack([sem_val_raw, meta_val])
        X_test_semantic = np.hstack([sem_test_raw, meta_test])

        # Manifold View
        X_train_manifold = np.hstack([man_train_raw, meta_train])
        X_val_manifold = np.hstack([man_val_raw, meta_val])
        X_test_manifold = np.hstack([man_test_raw, meta_test])

        # Contextual View (Pure Metadata)
        X_train_contextual = meta_train
        X_val_contextual = meta_val
        X_test_contextual = meta_test

        # 5. Package and Cache
        data_dict = {
            "y_train": y_train,
            "y_val": y_val,
            "test_ids": test_ids,
            "X_train_lexical": X_train_lexical,
            "X_val_lexical": X_val_lexical,
            "X_test_lexical": X_test_lexical,
            "X_train_behavioral": X_train_behavioral,
            "X_val_behavioral": X_val_behavioral,
            "X_test_behavioral": X_test_behavioral,
            "X_train_semantic": X_train_semantic,
            "X_val_semantic": X_val_semantic,
            "X_test_semantic": X_test_semantic,
            "X_train_manifold": X_train_manifold,
            "X_val_manifold": X_val_manifold,
            "X_test_manifold": X_test_manifold,
            "X_train_contextual": X_train_contextual,
            "X_val_contextual": X_val_contextual,
            "X_test_contextual": X_test_contextual,
        }

        save_to_cache(data_dict, cache_filename)

        return data_dict
