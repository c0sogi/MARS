import os
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import (
    CACHE_DIR,
    TEXT_COL,
    TITLE_COL,
    SUBREDDIT_COL,
    EXCLUDE_COLS,
    RF_ESTIMATORS,
    TEXT_TFIDF_MAX_FEATURES,
    TEXT_TFIDF_NGRAM_RANGE,
    SUBREDDIT_TFIDF_MAX_FEATURES,
    SUBREDDIT_TFIDF_NGRAM_RANGE,
    SBERT_MODEL_NAME,
    FEATURE_WORD_COUNT,
    FEATURE_SENT_COUNT,
    FEATURE_SENTIMENT,
    SEED,
)
from library.utils import set_seed, timer


class FeaturePipeline:
    def __init__(self):
        # Lexical View (Sparse)
        self.lexical_vectorizer = TfidfVectorizer(
            max_features=TEXT_TFIDF_MAX_FEATURES,
            ngram_range=TEXT_TFIDF_NGRAM_RANGE,
            stop_words="english",
            sublinear_tf=True,
        )

        # Behavioral View (Sparse)
        self.behavioral_vectorizer = TfidfVectorizer(
            max_features=SUBREDDIT_TFIDF_MAX_FEATURES,
            ngram_range=SUBREDDIT_TFIDF_NGRAM_RANGE,
            lowercase=False,  # Subreddit names can be case-sensitive or CamelCase
            token_pattern=r"(?u)\b\w+\b",  # Simple word tokenization
        )

        # Semantic View (Dense) components
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

        # SBERT model (lazy loading)
        self.sbert_model = None

        # Feature names tracking
        self.numerical_cols = []

    def _load_sbert(self):
        if self.sbert_model is None:
            # print(f"Loading SBERT model: {SBERT_MODEL_NAME}")
            self.sbert_model = SentenceTransformer(SBERT_MODEL_NAME)

    def _clean_text(self, text_series):
        """Basic text cleaning."""
        return text_series.astype(str).fillna("").apply(lambda x: x.lower().strip())

    def _process_subreddits(self, subreddit_series):
        """Converts list of subreddits to space-separated string."""

        def serialize(val):
            if isinstance(val, (list, np.ndarray)):
                return " ".join([str(s) for s in val])
            return str(val) if pd.notnull(val) else ""

        return subreddit_series.apply(serialize)

    def _get_numerical_features(self, df):
        """Extracts and imputes numerical features."""
        # Identify numerical columns excluding specific ones
        candidates = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in candidates if c not in EXCLUDE_COLS]
        return df[cols], cols

    def _compute_complexity_features(self, df):
        """Computes text complexity and heuristic sentiment."""
        text = df[TEXT_COL].astype(str).fillna("")

        # Word Count
        word_counts = text.apply(lambda x: len(x.split()))

        # Sentence Count (heuristic based on punctuation)
        sent_counts = text.apply(lambda x: len(re.findall(r"[.!?]+", x)) + 1)

        # Heuristic Sentiment
        # Positive/Negative word lists (minimal set to avoid dependencies)
        pos_words = {
            "thanks",
            "thank",
            "appreciate",
            "love",
            "good",
            "great",
            "help",
            "kind",
            "bless",
            "happy",
        }
        neg_words = {
            "bad",
            "hate",
            "awful",
            "terrible",
            "sad",
            "desperate",
            "broke",
            "hungry",
            "starving",
            "fail",
        }

        def get_sentiment(t):
            words = set(re.findall(r"\w+", t.lower()))
            score = sum(1 for w in words if w in pos_words) - sum(
                1 for w in words if w in neg_words
            )
            return score

        sentiment = text.apply(get_sentiment)

        features = pd.DataFrame(
            {
                FEATURE_WORD_COUNT: word_counts,
                FEATURE_SENT_COUNT: sent_counts,
                FEATURE_SENTIMENT: sentiment,
            }
        )
        return features

    def _get_sbert_embeddings(self, text_series):
        """Generates SBERT embeddings."""
        self._load_sbert()
        # Encode in batches to manage memory if needed, but transform is usually efficient
        embeddings = self.sbert_model.encode(
            text_series.tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    def fit(self, df):
        """Fits the vectorizers, imputer, and scaler."""
        # 1. Lexical Fit
        cleaned_text = self._clean_text(df[TEXT_COL])
        self.lexical_vectorizer.fit(cleaned_text)

        # 2. Behavioral Fit
        subreddits_str = self._process_subreddits(df[SUBREDDIT_COL])
        self.behavioral_vectorizer.fit(subreddits_str)

        # 3. Semantic/Dense Fit
        # Numerical features
        num_df, self.numerical_cols = self._get_numerical_features(df)
        self.imputer.fit(num_df)
        imputed_nums = self.imputer.transform(num_df)

        # Complexity features
        comp_df = self._compute_complexity_features(df)

        # SBERT Embeddings (needed for scaler fitting)
        embeddings = self._get_sbert_embeddings(cleaned_text)

        # Concatenate all dense features to fit scaler
        dense_matrix = np.hstack([embeddings, imputed_nums, comp_df.values])
        self.scaler.fit(dense_matrix)

        return self

    def transform(self, df):
        """Transforms data into the three views."""
        # 1. Lexical View
        cleaned_text = self._clean_text(df[TEXT_COL])
        X_lexical = self.lexical_vectorizer.transform(cleaned_text)

        # 2. Behavioral View
        subreddits_str = self._process_subreddits(df[SUBREDDIT_COL])
        X_behavioral = self.behavioral_vectorizer.transform(subreddits_str)

        # 3. Semantic View (Dense)
        # SBERT
        embeddings = self._get_sbert_embeddings(cleaned_text)

        # Numerical
        num_df, _ = self._get_numerical_features(df)
        # Ensure columns match fit time
        num_df = num_df[self.numerical_cols]
        imputed_nums = self.imputer.transform(num_df)

        # Complexity
        comp_df = self._compute_complexity_features(df)

        # Concatenate
        dense_raw = np.hstack([embeddings, imputed_nums, comp_df.values])

        # Scale
        X_semantic = self.scaler.transform(dense_raw)

        return {
            "lexical": X_lexical,
            "behavioral": X_behavioral,
            "semantic": X_semantic,
        }


def generate_data_views(df, stage, pipeline=None, fit=False, load_cached_data=True):
    """
    Generates or loads the three data views for a given stage (train/val/test).

    Args:
        df (pd.DataFrame): The dataframe to process.
        stage (str): 'train', 'val', or 'test'.
        pipeline (FeaturePipeline): Instance of pipeline. Must be provided.
        fit (bool): Whether to fit the pipeline on this data.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing 'lexical', 'behavioral', 'semantic' matrices.
    """
    if pipeline is None:
        raise ValueError("Pipeline instance must be provided.")

    # Define paths
    path_lex = os.path.join(CACHE_DIR, f"X_{stage}_lexical.npz")
    path_beh = os.path.join(CACHE_DIR, f"X_{stage}_behavioral.npz")
    path_sem = os.path.join(CACHE_DIR, f"X_{stage}_semantic.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(path_lex)
            and os.path.exists(path_beh)
            and os.path.exists(path_sem)
        ):
            print(f"Loading cached features for {stage}...")
            X_lex = sp.load_npz(path_lex)
            X_beh = sp.load_npz(path_beh)
            X_sem = np.load(path_sem)

            # If we are supposed to fit, we still need to fit the pipeline state
            # even if we loaded data, OR we assume the caller handles pipeline state.
            # However, usually if data is cached, we assume pipeline state for 'train'
            # might need to be restored. Since we can't pickle the pipeline easily
            # per instructions, we must re-run fit if fit=True, even if cache exists,
            # to ensure pipeline is ready for transform on other sets.
            # Optimization: If fit=True, we MUST run fit.
            if not fit:
                return {"lexical": X_lex, "behavioral": X_beh, "semantic": X_sem}
            else:
                print(
                    f"Cache found but fit=True. Re-fitting pipeline on {stage} data..."
                )

    # Process
    print(f"Processing features for {stage}...")

    if fit:
        with timer(f"Fitting pipeline on {stage}"):
            pipeline.fit(df)

    with timer(f"Transforming {stage} data"):
        views = pipeline.transform(df)

    # Save to cache
    print(f"Saving features for {stage} to {CACHE_DIR}...")
    sp.save_npz(path_lex, views["lexical"])
    sp.save_npz(path_beh, views["behavioral"])
    np.save(path_sem, views["semantic"])

    return views
