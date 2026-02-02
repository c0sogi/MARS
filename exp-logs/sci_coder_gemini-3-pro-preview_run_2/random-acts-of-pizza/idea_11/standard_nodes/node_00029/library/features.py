import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import set_seed


class SBERTTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that encodes text using a pre-trained Sentence-BERT model.
    Applies L2 normalization to the embeddings to project them onto the hypersphere,
    which is beneficial for linear models and cosine-similarity based spaces.
    """

    def __init__(self, model_name=Config.TRANSFORMER_MODEL_NAME):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        """
        No training required for the pre-trained SBERT model.
        """
        return self

    def transform(self, X):
        """
        Encodes the input text data.

        Args:
            X: pandas Series, DataFrame, or list of strings.

        Returns:
            numpy.ndarray: L2-normalized embeddings.
        """
        # Ensure X is a list of strings
        if isinstance(X, pd.DataFrame):
            sentences = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, pd.Series):
            sentences = X.astype(str).tolist()
        else:
            sentences = list(X)

        # Lazy loading of model to ensure it's on the right device/process
        # and to avoid loading it during pipeline initialization if not needed
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

        # Encode sentences
        # normalize_embeddings=True performs L2 normalization
        embeddings = self.model.encode(
            sentences,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings


class ListJoiner(BaseEstimator, TransformerMixin):
    """
    Transformer that joins a list of strings into a single space-separated string.
    Used for processing subreddit lists for TF-IDF vectorization.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle DataFrame/Series input
        if isinstance(X, pd.DataFrame):
            data = X.iloc[:, 0]
        else:
            data = X

        # Join list elements with space, handling potential non-list types safely
        return [" ".join(x) if isinstance(x, list) else str(x) for x in data]


def get_feature_pipeline():
    """
    Constructs the Dimensionality-Controlled Hybrid Early Fusion pipeline.

    This pipeline processes heterogeneous data sources (Text, Community, Metadata)
    in parallel branches and concatenates them into a single feature vector.
    It incorporates supervised feature selection (Chi2) to control dimensionality.

    Returns:
        sklearn.compose.ColumnTransformer: The compiled feature engineering pipeline.
    """
    set_seed(Config.SEED)

    # Column Definitions
    text_col = "text_combined"
    community_col = "requester_subreddits_at_request"

    numeric_cols = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # 1. Dense Semantic View (SBERT)
    # Projects text into a dense, normalized semantic space (384 dims)
    dense_transformer = SBERTTransformer(model_name=Config.TRANSFORMER_MODEL_NAME)

    # 2. Sparse Lexical View (TF-IDF + Chi2)
    # Captures specific high-signal keywords (unigrams/bigrams)
    # SelectKBest requires 'y' during fit, which is handled by the pipeline
    sparse_transformer = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=5, max_df=0.9, stop_words="english"
                ),
            ),
            ("selector", SelectKBest(score_func=chi2, k=Config.TOP_K_LEXICAL)),
        ]
    )

    # 3. Community View (TF-IDF + Chi2)
    # Captures subreddit participation patterns indicating homophily or need
    community_transformer = Pipeline(
        [
            ("joiner", ListJoiner()),
            ("tfidf", TfidfVectorizer(min_df=2, max_df=0.9, stop_words="english")),
            ("selector", SelectKBest(score_func=chi2, k=Config.TOP_K_COMMUNITY)),
        ]
    )

    # 4. Interaction-Aware Metadata View
    # Normalizes distributions via RankGauss and captures interactions (e.g., Karma * Account Age)
    metadata_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
            ),
            (
                "poly",
                PolynomialFeatures(
                    degree=Config.POLY_DEGREE, interaction_only=True, include_bias=False
                ),
            ),
        ]
    )

    # Combine all views
    preprocessor = ColumnTransformer(
        transformers=[
            ("dense", dense_transformer, text_col),
            ("sparse", sparse_transformer, text_col),
            ("community", community_transformer, community_col),
            ("meta", metadata_transformer, numeric_cols),
        ],
        remainder="drop",  # Drop any columns not explicitly transformed
        verbose_feature_names_out=False,
    )

    return preprocessor
