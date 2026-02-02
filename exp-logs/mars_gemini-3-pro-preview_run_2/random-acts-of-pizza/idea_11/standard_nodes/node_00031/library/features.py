import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer
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


def get_feature_pipeline():
    """
    Constructs the Simplified Dense Feature pipeline.

    Cite Lesson 00029: Removes sparse lexical and community features to avoid
    dimensionality explosion and noise on small datasets.
    Cite Lesson 00019: Uses RankGauss (QuantileTransformer) for metadata.

    Returns:
        sklearn.compose.ColumnTransformer: The compiled feature engineering pipeline.
    """
    set_seed(Config.SEED)

    # Column Definitions
    text_col = "text_combined"

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

    # 2. Metadata View (RankGauss)
    # Normalizes distributions via RankGauss.
    # Removed PolynomialFeatures to reduce noise (Cite Lesson 00029).
    metadata_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                QuantileTransformer(
                    output_distribution="normal", random_state=Config.SEED
                ),
            ),
        ]
    )

    # Combine all views
    preprocessor = ColumnTransformer(
        transformers=[
            ("dense", dense_transformer, text_col),
            ("meta", metadata_transformer, numeric_cols),
        ],
        remainder="drop",  # Drop any columns not explicitly transformed
        verbose_feature_names_out=False,
    )

    return preprocessor
