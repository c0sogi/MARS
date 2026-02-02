import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import QuantileTransformer
from library.config import Config
from library.utils import setup_logger

# Initialize Logger
logger = setup_logger("feature_topic")


class LDATopicExtractor(BaseEstimator, TransformerMixin):
    """
    Topic Modeling Pipeline for User History (Subreddits).

    This transformer encapsulates the following steps:
    1. CountVectorizer: Converts space-separated subreddit strings into a Bag-of-Words sparse matrix.
    2. LatentDirichletAllocation (LDA): Extracts latent topics from the sparse matrix.
    3. QuantileTransformer (RankGauss): Normalizes the topic distributions to a Gaussian distribution.

    This class is designed to be used inside a Cross-Validation loop to prevent data leakage.
    """

    def __init__(
        self,
        n_components: int = Config.LDA_N_COMPONENTS,
        min_df: int = Config.LDA_MIN_DF,
        random_state: int = Config.LDA_RANDOM_STATE,
        n_jobs: int = -1,
    ):
        """
        Args:
            n_components (int): Number of topics to extract.
            min_df (int): Minimum document frequency for CountVectorizer.
            random_state (int): Seed for reproducibility.
            n_jobs (int): Number of parallel jobs for LDA.
        """
        self.n_components = n_components
        self.min_df = min_df
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize components
        # We use a simple whitespace splitting pattern to preserve subreddit names
        # lowercase=True merges "AskReddit" and "askreddit"
        self.vectorizer = CountVectorizer(
            min_df=self.min_df,
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
        self.lda = LatentDirichletAllocation(
            n_components=self.n_components,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            learning_method="batch",
        )
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=self.random_state
        )

    def fit(self, X, y=None):
        """
        Fits the pipeline components on the input data.

        Args:
            X: Iterable of strings (space-separated subreddits) or DataFrame/Series.
            y: Ignored.

        Returns:
            self
        """
        # Ensure X is a list of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()

        n_samples = len(X)
        logger.info(
            f"Fitting LDATopicExtractor on {n_samples} samples "
            f"(n_components={self.n_components}, min_df={self.min_df})..."
        )

        # 1. Count Vectorization
        X_counts = self.vectorizer.fit_transform(X)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # 2. LDA
        logger.info("Fitting LDA model...")
        X_topics = self.lda.fit_transform(X_counts)

        # 3. RankGauss (QuantileTransformer)
        # Adjust n_quantiles if sample size is small (e.g., in Debug mode)
        # QuantileTransformer requires n_quantiles <= n_samples
        default_quantiles = 1000
        if n_samples < default_quantiles:
            logger.warning(
                f"Sample size ({n_samples}) is smaller than default n_quantiles ({default_quantiles}). "
                f"Adjusting n_quantiles to {n_samples}."
            )
            self.scaler.n_quantiles = n_samples
        else:
            self.scaler.n_quantiles = default_quantiles

        logger.info("Fitting QuantileTransformer (RankGauss)...")
        self.scaler.fit(X_topics)

        return self

    def transform(self, X):
        """
        Transforms the input data using the fitted pipeline.

        Args:
            X: Iterable of strings or DataFrame/Series.

        Returns:
            np.ndarray: Transformed features (n_samples, n_components).
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            X = X.tolist()

        # 1. Count Vectorization
        X_counts = self.vectorizer.transform(X)

        # 2. LDA
        X_topics = self.lda.transform(X_counts)

        # 3. RankGauss
        X_scaled = self.scaler.transform(X_topics)

        return X_scaled

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for the output array.
        """
        return np.array([f"topic_{i}" for i in range(self.n_components)])
