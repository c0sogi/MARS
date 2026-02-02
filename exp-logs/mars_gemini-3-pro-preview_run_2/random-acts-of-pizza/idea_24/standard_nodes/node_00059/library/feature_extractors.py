import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class TextEmbedder(BaseEstimator, TransformerMixin):
    """
    Generates SBERT embeddings for text features.
    Includes caching mechanisms to persist embeddings to disk.
    """

    def __init__(self, model_name=Config.SBERT_MODEL, batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.logger = setup_logger("TextEmbedder")
        self.model = None

    def fit(self, X, y=None):
        """
        Stateless transformer, fit does nothing.
        """
        return self

    def _load_model(self):
        if self.model is None:
            self.logger.info(f"Loading SBERT model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def transform(
        self, df: pd.DataFrame, cache_name: str = None, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates or loads embeddings for the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing text columns.
            cache_name (str): Unique identifier for the cache file (e.g., 'train', 'test').
                              If None, caching is disabled.
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: L2-normalized embeddings.
        """
        # Construct cache path if cache_name is provided
        cache_path = None
        if cache_name:
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            cache_path = os.path.join(
                Config.WORKING_DIR, f"{cache_name}_embeddings.npy"
            )

        # 1. Try loading from cache
        if load_cached_data and cache_path and os.path.exists(cache_path):
            self.logger.info(f"Loading embeddings from cache: {cache_path}")
            try:
                embeddings = np.load(cache_path)
                # Verify shape matches
                if len(embeddings) == len(df):
                    return embeddings
                else:
                    self.logger.warning(
                        f"Cached embeddings shape {embeddings.shape} mismatch with df {len(df)}. Recomputing."
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Recomputing.")

        # 2. Compute Embeddings
        self._load_model()

        # Concatenate text columns
        self.logger.info("Preprocessing text columns...")
        text_data = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        for col in Config.TEXT_COLS[1:]:
            text_data = text_data + " " + df[col].fillna("").astype(str)

        sentences = text_data.tolist()

        self.logger.info(f"Encoding {len(sentences)} sentences...")
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # 3. L2 Normalization
        self.logger.info("Applying L2 normalization...")
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norm[norm == 0] = 1e-10
        embeddings = embeddings / norm

        # 4. Save to cache
        if cache_path:
            self.logger.info(f"Saving embeddings to cache: {cache_path}")
            try:
                np.save(cache_path, embeddings)
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

        return embeddings


class BayesianSubredditEncoder(BaseEstimator, TransformerMixin):
    """
    Implements Bayesian Target Encoding for lists of subreddits.
    Computes a smoothed success rate for each subreddit and aggregates per user.
    """

    def __init__(
        self,
        col_name=Config.HISTORY_COL,
        smoothing=Config.HISTORY_SMOOTHING,
        min_samples=Config.HISTORY_MIN_SAMPLES,
    ):
        self.col_name = col_name
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.subreddit_map = {}
        self.global_mean = 0.0
        self.logger = setup_logger("BayesianSubredditEncoder")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Computes the smoothed mean target for each subreddit.

        Args:
            X (pd.DataFrame): Dataframe containing the subreddit list column.
            y (pd.Series): Target values.
        """
        self.logger.info("Fitting Bayesian Subreddit Encoder...")

        # Ensure input format
        if self.col_name not in X.columns:
            raise ValueError(f"Column {self.col_name} not found in input DataFrame.")

        # Calculate Global Mean
        self.global_mean = y.mean()

        # Create a temporary dataframe for explosion
        temp_df = pd.DataFrame({"subreddits": X[self.col_name], "target": y.values})

        # Explode the list column to have one row per subreddit-user pair
        exploded = temp_df.explode("subreddits")

        # Filter out NaNs if any (empty lists result in NaNs after explode)
        exploded = exploded.dropna(subset=["subreddits"])

        # Aggregation: Count and Sum of target per subreddit
        stats = exploded.groupby("subreddits")["target"].agg(["count", "sum"])

        # Apply Bayesian Smoothing
        # Formula: (C * global_mean + sum) / (C + count)
        # where C is smoothing factor
        stats["smoothed_score"] = (self.smoothing * self.global_mean + stats["sum"]) / (
            self.smoothing + stats["count"]
        )

        # Store the map
        self.subreddit_map = stats["smoothed_score"].to_dict()

        self.logger.info(
            f"Encoded {len(self.subreddit_map)} unique subreddits. Global mean: {self.global_mean:.4f}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Maps subreddits to scores and aggregates per user.

        Args:
            X (pd.DataFrame): Dataframe to transform.

        Returns:
            np.ndarray: Column vector of history scores.
        """
        self.logger.info("Transforming subreddit history...")

        if self.col_name not in X.columns:
            raise ValueError(f"Column {self.col_name} not found in input DataFrame.")

        def get_user_score(sub_list):
            if not isinstance(sub_list, list) or len(sub_list) == 0:
                return self.global_mean

            # Map each subreddit to its score, fallback to global_mean if unknown
            scores = [self.subreddit_map.get(sub, self.global_mean) for sub in sub_list]

            # Aggregate: Mean of scores
            return np.mean(scores)

        # Apply row-wise
        scores = X[self.col_name].apply(get_user_score).values

        return scores.reshape(-1, 1)


class RankGaussScaler(BaseEstimator, TransformerMixin):
    """
    Applies QuantileTransformer with output_distribution='normal' (RankGauss).
    """

    def __init__(self, numeric_cols=Config.NUMERIC_COLS):
        self.numeric_cols = numeric_cols
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )
        self.logger = setup_logger("RankGaussScaler")

    def fit(self, X: pd.DataFrame, y=None):
        self.logger.info("Fitting RankGauss Scaler...")
        # Select only numeric columns
        X_num = X[self.numeric_cols].copy()
        # Handle NaNs if any (simple fill, though QuantileTransformer handles NaNs in recent versions, safe to fill)
        X_num = X_num.fillna(X_num.median())
        self.scaler.fit(X_num)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self.logger.info("Transforming numeric metadata...")
        X_num = X[self.numeric_cols].copy()
        X_num = X_num.fillna(X_num.median())
        return self.scaler.transform(X_num)
