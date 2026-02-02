import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("feature_engine")


class TextEmbedder:
    """
    Generates semantic embeddings for text data using Sentence-BERT.
    Includes caching mechanisms and L2 normalization.
    """

    def __init__(
        self,
        model_name: str = Config.SBERT_MODEL_NAME,
        batch_size: int = Config.SBERT_BATCH_SIZE,
        device: str = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading SBERT model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def generate_embeddings(
        self, df: pd.DataFrame, save_path: str = None, load_cached: bool = True
    ) -> np.ndarray:
        """
        Generates or loads embeddings for the given DataFrame.

        Args:
            df (pd.DataFrame): Data containing 'request_title' and 'request_text_edit_aware'.
            save_path (str): Path to save/load the .npy file.
            load_cached (bool): Whether to attempt loading from cache.

        Returns:
            np.ndarray: L2-normalized embeddings of shape (n_samples, embedding_dim).
        """
        # 1. Check Cache
        if load_cached and save_path and os.path.exists(save_path):
            logger.info(f"Loading embeddings from cache: {save_path}")
            try:
                embeddings = np.load(save_path)
                if embeddings.shape[0] == len(df):
                    return embeddings
                else:
                    logger.warning(
                        f"Cached embeddings shape {embeddings.shape} does not match DataFrame length {len(df)}. Recomputing."
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}. Recomputing.")

        # 2. Prepare Text
        logger.info("Preparing text for embedding generation...")
        # Concatenate title and text
        # Ensure columns exist and are strings
        titles = df["request_title"].fillna("").astype(str)
        texts = df["request_text_edit_aware"].fillna("").astype(str)

        # Format: "Title. Text"
        input_texts = (titles + ". " + texts).tolist()

        # 3. Generate Embeddings
        self._load_model()
        logger.info(
            f"Encoding {len(input_texts)} texts with batch size {self.batch_size}..."
        )

        # encode() with normalize_embeddings=True performs L2 normalization
        embeddings = self.model.encode(
            input_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # 4. Save Cache
        if save_path:
            logger.info(f"Saving embeddings to cache: {save_path}")
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, embeddings)
            except Exception as e:
                logger.error(f"Failed to save embeddings to cache: {e}")

        return embeddings


class HomophilyTargetEncoder:
    """
    Encodes subreddit lists into dense features based on historical success rates.
    Implements Out-of-Fold (OOF) generation to prevent target leakage during training.
    """

    def __init__(self, smoothing: float = 10.0):
        """
        Args:
            smoothing (float): Smoothing parameter (alpha) for Bayesian smoothing.
                               Higher values pull rare subreddits closer to the global mean.
        """
        self.smoothing = smoothing
        self.subreddit_map = {}
        self.global_mean = 0.0
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = Config.TARGET_COL):
        """
        Learns the success rates of subreddits from the provided DataFrame.
        Used for the final model training (fitting on full train to transform test).
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame.")

        # Calculate global mean
        self.global_mean = df[target_col].mean()

        # Explode the subreddit list column to have one row per subreddit-user pair
        # We assume 'requester_subreddits_at_request' is a list of strings
        df_exploded = df[[Config.SUBREDDIT_COL, target_col]].explode(
            Config.SUBREDDIT_COL
        )

        # Remove NaNs (empty lists result in NaNs after explode)
        df_exploded = df_exploded.dropna(subset=[Config.SUBREDDIT_COL])

        if df_exploded.empty:
            logger.warning("No subreddits found to fit. Using global mean only.")
            self.is_fitted = True
            return self

        # Aggregation: Count and Sum of target
        stats = df_exploded.groupby(Config.SUBREDDIT_COL)[target_col].agg(
            ["count", "sum"]
        )

        # Bayesian Smoothing
        # smoothed_mean = (sum + alpha * global_mean) / (count + alpha)
        stats["smoothed_mean"] = (stats["sum"] + self.smoothing * self.global_mean) / (
            stats["count"] + self.smoothing
        )

        # Store as dictionary
        self.subreddit_map = stats["smoothed_mean"].to_dict()
        self.is_fitted = True

        logger.info(
            f"Fitted HomophilyTargetEncoder on {len(self.subreddit_map)} subreddits."
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the subreddit lists in df into [mean_success, max_success] features.
        """
        if not self.is_fitted:
            raise RuntimeError("HomophilyTargetEncoder is not fitted.")

        return self._transform_internal(df, self.subreddit_map, self.global_mean)

    def fit_transform_oof(
        self, df: pd.DataFrame, target_col: str = Config.TARGET_COL, n_folds: int = 5
    ) -> np.ndarray:
        """
        Generates Out-of-Fold features for the training set.
        Splits data into K folds; for each fold, fits on the other K-1 and transforms the current fold.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in DataFrame.")

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED
        )

        # Initialize output array: [n_samples, 2] -> (mean_success, max_success)
        oof_features = np.zeros((len(df), 2), dtype=np.float32)

        y = df[target_col].values

        logger.info(f"Starting OOF Target Encoding with {n_folds} folds...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            # Create a temporary encoder for this fold
            fold_encoder = HomophilyTargetEncoder(smoothing=self.smoothing)

            # Get train and val subsets
            df_train_fold = df.iloc[train_idx]
            df_val_fold = df.iloc[val_idx]

            # Fit on training fold
            fold_encoder.fit(df_train_fold, target_col=target_col)

            # Transform validation fold
            fold_feats = fold_encoder.transform(df_val_fold)

            # Store in OOF array
            oof_features[val_idx] = fold_feats

        # After OOF generation, we fit the encoder on the FULL dataset
        # so it's ready for subsequent inference on test data if needed.
        self.fit(df, target_col=target_col)

        return oof_features

    def _transform_internal(
        self, df: pd.DataFrame, mapping: dict, default_val: float
    ) -> np.ndarray:
        """
        Internal helper to map subreddits to scores and aggregate.
        """
        # We process row by row. While slower than vectorized pandas for simple ops,
        # list aggregation is often cleaner this way or using apply.
        # Given dataset size (~3k-4k), apply is sufficient.

        def get_stats(subreddits):
            if not isinstance(subreddits, list) or len(subreddits) == 0:
                # No history -> return priors (global mean)
                return pd.Series([default_val, default_val])

            # Map subreddits to scores, using default_val for unknown subreddits
            scores = [mapping.get(sub, default_val) for sub in subreddits]

            if not scores:
                return pd.Series([default_val, default_val])

            mean_score = np.mean(scores)
            max_score = np.max(scores)

            return pd.Series([mean_score, max_score])

        # Apply to the subreddit column
        # Result is a DataFrame with 2 columns
        features = df[Config.SUBREDDIT_COL].apply(get_stats)

        return features.values.astype(np.float32)
