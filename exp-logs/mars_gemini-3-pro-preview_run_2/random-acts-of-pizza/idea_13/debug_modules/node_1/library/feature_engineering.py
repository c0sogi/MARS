import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from library.config import Config
from library.utils import setup_logger


class TextEmbedder:
    """
    Handles the generation and caching of L2-normalized text embeddings
    using a pre-trained SentenceTransformer model.
    """

    def __init__(self):
        self.logger = setup_logger("text_embedder")
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.model = None

    def _load_model(self):
        """Lazy loads the SentenceTransformer model to resource usage."""
        if self.model is None:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

    def get_embeddings(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> np.ndarray:
        """
        Generates or loads L2-normalized text embeddings.

        Args:
            df (pd.DataFrame): Dataframe containing the text columns defined in Config.
            split_name (str): Identifier for the split ('train', 'val', 'test') for caching.
            load_cached_data (bool): If True, attempts to load from disk cache first.

        Returns:
            np.ndarray: Matrix of shape (N, 384) containing L2-normalized embeddings.
        """
        # Determine cache path based on split
        if split_name == "train":
            cache_path = Config.TRAIN_EMBEDDINGS_PATH
        elif split_name == "val":
            cache_path = Config.VAL_EMBEDDINGS_PATH
        elif split_name == "test":
            cache_path = Config.TEST_EMBEDDINGS_PATH
        else:
            # Allow custom splits for debugging/demo purposes (Cite debug_lesson_3)
            cache_path = os.path.join(
                Config.WORKING_DIR, f"{split_name}_embeddings.npy"
            )

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            self.logger.info(
                f"Loading {split_name} embeddings from cache: {cache_path}"
            )
            try:
                embeddings = np.load(cache_path)
                return embeddings
            except Exception as e:
                self.logger.warning(
                    f"Failed to load cache for {split_name}: {e}. Recomputing..."
                )

        # 2. Compute embeddings
        self.logger.info(f"Computing embeddings for {split_name}...")
        self._load_model()

        # Concatenate text columns
        # We assume the first column is the start, and append others with space
        if not Config.TEXT_COLS:
            raise ValueError("No text columns defined in Config.")

        texts_series = df[Config.TEXT_COLS[0]].fillna("").astype(str)
        for col in Config.TEXT_COLS[1:]:
            texts_series = texts_series + " " + df[col].fillna("").astype(str)

        texts = texts_series.tolist()

        # Encode (returns numpy array by default with convert_to_numpy=True)
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )

        # 3. Apply L2 Normalization
        # This projects embeddings onto the unit hypersphere, ensuring consistent scale
        embeddings = normalize(embeddings, norm="l2", axis=1)

        # 4. Save to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            self.logger.info(f"Saved {split_name} embeddings to {cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cache for {split_name}: {e}")

        return embeddings


class TabularProcessor:
    """
    Handles the processing of numeric metadata features:
    1. Imputation (Median)
    2. RankGauss Transformation (QuantileTransformer -> Normal)
    """

    def __init__(self):
        self.logger = setup_logger("tabular_processor")
        self.imputer = SimpleImputer(strategy="median")
        # RankGauss: transforms features to follow a standard normal distribution
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.SEED
        )
        self.feature_cols = Config.NUMERIC_FEATURES

    def process_numeric_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """
        Extracts numeric features, fits transformations on training data,
        and transforms all splits.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.
            test_df (pd.DataFrame): Test data.

        Returns:
            tuple: (X_train_tab, X_val_tab, X_test_tab) as numpy arrays.
        """
        self.logger.info("Processing numeric features...")

        # Extract features
        X_train = train_df[self.feature_cols].copy()
        X_val = val_df[self.feature_cols].copy()
        X_test = test_df[self.feature_cols].copy()

        # Validate columns
        if X_train.shape[1] != len(self.feature_cols):
            raise ValueError(
                "DataFrames do not contain all numeric features defined in Config."
            )

        # 1. Impute
        self.logger.info("Imputing missing values (Median)...")
        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)
        X_test = self.imputer.transform(X_test)

        # 2. Transform (RankGauss)
        self.logger.info("Applying QuantileTransformer (RankGauss)...")
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train, X_val, X_test


class FeatureFuser:
    """
    Implements the Differential Scaling logic for feature fusion.
    """

    @staticmethod
    def fuse(
        text_embeddings: np.ndarray, tabular_features: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Fuses text embeddings and tabular features.

        The tabular features are multiplied by 'alpha' before concatenation.
        Since the downstream model is linear with L2 regularization, scaling features UP
        allows them to achieve the same logit contribution with SMALLER weights.
        Smaller weights incur less L2 penalty.
        Therefore, alpha > 1 effectively WEAKENS the regularization on tabular features
        relative to the text embeddings (which are fixed at unit scale).

        Args:
            text_embeddings (np.ndarray): (N, 384) L2-normalized embeddings.
            tabular_features (np.ndarray): (N, D) RankGauss-transformed metadata.
            alpha (float): Modality Balance Factor.

        Returns:
            np.ndarray: (N, 384 + D) fused feature matrix.
        """
        # Apply differential scaling
        scaled_tabular = tabular_features * alpha

        # Concatenate
        fused = np.hstack([text_embeddings, scaled_tabular])

        return fused
