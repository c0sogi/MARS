import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, normalize
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import setup_logger, set_seed

# Initialize logger
logger = setup_logger("feature_engineering")


class TextEmbedder:
    """
    Handles text embedding generation using a pre-trained SentenceTransformer.
    Applies L2 normalization to the output embeddings.
    """

    def __init__(
        self,
        model_name=Config.EMBEDDING_MODEL_NAME,
        batch_size=Config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.model_name, device=device)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates L2-normalized embeddings for the text columns in the DataFrame.
        Concatenates title and text before embedding.
        """
        self._load_model()

        # Ensure text columns exist and fill NaNs
        title = df["request_title"].fillna("").astype(str)

        # Use edit aware text if available, else fallback (though Config specifies edit_aware)
        text_col = (
            "request_text_edit_aware"
            if "request_text_edit_aware" in df.columns
            else "request_text"
        )
        text = df[text_col].fillna("").astype(str)

        # Concatenate title and text with a separator
        combined_text = title + " " + text
        sentences = combined_text.tolist()

        logger.info(f"Encoding {len(sentences)} texts...")
        # Generate embeddings
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,  # We perform explicit L2 normalization below
        )

        # Apply L2 Normalization
        logger.info("Applying L2 normalization to embeddings...")
        embeddings = normalize(embeddings, norm="l2")

        return embeddings.astype(np.float32)


class TabularProcessor:
    """
    Handles preprocessing of numerical metadata features.
    Performs Median Imputation and RankGauss (QuantileTransformer) scaling.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = QuantileTransformer(
            output_distribution="normal", random_state=Config.RANDOM_SEED
        )
        self.feature_cols = []

    def _get_numeric_cols(self, df: pd.DataFrame) -> list:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude specific columns defined in Config
        final_cols = [c for c in numeric_cols if c not in Config.EXCLUDE_COLS]
        return final_cols

    def fit(self, df: pd.DataFrame):
        """
        Identifies feature columns and fits the imputer and scaler on the training data.
        """
        self.feature_cols = self._get_numeric_cols(df)
        logger.info(f"Identified {len(self.feature_cols)} tabular features.")

        X = df[self.feature_cols].values

        # Fit imputer
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)

        # Fit scaler
        self.scaler.fit(X_imputed)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the DataFrame using the fitted imputer and scaler.
        """
        if not self.feature_cols:
            # Fallback if transform called before fit (should not happen in correct pipeline)
            self.feature_cols = self._get_numeric_cols(df)

        # Select features
        try:
            X = df[self.feature_cols].values
        except KeyError as e:
            missing = list(set(self.feature_cols) - set(df.columns))
            raise KeyError(f"Missing columns in input data: {missing}") from e

        # Apply transformations
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled.astype(np.float32)


def prepare_feature_matrices(df_train, df_val, df_test, load_cached_data=True):
    """
    Orchestrates the feature engineering process.

    1. Checks for cached numpy arrays.
    2. If not found, generates text embeddings and processes tabular data.
    3. Concatenates text and tabular features into combined matrices.
    4. Caches the results.

    Returns:
        X_train, y_train, X_val, y_val, X_test
    """
    Config.ensure_directories()

    # Define cache file paths
    cache_paths = {
        "X_train": os.path.join(Config.CACHE_DIR, "X_train_combined.npy"),
        "y_train": os.path.join(Config.CACHE_DIR, "y_train.npy"),
        "X_val": os.path.join(Config.CACHE_DIR, "X_val_combined.npy"),
        "y_val": os.path.join(Config.CACHE_DIR, "y_val.npy"),
        "X_test": os.path.join(Config.CACHE_DIR, "X_test_combined.npy"),
    }

    # Check if all cache files exist
    all_exist = all(os.path.exists(p) for p in cache_paths.values())

    if load_cached_data and all_exist:
        logger.info("Loading feature matrices from cache...")
        try:
            X_train = np.load(cache_paths["X_train"])
            y_train = np.load(cache_paths["y_train"])
            X_val = np.load(cache_paths["X_val"])
            y_val = np.load(cache_paths["y_val"])
            X_test = np.load(cache_paths["X_test"])
            return X_train, y_train, X_val, y_val, X_test
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Recomputing features.")

    logger.info("Starting feature engineering pipeline...")

    # 1. Text Embeddings
    embedder = TextEmbedder()

    logger.info("Generating text embeddings for Train...")
    X_text_train = embedder.transform(df_train)

    logger.info("Generating text embeddings for Val...")
    X_text_val = embedder.transform(df_val)

    logger.info("Generating text embeddings for Test...")
    X_text_test = embedder.transform(df_test)

    # 2. Tabular Processing
    tabular_processor = TabularProcessor()

    logger.info("Fitting tabular processor on Train...")
    tabular_processor.fit(df_train)

    logger.info("Transforming tabular features...")
    X_tab_train = tabular_processor.transform(df_train)
    X_tab_val = tabular_processor.transform(df_val)
    X_tab_test = tabular_processor.transform(df_test)

    # 3. Combine Features
    # Concatenate text (first) and tabular (second)
    logger.info("Combining text and tabular features...")
    X_train = np.hstack([X_text_train, X_tab_train])
    X_val = np.hstack([X_text_val, X_tab_val])
    X_test = np.hstack([X_text_test, X_tab_test])

    # 4. Extract Targets
    logger.info("Extracting targets...")
    y_train = df_train[Config.TARGET_COL].values.astype(int)
    y_val = df_val[Config.TARGET_COL].values.astype(int)
    # Note: Test set does not have the target column

    # 5. Save to Cache
    logger.info("Saving feature matrices to cache...")
    try:
        np.save(cache_paths["X_train"], X_train)
        np.save(cache_paths["y_train"], y_train)
        np.save(cache_paths["X_val"], X_val)
        np.save(cache_paths["y_val"], y_val)
        np.save(cache_paths["X_test"], X_test)
    except Exception as e:
        logger.error(f"Failed to save to cache: {e}")

    return X_train, y_train, X_val, y_val, X_test
