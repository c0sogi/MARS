import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sentence_transformers import InputExample
from library import config, data_handler
from library.utils import setup_logger

logger = setup_logger("preprocessor")


class TextPreprocessor:
    """
    Helper class to extract and format text data for the model.
    """

    def __init__(self):
        pass

    def get_texts(self, df: pd.DataFrame) -> list:
        """
        Returns the combined text column as a list of strings.
        """
        # data_handler creates 'combined_text'
        if "combined_text" in df.columns:
            return df["combined_text"].tolist()

        # Fallback logic if needed, though data_handler should handle this
        logger.warning(
            "'combined_text' column missing. Reconstructing from config columns."
        )
        return (
            df[config.TEXT_COLS]
            .fillna("")
            .apply(lambda x: " ".join(x.astype(str)), axis=1)
            .tolist()
        )


class TabularScaler:
    """
    Manages the scaling of numerical metadata using RobustScaler.
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.columns = config.NUMERICAL_COLS

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on the provided DataFrame.
        """
        self.scaler.fit(df[self.columns])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms the DataFrame using the fitted scaler.
        """
        return self.scaler.transform(df[self.columns])


class SiameseDatasetBuilder:
    """
    Constructs datasets for Siamese Network training (SentenceTransformers).
    """

    def __init__(self):
        pass

    def create_examples(self, df: pd.DataFrame) -> list:
        """
        Converts a DataFrame into a list of InputExample objects.
        Only processes rows where the target label is present.
        """
        examples = []
        # Ensure text is ready
        text_prep = TextPreprocessor()
        texts = text_prep.get_texts(df)

        if config.TARGET_COL in df.columns:
            labels = df[config.TARGET_COL].tolist()
            for text, label in zip(texts, labels):
                # InputExample for BatchHardTripletLoss needs text and integer label
                examples.append(InputExample(texts=[str(text)], label=int(label)))

        return examples


def get_scaled_features(load_cached_data: bool = True):
    """
    Orchestrates the loading of data, fitting of the scaler, and transformation of features.
    Caches the resulting numpy arrays to disk.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled) as numpy arrays.
    """
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_path = os.path.join(cache_dir, "X_train_scaled.npy")
    val_path = os.path.join(cache_dir, "X_val_scaled.npy")
    test_path = os.path.join(cache_dir, "X_test_scaled.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            logger.info("Loading scaled tabular features from cache...")
            try:
                X_train = np.load(train_path)
                X_val = np.load(val_path)
                X_test = np.load(test_path)
                return X_train, X_val, X_test
            except Exception as e:
                logger.warning(f"Failed to load numpy cache: {e}. Recomputing...")

    logger.info("Computing scaled tabular features...")
    # Load DataFrames
    df_train, df_val, df_test = data_handler.load_datasets(
        load_cached_data=load_cached_data
    )

    # Initialize and fit scaler
    scaler = TabularScaler()
    scaler.fit(df_train)

    # Transform
    X_train = scaler.transform(df_train)
    X_val = scaler.transform(df_val)
    X_test = scaler.transform(df_test)

    # Save to cache
    logger.info("Caching scaled features...")
    np.save(train_path, X_train)
    np.save(val_path, X_val)
    np.save(test_path, X_test)

    return X_train, X_val, X_test
