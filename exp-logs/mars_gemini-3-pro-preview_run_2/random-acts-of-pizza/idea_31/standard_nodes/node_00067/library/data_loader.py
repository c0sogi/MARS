import os
import pandas as pd
import numpy as np
from library.config import Config, load_data, get_text_embeddings, get_metadata_features
from library.utils import setup_logger


class Embedder:
    """
    Manages the generation and caching of text embeddings for the Tri-Backbone architecture.
    """

    def __init__(self):
        self.logger = setup_logger(
            "Embedder", os.path.join(Config.WORKING_DIR, "data_loader.log")
        )
        self.models = Config.MODELS
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

    def generate(
        self, df: pd.DataFrame, split_name: str, load_cached_data: bool = True
    ) -> dict:
        """
        Generates embeddings for all configured backbones (Anchor, Aux1, Aux2).

        Args:
            df (pd.DataFrame): The dataframe containing text data.
            split_name (str): 'train' or 'test', used for cache file naming.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            dict: A dictionary where keys are model keys (e.g., 'anchor') and values are numpy arrays of embeddings.
        """
        embeddings = {}
        self.logger.info(f"Generating embeddings for split: {split_name}")

        for key, model_name in self.models.items():
            # Construct a unique cache prefix: e.g., "train_anchor"
            cache_prefix = f"{split_name}_{key}"

            # Use the library function which handles model loading, encoding, and caching
            emb = get_text_embeddings(
                df=df,
                model_name=model_name,
                cache_prefix=cache_prefix,
                load_cached_data=load_cached_data,
            )
            embeddings[key] = emb

        return embeddings


def load_datasets(load_cached_data: bool = True, debug_sample_size: int = None) -> dict:
    """
    Loads raw data, extracts metadata, and generates embeddings.

    Args:
        load_cached_data (bool): Whether to use cached embeddings.
        debug_sample_size (int, optional): If set, truncates the dataset for debugging.

    Returns:
        dict: Contains 'df_train', 'df_test', 'y_train', 'meta_train', 'meta_test',
              'train_embeddings', 'test_embeddings'.
    """
    logger = setup_logger(
        "DataLoader", os.path.join(Config.WORKING_DIR, "data_loader.log")
    )
    logger.info("Starting data loading process...")

    # 1. Load Raw Data (Merged with Metadata)
    # load_data() returns (df_full_train, df_test)
    df_train, df_test = load_data()

    # 2. Debugging: Subsample if requested
    if debug_sample_size is not None:
        logger.info(
            f"Debugging mode enabled. Subsampling to {debug_sample_size} records."
        )
        df_train = df_train.iloc[:debug_sample_size].reset_index(drop=True)
        df_test = df_test.iloc[:debug_sample_size].reset_index(drop=True)

    # 3. Extract Targets
    y_train = df_train["requester_received_pizza"].values.astype(int)

    # 4. Extract Tabular Metadata
    logger.info("Extracting metadata features...")
    meta_train = get_metadata_features(df_train)
    meta_test = get_metadata_features(df_test)

    # 5. Generate/Load Embeddings
    embedder = Embedder()

    logger.info("Processing Training Embeddings...")
    train_embeddings = embedder.generate(
        df_train, "train", load_cached_data=load_cached_data
    )

    logger.info("Processing Test Embeddings...")
    test_embeddings = embedder.generate(
        df_test, "test", load_cached_data=load_cached_data
    )

    logger.info("Data loading complete.")

    return {
        "df_train": df_train,
        "df_test": df_test,
        "y_train": y_train,
        "meta_train": meta_train,
        "meta_test": meta_test,
        "train_embeddings": train_embeddings,
        "test_embeddings": test_embeddings,
    }
