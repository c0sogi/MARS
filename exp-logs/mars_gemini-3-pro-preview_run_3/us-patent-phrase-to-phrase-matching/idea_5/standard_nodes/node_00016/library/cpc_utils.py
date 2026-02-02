import os
import pandas as pd
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("cpc_utils")


def get_cpc_texts():
    """
    Returns the dictionary mapping CPC codes to their textual descriptions.
    Wraps the dictionary defined in Config.
    """
    return Config.cpc_codes


def load_context_enriched_data(
    data_type: str, load_cached_data: bool = True, nrows: int = None
) -> pd.DataFrame:
    """
    Loads the dataset (train, val, or test), enriches it with CPC context descriptions,
    and handles caching to Parquet format.

    Args:
        data_type (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.
        nrows (int, optional): Number of rows to load (for debugging). If set, caching is disabled.

    Returns:
        pd.DataFrame: The enriched dataframe with a new 'context_text' column.
    """
    # Determine paths based on data_type
    if data_type == "train":
        input_path = Config.train_path
        cache_path = Config.train_cache_path
    elif data_type == "val":
        input_path = Config.val_path
        cache_path = Config.val_cache_path
    elif data_type == "test":
        input_path = Config.test_path
        cache_path = Config.test_cache_path
    else:
        raise ValueError(
            f"Invalid data_type: {data_type}. Must be 'train', 'val', or 'test'."
        )

    # Ensure working directory exists for cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Caching Logic:
    # 1. If nrows is set (debug mode), we skip loading from cache to ensure we get the exact requested subset from source.
    # 2. If nrows is None, we respect load_cached_data flag.
    if nrows is None and load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {data_type} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Reloading from source.")

    # Load from source
    logger.info(f"Loading raw {data_type} data from {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, nrows=nrows)

    # Enrich with Context Descriptions
    logger.info("Enriching data with CPC context descriptions...")
    cpc_codes = get_cpc_texts()

    # Map context codes to descriptions.
    # We use .map() for efficiency and fillna("") to handle any potential missing codes gracefully.
    if "context" in df.columns:
        df["context_text"] = df["context"].map(cpc_codes).fillna("")
    else:
        logger.warning(
            f"Column 'context' not found in {data_type} data. Skipping enrichment."
        )
        df["context_text"] = ""

    # Save to cache only if we loaded the full dataset (nrows is None)
    # We do not cache partial/debug datasets to avoid overwriting the full cache.
    if nrows is None:
        logger.info(f"Saving enriched {data_type} data to cache at {cache_path}")
        df.to_parquet(cache_path, index=False)
    else:
        logger.info(f"Debug mode (nrows={nrows}): Skipping cache save.")

    return df
