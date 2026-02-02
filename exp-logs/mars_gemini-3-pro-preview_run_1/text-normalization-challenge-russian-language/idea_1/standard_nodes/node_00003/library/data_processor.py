import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import setup_logger, load_or_process

# Initialize logger
logger = setup_logger("DataProcessor")


def load_raw_data(
    path: str, is_test: bool = False, max_samples: int = None
) -> pd.DataFrame:
    """
    Loads raw data from CSV, handles missing values, and enforces data types.
    Supports subsampling by sentence_id for debugging.
    """
    logger.info(f"Loading raw data from {path}...")

    # Define dtypes to ensure text is read as string, not int/float
    dtype_dict = {
        Config.SENTENCE_ID_COL: "int32",
        Config.TOKEN_ID_COL: "int32",
        Config.INPUT_COL: "object",
    }

    if not is_test:
        dtype_dict[Config.TARGET_COL] = "object"
        dtype_dict[Config.CLASS_COL] = "object"

    try:
        df = pd.read_csv(path, dtype=dtype_dict)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise

    # Fill NaNs (e.g., if "null" or "nan" was in the text)
    df[Config.INPUT_COL] = df[Config.INPUT_COL].fillna("")
    if not is_test:
        df[Config.TARGET_COL] = df[Config.TARGET_COL].fillna("")
        df[Config.CLASS_COL] = df[Config.CLASS_COL].fillna("UNKNOWN")

    # Subsampling for debugging
    if max_samples is not None:
        logger.info(f"Subsampling: Keeping first {max_samples} sentences.")
        unique_sentences = df[Config.SENTENCE_ID_COL].unique()
        if len(unique_sentences) > max_samples:
            keep_ids = unique_sentences[:max_samples]
            df = df[df[Config.SENTENCE_ID_COL].isin(keep_ids)].copy()

    logger.info(f"Loaded {len(df)} rows.")
    return df


def add_sequence_boundaries(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Adds <BOS> and <EOS> tokens to each sentence using vectorized operations.
    Sorts the result to ensure correct sequence order (BOS -> tokens -> EOS).
    """
    logger.info("Adding sequence boundaries (<BOS>, <EOS>)...")

    # Get unique sentence IDs
    sentence_ids = df[Config.SENTENCE_ID_COL].unique()

    # --- Create BOS DataFrame ---
    # token_id = -1 ensures it comes before 0
    df_bos = pd.DataFrame(
        {
            Config.SENTENCE_ID_COL: sentence_ids,
            Config.TOKEN_ID_COL: -1,
            Config.INPUT_COL: Config.BOS_TOKEN,
        }
    )

    if not is_test:
        df_bos[Config.TARGET_COL] = Config.BOS_TOKEN
        df_bos[Config.CLASS_COL] = Config.BOS_TOKEN  # Or specific class for boundary

    # --- Create EOS DataFrame ---
    # We need to find the max token_id for each sentence to place EOS after it.
    # Groupby transform is expensive on large data, but necessary for variable lengths.
    # Optimization: We can just use a very large number if we re-index later,
    # but to be safe and clean, we calculate max + 1.
    max_token_ids = df.groupby(Config.SENTENCE_ID_COL)[Config.TOKEN_ID_COL].max()

    # Map max_ids back to sentence_ids order
    # We can construct df_eos directly from the series index and values
    df_eos = pd.DataFrame(
        {
            Config.SENTENCE_ID_COL: max_token_ids.index,
            Config.TOKEN_ID_COL: max_token_ids.values + 1,
            Config.INPUT_COL: Config.EOS_TOKEN,
        }
    )

    if not is_test:
        df_eos[Config.TARGET_COL] = Config.EOS_TOKEN
        df_eos[Config.CLASS_COL] = Config.EOS_TOKEN

    # --- Concatenate and Sort ---
    # Combine original, BOS, and EOS
    df_final = pd.concat([df, df_bos, df_eos], ignore_index=True)

    # Sort by sentence_id then token_id to reconstruct the sequence
    df_final = df_final.sort_values(
        by=[Config.SENTENCE_ID_COL, Config.TOKEN_ID_COL], ascending=[True, True]
    )

    # Reset index for clean dataframe
    df_final = df_final.reset_index(drop=True)

    logger.info(f"Processed sequences. New shape: {df_final.shape}")
    return df_final


def _process_train_wrapper():
    """Wrapper to load and process training data for caching."""
    df = load_raw_data(
        Config.TRAIN_DATA_PATH, is_test=False, max_samples=Config.MAX_TRAIN_SAMPLES
    )
    return add_sequence_boundaries(df, is_test=False)


def _process_val_wrapper():
    """Wrapper to load and process validation data for caching."""
    # We apply MAX_TRAIN_SAMPLES to validation as well if set, to keep debug runs fast
    df = load_raw_data(
        Config.VAL_DATA_PATH, is_test=False, max_samples=Config.MAX_TRAIN_SAMPLES
    )
    return add_sequence_boundaries(df, is_test=False)


def _process_test_wrapper():
    """Wrapper to load and process test data for caching."""
    # Usually we don't subsample test data for submission, but for local debug we might.
    # However, Config.MAX_TRAIN_SAMPLES implies training constraint.
    # We will load full test set unless specifically constrained, but here we load full.
    df = load_raw_data(Config.TEST_DATA_PATH, is_test=True, max_samples=None)
    return add_sequence_boundaries(df, is_test=True)


def get_data(split: str, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Main entry point to get processed data.
    Uses caching to avoid re-computing boundaries.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from parquet cache.

    Returns:
        pd.DataFrame: The processed dataframe with BOS/EOS tokens.
    """
    if split == "train":
        return load_or_process(
            Config.TRAIN_CACHE_PATH,
            _process_train_wrapper,
            load_cached_data=load_cached_data,
        )
    elif split == "val":
        return load_or_process(
            Config.VAL_CACHE_PATH,
            _process_val_wrapper,
            load_cached_data=load_cached_data,
        )
    elif split == "test":
        return load_or_process(
            Config.TEST_CACHE_PATH,
            _process_test_wrapper,
            load_cached_data=load_cached_data,
        )
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'val', or 'test'.")
