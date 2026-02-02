import os
import pandas as pd
import numpy as np
from library.utils import setup_logger

logger = setup_logger("DataLoader")


def load_dataset(
    split,
    base_dir="./metadata",
    cache_dir="./working/idea_1",
    process_context=False,
    load_cached_data=True,
    sample_ratio=1.0,
    seed=42,
):
    """
    Loads the dataset for a specific split. Can optionally process the data to add
    contextual features (previous token) and cache the result.

    Args:
        split (str): The dataset split ('train', 'val', 'test').
        base_dir (str): Directory containing the raw metadata parquet files.
        cache_dir (str): Directory to store/load cached processed files.
        process_context (bool): If True, adds 'prev_before' column using sentence context.
        load_cached_data (bool): If True, tries to load from cache when process_context is True.
        sample_ratio (float): Fraction of sentences to load (0.0 < sample_ratio <= 1.0).
        seed (int): Random seed for reproducibility during sampling.

    Returns:
        pd.DataFrame: The loaded (and potentially processed) dataset.
    """
    # Construct paths
    raw_path = os.path.join(base_dir, f"{split}.parquet")
    processed_filename = f"{split}_processed.parquet"
    cache_path = os.path.join(cache_dir, processed_filename)

    # 1. Try Loading from Cache (only if processing is requested and we want full data)
    # We avoid caching sampled data to prevent confusion between partial and full datasets.
    if process_context and load_cached_data and sample_ratio == 1.0:
        if os.path.exists(cache_path):
            logger.info(f"Loading cached processed data from {cache_path}")
            df = pd.read_parquet(cache_path)
            # Check if we have the new column 'next_before'
            if "next_before" not in df.columns:
                logger.info("Cached data missing 'next_before'. Recomputing...")
            else:
                # Ensure 'before' is string to prevent NoneType errors in regex
                if "before" in df.columns:
                    df["before"] = df["before"].fillna("").astype(str)
                return df

    # 2. Load Raw Data
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_parquet(raw_path)

    # Ensure 'before' is string to prevent NoneType errors in regex
    if "before" in df.columns:
        df["before"] = df["before"].fillna("").astype(str)

    # 3. Apply Sampling (if requested)
    # We sample by sentence_id to preserve context integrity (tokens within a sentence must stay together)
    if sample_ratio < 1.0:
        logger.info(f"Sampling {sample_ratio*100:.2f}% of sentences...")
        if "sentence_id" not in df.columns:
            logger.warning(
                "sentence_id column missing. Cannot perform sentence-level sampling."
            )
        else:
            unique_sentences = df["sentence_id"].unique()
            rng = np.random.RandomState(seed)

            # Calculate number of sentences to keep
            n_keep = int(len(unique_sentences) * sample_ratio)
            if n_keep > 0:
                keep_ids = rng.choice(unique_sentences, size=n_keep, replace=False)
                # Filter dataframe
                df = df[df["sentence_id"].isin(keep_ids)].copy()
            else:
                logger.warning(
                    "Sample ratio resulted in 0 sentences. Returning empty DataFrame."
                )
                df = df.iloc[0:0].copy()

    # 4. Process Context (if requested)
    if process_context:
        logger.info("Processing data: Adding 'prev_before' context...")

        # Ensure data is sorted by sentence and token id to guarantee correct shifting
        if "sentence_id" in df.columns and "token_id" in df.columns:
            df = df.sort_values(["sentence_id", "token_id"])

        # Add previous token column
        # Group by sentence_id and shift 'before' column by 1
        # fillna("<START>") marks the beginning of a sentence
        df["prev_before"] = (
            df.groupby("sentence_id")["before"].shift(1).fillna("<START>")
        )

        # Add next token column
        # Group by sentence_id and shift 'before' column by -1
        # fillna("<END>") marks the end of a sentence
        df["next_before"] = (
            df.groupby("sentence_id")["before"].shift(-1).fillna("<END>")
        )

        # 5. Save to Cache (only if full data)
        if sample_ratio == 1.0:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Caching processed data to {cache_path}")
            df.to_parquet(cache_path, index=False)

    return df


def get_sentence_iterator(df):
    """
    Creates an iterator that yields data grouped by sentence.

    Args:
        df (pd.DataFrame): The dataset containing a 'sentence_id' column.

    Yields:
        tuple: (sentence_id, DataFrame of tokens in that sentence)
    """
    # We assume the dataframe is already sorted by sentence_id and token_id.
    # We use sort=False for performance as we don't need to re-sort groups.
    return df.groupby("sentence_id", sort=False)
