import os
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    WORK_DIR,
    PLAIN_SAMPLE_RATIO,
    SEED,
)


def create_context_windows(df):
    """
    Groups tokens by sentence_id and extracts surrounding context (prev/next tokens).
    Adds 'prev_token' and 'next_token' columns to the dataframe.
    """
    # Ensure data is sorted by sentence and token id to guarantee sequence
    # (Assuming input is already sorted, but good to be safe if overhead is low.
    # Given dataset size, we assume input is sorted as per metadata generation)

    # Convert to string and handle NaNs
    tokens = df["before"].astype(str).fillna("")
    sentence_ids = df["sentence_id"].values

    # Calculate Previous Tokens
    prev_tokens = tokens.shift(1).fillna("")
    # Mask where sentence_id changes (start of new sentence)
    # If sentence_id[i] != sentence_id[i-1], then prev_token is invalid
    is_start = sentence_ids != np.roll(sentence_ids, 1)
    is_start[0] = True  # First element is always start
    prev_tokens[is_start] = ""

    # Calculate Next Tokens
    next_tokens = tokens.shift(-1).fillna("")
    # Mask where sentence_id changes (end of sentence)
    # If sentence_id[i] != sentence_id[i+1], then next_token is invalid
    is_end = sentence_ids != np.roll(sentence_ids, -1)
    is_end[-1] = True  # Last element is always end
    next_tokens[is_end] = ""

    # Assign columns
    df["prev_token"] = prev_tokens
    df["next_token"] = next_tokens

    return df


def downsample_training_data(df, ratio=PLAIN_SAMPLE_RATIO, seed=SEED):
    """
    Downsamples the majority 'PLAIN' class to balance the dataset.
    Retains all tokens from other classes.
    """
    if ratio is None or ratio >= 1.0:
        return df

    # Separate classes
    df_plain = df[df["class"] == "PLAIN"]
    df_others = df[df["class"] != "PLAIN"]

    # Sample PLAIN class
    df_plain_sampled = df_plain.sample(frac=ratio, random_state=seed)

    # Combine and shuffle
    df_balanced = pd.concat([df_others, df_plain_sampled], axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df_balanced


def load_train_data(load_cached_data=True, downsample_ratio=PLAIN_SAMPLE_RATIO):
    """
    Loads training data.
    1. Checks cache (parquet).
    2. If no cache: Loads CSV, creates context, downsamples, saves cache.
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    cache_path = os.path.join(WORK_DIR, "train_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed training data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Loading raw training data from {TRAIN_FILE}...")
    df = pd.read_csv(
        TRAIN_FILE,
        keep_default_na=False,
        dtype={
            "sentence_id": "int32",
            "token_id": "int32",
            "class": "category",
            "before": "object",
            "after": "object",
        },
    )

    print("Creating context windows...")
    df = create_context_windows(df)

    print(f"Downsampling PLAIN class (ratio={downsample_ratio})...")
    df = downsample_training_data(df, ratio=downsample_ratio)

    print(f"Saving processed training data to {cache_path}...")
    df.to_parquet(cache_path)

    return df


def load_val_data(load_cached_data=True):
    """
    Loads validation data.
    1. Checks cache (parquet).
    2. If no cache: Loads CSV, creates context, saves cache.
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    cache_path = os.path.join(WORK_DIR, "val_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed validation data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Loading raw validation data from {VAL_FILE}...")
    df = pd.read_csv(
        VAL_FILE,
        keep_default_na=False,
        dtype={
            "sentence_id": "int32",
            "token_id": "int32",
            "class": "category",
            "before": "object",
            "after": "object",
        },
    )

    print("Creating context windows...")
    df = create_context_windows(df)

    print(f"Saving processed validation data to {cache_path}...")
    df.to_parquet(cache_path)

    return df


def load_test_data(load_cached_data=True):
    """
    Loads test data.
    1. Checks cache (parquet).
    2. If no cache: Loads CSV, creates context, saves cache.
    """
    os.makedirs(WORK_DIR, exist_ok=True)
    cache_path = os.path.join(WORK_DIR, "test_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed test data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Loading raw test data from {TEST_FILE}...")
    df = pd.read_csv(
        TEST_FILE,
        keep_default_na=False,
        dtype={"sentence_id": "int32", "token_id": "int32", "before": "object"},
    )

    print("Creating context windows...")
    df = create_context_windows(df)

    print(f"Saving processed test data to {cache_path}...")
    df.to_parquet(cache_path)

    return df
