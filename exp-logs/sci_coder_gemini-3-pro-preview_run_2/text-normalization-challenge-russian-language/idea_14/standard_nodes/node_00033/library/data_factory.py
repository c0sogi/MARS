import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed
from library.transformer_model import NormalizationDataset


def _add_context(df):
    """
    Adds 'prev' and 'next' context columns to the dataframe.
    Respects sentence boundaries using sentence_id.
    """
    # Ensure sorting
    if "sentence_id" in df.columns and "token_id" in df.columns:
        df = df.sort_values(["sentence_id", "token_id"])

    # Shift for context
    df["prev"] = df["before"].shift(1)
    df["next"] = df["before"].shift(-1)

    # Handle boundaries
    if "sentence_id" in df.columns:
        # Start of sentence
        start_mask = df["sentence_id"] != df["sentence_id"].shift(1)
        start_mask.iloc[0] = True  # First row is always start
        df.loc[start_mask, "prev"] = "<START>"

        # End of sentence
        end_mask = df["sentence_id"] != df["sentence_id"].shift(-1)
        end_mask.iloc[-1] = True  # Last row is always end
        df.loc[end_mask, "next"] = "<END>"

    # Fill remaining NaNs (e.g. single token sentences or edge cases)
    df["prev"] = df["prev"].fillna("<START>")
    df["next"] = df["next"].fillna("<END>")

    return df


def _process_dataframe(df, is_train=True):
    """
    Applies filtering and upsampling logic.
    """
    # 1. Add Context
    df = _add_context(df)

    # 2. Filter for Semiotic Tokens (Digits or Latin)
    # We want to focus the neural network on hard cases, not simple words.
    # Regex: \d for digits, [a-zA-Z] for latin
    semiotic_mask = df["before"].astype(str).str.contains(r"\d|[a-zA-Z]", regex=True)
    df_filtered = df[semiotic_mask].copy()

    if is_train:
        # 3. Upsample Rare Classes (Train only)
        if "class" in df_filtered.columns:
            rare_classes = ["MONEY", "DECIMAL", "TELEPHONE", "ELECTRONIC", "DIGIT"]
            dfs_to_concat = [df_filtered]

            for cls in rare_classes:
                subset = df_filtered[df_filtered["class"] == cls]
                if len(subset) > 0:
                    # Upsample 5x
                    dfs_to_concat.extend([subset] * 5)

            df_filtered = pd.concat(dfs_to_concat)

        # 4. Shuffle
        df_filtered = df_filtered.sample(
            frac=1.0, random_state=Config.SEED
        ).reset_index(drop=True)

        # Debugging hook
        if Config.DEBUG:
            df_filtered = df_filtered.head(Config.DEBUG_SIZE)

    else:
        # For validation, we also filter to semiotic tokens to measure relevant accuracy,
        # but we drop duplicates to avoid skewing metrics with repeated sentences if any.
        # (Though metadata split should be clean, safe to ensure uniqueness).
        df_filtered = df_filtered.drop_duplicates(subset=["sentence_id", "token_id"])

        if Config.DEBUG:
            df_filtered = df_filtered.head(Config.DEBUG_SIZE)

    return df_filtered


def get_processed_data(file_path, is_train, load_cached_data=True):
    """
    Loads raw data, processes it, and handles caching.
    """
    # Define cache path
    filename = "processed_train.parquet" if is_train else "processed_val.parquet"
    cache_dir = os.path.join(Config.CACHE_DIR, "transformer_data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)

    # Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(
            f"Loading processed {'training' if is_train else 'validation'} data from cache: {cache_path}"
        )
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # Compute from scratch
    print(f"Processing {'training' if is_train else 'validation'} data from scratch...")

    # Load raw
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Ensure string types
    df["before"] = df["before"].fillna("").astype(str)
    if "after" in df.columns:
        df["after"] = df["after"].fillna("").astype(str)

    # Process
    df_proc = _process_dataframe(df, is_train=is_train)

    # Save to cache
    print(f"Saving processed data to cache: {cache_path}")
    df_proc.to_parquet(cache_path)

    return df_proc


def create_dataloaders(tokenizer, hfbb_model, load_cached_data=True):
    """
    Creates Train and Validation DataLoaders.

    Args:
        tokenizer: Instance of HybridTokenizer.
        hfbb_model: Instance of HierarchicalBackoff (used for soft-residual weighting).
        load_cached_data (bool): Whether to use cached intermediate dataframes.

    Returns:
        tuple: (train_loader, val_loader)
    """
    set_seed(Config.SEED)

    # 1. Get Processed Dataframes
    df_train = get_processed_data(
        Config.TRAIN_DATA, is_train=True, load_cached_data=load_cached_data
    )
    df_val = get_processed_data(
        Config.VAL_DATA, is_train=False, load_cached_data=load_cached_data
    )

    print(f"Final Dataset Sizes - Train: {len(df_train)}, Val: {len(df_val)}")

    # 2. Create Datasets
    # Pass hfbb_model to train dataset to enable weight computation
    train_dataset = NormalizationDataset(
        df_train, tokenizer, hfbb_model=hfbb_model, is_train=True
    )

    # For validation, we don't need weights, so we can pass hfbb_model=None to save compute
    # However, is_train=True is required to include targets ('after') for evaluation
    val_dataset = NormalizationDataset(
        df_val, tokenizer, hfbb_model=None, is_train=True
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
