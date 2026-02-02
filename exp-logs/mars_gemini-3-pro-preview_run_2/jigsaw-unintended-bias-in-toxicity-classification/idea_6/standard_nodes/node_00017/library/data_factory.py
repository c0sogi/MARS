import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from library.config import Config
from library.utils import ensure_directory


def load_tokenizer():
    """
    Loads the tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)


def _read_and_merge_data(metadata_path, text_source_path, split_name="data"):
    """
    Helper function to read metadata, read source text, and merge them.
    """
    print(f"Loading {split_name} metadata from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    if Config.DEBUG:
        print(
            f"DEBUG mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows from {split_name}."
        )
        meta_df = meta_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    print(f"Loading source text from {text_source_path}...")
    # Only read ID and Text to save memory
    text_df = pd.read_csv(text_source_path, usecols=[Config.ID_COL, Config.TEXT_COL])

    # Fill NaNs in text
    text_df[Config.TEXT_COL] = text_df[Config.TEXT_COL].fillna("")

    print(f"Merging {split_name} metadata with text...")
    merged_df = meta_df.merge(text_df, on=Config.ID_COL, how="inner")

    return merged_df


def _tokenize_texts(texts, tokenizer):
    """
    Tokenizes a list/series of texts using the provided tokenizer.
    Returns input_ids and attention_masks as numpy arrays.
    """
    print(f"Tokenizing {len(texts)} texts...")
    encoding = tokenizer.batch_encode_plus(
        texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LEN,
        return_tensors="np",
        return_attention_mask=True,
    )

    # Cast to smaller types to save memory/disk space
    input_ids = encoding["input_ids"].astype(np.int32)
    attention_masks = encoding["attention_mask"].astype(np.int8)

    return input_ids, attention_masks


def process_and_cache_data(load_cached_data=True):
    """
    Main data processing function.
    Checks for cached .npy files. If found and load_cached_data is True, loads them.
    Otherwise, processes raw data, computes features/weights, saves to cache, and returns them.

    Returns:
        dict: A dictionary containing loaded numpy arrays for train, val, and test.
    """
    ensure_directory(Config.WORKING_DIR)

    # Define cache file paths map for easy checking
    cache_files = {
        "train_input_ids": Config.CACHE_TRAIN_INPUT_IDS,
        "train_attn_masks": Config.CACHE_TRAIN_ATTN_MASKS,
        "train_targets": Config.CACHE_TRAIN_TARGETS,
        "train_aux_targets": Config.CACHE_TRAIN_AUX_TARGETS,
        "train_weights": Config.CACHE_TRAIN_SAMPLE_WEIGHTS,
        "val_input_ids": Config.CACHE_VAL_INPUT_IDS,
        "val_attn_masks": Config.CACHE_VAL_ATTN_MASKS,
        "val_targets": Config.CACHE_VAL_TARGETS,
        "val_aux_targets": Config.CACHE_VAL_AUX_TARGETS,
        "val_ids": Config.CACHE_VAL_IDS,
        "test_input_ids": Config.CACHE_TEST_INPUT_IDS,
        "test_attn_masks": Config.CACHE_TEST_ATTN_MASKS,
        "test_ids": Config.CACHE_TEST_IDS,
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(path) for path in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading cached data from disk...")
        data = {}
        for key, path in cache_files.items():
            data[key] = np.load(path)
        print("Data loaded successfully.")
        return data

    print("Cache missing or reload requested. Processing data from scratch...")
    tokenizer = load_tokenizer()

    # --------------------------------------------------------------------------
    # 1. Process Training Data
    # --------------------------------------------------------------------------
    train_df = _read_and_merge_data(
        Config.TRAIN_METADATA_PATH, Config.TRAIN_TEXT_SOURCE, "Train"
    )

    # Tokenize
    train_input_ids, train_attn_masks = _tokenize_texts(
        train_df[Config.TEXT_COL], tokenizer
    )

    # Extract Targets
    train_targets = train_df[Config.TARGET_COL].values.astype(np.float32)

    # Extract Aux Targets (Identities)
    train_aux_targets = (
        train_df[Config.IDENTITY_COLS].fillna(0).values.astype(np.float32)
    )

    # Compute Sample Weights
    # Logic: If any identity is mentioned (sum > 0), boost weight. Else 1.0.
    identity_sum = train_df[Config.IDENTITY_COLS].fillna(0).sum(axis=1).values
    train_weights = np.where(
        identity_sum > 0, Config.IDENTITY_WEIGHT_BOOST, 1.0
    ).astype(np.float32)

    # Save Train Cache
    np.save(Config.CACHE_TRAIN_INPUT_IDS, train_input_ids)
    np.save(Config.CACHE_TRAIN_ATTN_MASKS, train_attn_masks)
    np.save(Config.CACHE_TRAIN_TARGETS, train_targets)
    np.save(Config.CACHE_TRAIN_AUX_TARGETS, train_aux_targets)
    np.save(Config.CACHE_TRAIN_SAMPLE_WEIGHTS, train_weights)

    # Free memory
    del train_df

    # --------------------------------------------------------------------------
    # 2. Process Validation Data
    # --------------------------------------------------------------------------
    val_df = _read_and_merge_data(
        Config.VAL_METADATA_PATH, Config.TRAIN_TEXT_SOURCE, "Validation"
    )

    val_input_ids, val_attn_masks = _tokenize_texts(val_df[Config.TEXT_COL], tokenizer)
    val_targets = val_df[Config.TARGET_COL].values.astype(np.float32)
    val_aux_targets = val_df[Config.IDENTITY_COLS].fillna(0).values.astype(np.float32)
    val_ids = val_df[Config.ID_COL].values.astype(
        np.int64
    )  # Keep IDs for bias metric calc

    # Save Val Cache
    np.save(Config.CACHE_VAL_INPUT_IDS, val_input_ids)
    np.save(Config.CACHE_VAL_ATTN_MASKS, val_attn_masks)
    np.save(Config.CACHE_VAL_TARGETS, val_targets)
    np.save(Config.CACHE_VAL_AUX_TARGETS, val_aux_targets)
    np.save(Config.CACHE_VAL_IDS, val_ids)

    del val_df

    # --------------------------------------------------------------------------
    # 3. Process Test Data
    # --------------------------------------------------------------------------
    test_df = _read_and_merge_data(
        Config.TEST_METADATA_PATH, Config.TEST_TEXT_SOURCE, "Test"
    )

    test_input_ids, test_attn_masks = _tokenize_texts(
        test_df[Config.TEXT_COL], tokenizer
    )
    test_ids = test_df[Config.ID_COL].values.astype(np.int64)

    # Save Test Cache
    np.save(Config.CACHE_TEST_INPUT_IDS, test_input_ids)
    np.save(Config.CACHE_TEST_ATTN_MASKS, test_attn_masks)
    np.save(Config.CACHE_TEST_IDS, test_ids)

    del test_df

    print("Data processing complete. All files cached.")

    # Return dictionary of in-memory arrays
    return {
        "train_input_ids": train_input_ids,
        "train_attn_masks": train_attn_masks,
        "train_targets": train_targets,
        "train_aux_targets": train_aux_targets,
        "train_weights": train_weights,
        "val_input_ids": val_input_ids,
        "val_attn_masks": val_attn_masks,
        "val_targets": val_targets,
        "val_aux_targets": val_aux_targets,
        "val_ids": val_ids,
        "test_input_ids": test_input_ids,
        "test_attn_masks": test_attn_masks,
        "test_ids": test_ids,
    }
