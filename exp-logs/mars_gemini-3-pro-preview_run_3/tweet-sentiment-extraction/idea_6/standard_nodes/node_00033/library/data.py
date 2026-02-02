import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from library.config import Config
from library.model import TweetDataset, process_data


def get_loaders(
    tokenizer, batch_size=Config.TRAIN_BATCH_SIZE, load_cached_data=True, debug=False
):
    """
    Creates DataLoaders for the fixed Train/Val split (from metadata) and Test set.
    Filters out 'neutral' sentiment rows from Train and Val sets as per the strategy
    to focus the model on extracting spans for positive/negative sentiments.

    Args:
        tokenizer: Transformers tokenizer instance.
        batch_size (int): Batch size for the training loader.
        load_cached_data (bool): Whether to attempt loading processed data from cache.
        debug (bool): If True, loads a small subset of data for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Define cache names based on debug flag to prevent collisions between debug/full runs
    suffix = "_debug" if debug else ""
    train_cache = f"train_fixed_no_neutral{suffix}"
    val_cache = f"val_fixed_no_neutral{suffix}"
    test_cache = f"test_fixed{suffix}"

    # 1. Load Metadata
    # Using the pre-split metadata files
    df_train = pd.read_csv(Config.TRAIN_FILE)
    df_val = pd.read_csv(Config.VAL_FILE)
    df_test = pd.read_csv(Config.TEST_FILE)

    if debug:
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Filter Neutrals (Strategy Requirement)
    # The model is designed to extract spans for positive/negative sentiments.
    # Neutral tweets are handled by a deterministic rule (return full text) during inference,
    # so we exclude them from the trainable data to save capacity and reduce noise.
    df_train = df_train[df_train["sentiment"] != "neutral"].reset_index(drop=True)
    df_val = df_val[df_val["sentiment"] != "neutral"].reset_index(drop=True)

    # 3. Process Data
    # process_data (from library.model) handles:
    # - Tokenization (without whitespace normalization)
    # - Mask-Based Overlap target generation
    # - Alignment filtering
    # - Caching (saving/loading .npy files in Config.WORKING_DIR)

    # Process Train
    train_out = process_data(
        df_train,
        tokenizer,
        Config.MAX_LEN,
        train_cache,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    # Unpack: input_ids, attention_masks, start_tokens, end_tokens, offsets_list, valid_indices
    train_ids, train_masks, train_start, train_end, _, _ = train_out

    # Process Val
    val_out = process_data(
        df_val,
        tokenizer,
        Config.MAX_LEN,
        val_cache,
        load_cached_data=load_cached_data,
        is_test=False,
    )
    val_ids, val_masks, val_start, val_end, _, _ = val_out

    # Process Test
    test_out = process_data(
        df_test,
        tokenizer,
        Config.MAX_LEN,
        test_cache,
        load_cached_data=load_cached_data,
        is_test=True,
    )
    # Unpack: input_ids, attention_masks, None, None, offsets_list, None
    test_ids, test_masks, _, _, _, _ = test_out

    # 4. Create Datasets
    # Using the TweetDataset class imported from library.model
    train_ds = TweetDataset(train_ids, train_masks, train_start, train_end)
    val_ds = TweetDataset(val_ids, val_masks, val_start, val_end)
    test_ds = TweetDataset(test_ids, test_masks)

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader
