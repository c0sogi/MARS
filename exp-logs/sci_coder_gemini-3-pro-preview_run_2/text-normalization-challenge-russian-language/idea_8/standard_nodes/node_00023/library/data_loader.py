import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset_builder import DatasetBuilder
from library.text_processing import build_tokenizers
from library.transformer_model import ResidualDataset, collate_fn

# Alias the provided ResidualDataset to NormalizationDataset as requested by the task description
NormalizationDataset = ResidualDataset


def get_dataloaders(load_cached_data=True, batch_size=None, num_workers=None):
    """
    Orchestrates the creation of DataLoaders for the Transformer model.

    This function manages:
    1. Loading or training tokenizers (Char-level for input, BPE for target).
    2. Loading or building the curriculum-enriched residual datasets.
    3. Creating PyTorch DataLoaders with appropriate collation.

    Args:
        load_cached_data (bool): If True, attempts to load pre-computed data/tokenizers from disk.
                                 If False, forces re-computation.
        batch_size (int, optional): Batch size for the DataLoaders. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of worker processes. Defaults to Config.NUM_WORKERS.

    Returns:
        tuple: (train_loader, val_loader, char_tokenizer, target_tokenizer)
    """
    # 1. Configure Defaults
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS

    print(f"DataLoader: Preparing data (load_cached_data={load_cached_data})...")

    # 2. Handle Tokenizers
    # We need tokenizers before we can create the Dataset objects.
    # If cache is missing or ignored, we must load the FULL training data to ensure
    # the vocabulary is robust (covering all tokens, not just residuals).

    char_vocab_exists = os.path.exists(Config.CHAR_VOCAB_PATH)
    bpe_model_exists = os.path.exists(Config.BPE_MODEL_PATH)

    tokenizer_train_df = None

    # Determine if we need to load raw data for tokenizers
    if not load_cached_data or not (char_vocab_exists and bpe_model_exists):
        print(
            "DataLoader: Tokenizer cache missing or ignored. Loading full training data for tokenizer fitting..."
        )
        try:
            tokenizer_train_df = pd.read_csv(Config.TRAIN_FILE)
            # Ensure columns are treated as strings
            tokenizer_train_df["before"] = (
                tokenizer_train_df["before"].fillna("").astype(str)
            )
            tokenizer_train_df["after"] = (
                tokenizer_train_df["after"].fillna("").astype(str)
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find training data at {Config.TRAIN_FILE} to build tokenizers."
            )

    # Build or Load Tokenizers
    char_tokenizer, target_tokenizer = build_tokenizers(
        train_df=tokenizer_train_df, load_cached_data=load_cached_data
    )

    # 3. Handle Datasets (Residuals & Anchors)
    # DatasetBuilder handles the caching logic for the processed dataframes (parquet files).
    builder = DatasetBuilder()
    train_df, val_df = builder.build_dataset(load_cached_data=load_cached_data)

    print(f"DataLoader: Instantiating Datasets...")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")

    # Instantiate the PyTorch Datasets
    # NormalizationDataset (alias for ResidualDataset) expects:
    # df, char_tokenizer, target_tokenizer, max_len
    train_dataset = NormalizationDataset(
        df=train_df,
        char_tokenizer=char_tokenizer,
        target_tokenizer=target_tokenizer,
        max_len=Config.MAX_SEQ_LEN,
    )

    val_dataset = NormalizationDataset(
        df=val_df,
        char_tokenizer=char_tokenizer,
        target_tokenizer=target_tokenizer,
        max_len=Config.MAX_SEQ_LEN,
    )

    # 4. Create DataLoaders
    print(
        f"DataLoader: Creating DataLoaders (batch_size={batch_size}, workers={num_workers})..."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    print("DataLoader: Ready.")
    return train_loader, val_loader, char_tokenizer, target_tokenizer
