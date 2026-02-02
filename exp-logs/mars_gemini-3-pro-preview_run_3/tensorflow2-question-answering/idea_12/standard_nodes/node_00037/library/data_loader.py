import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from library.config import Config
from library.text_utils import Vocab
from library.ranker_net import RankerDataset, prepare_ranker_data
from library.reader_net import (
    ReaderDataset,
    prepare_reader_data,
    prepare_reader_test_data,
)


# Set seeds for reproducibility
def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(Config.SEED)


class NQRankerDataset(RankerDataset):
    """
    Dataset for the Ranker model.
    Inherits from library.ranker_net.RankerDataset to provide positive/negative pairs.
    """

    def __init__(self, data_df, max_q_len, max_p_len):
        super().__init__(data_df, max_q_len, max_p_len)


class NQReaderDataset(ReaderDataset):
    """
    Dataset for the Reader model.
    Inherits from library.reader_net.ReaderDataset to provide concatenated sequences and span targets.
    """

    def __init__(self, data_df, max_len):
        super().__init__(data_df, max_len)


def load_vocab():
    """
    Loads the vocabulary from the cached artifacts.
    """
    vocab = Vocab()
    if os.path.exists(Config.VOCAB_PATH) and os.path.exists(
        Config.EMBEDDING_MATRIX_PATH
    ):
        vocab.load(Config.VOCAB_PATH, Config.EMBEDDING_MATRIX_PATH)
        return vocab
    else:
        raise FileNotFoundError(
            f"Vocab artifacts not found at {Config.VOCAB_PATH} or {Config.EMBEDDING_MATRIX_PATH}. Please ensure vocab is built."
        )


def get_ranker_loaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    train_sample_size=None,
    val_sample_size=None,
):
    """
    Creates DataLoaders for training and validating the Ranker model.

    Args:
        batch_size (int): Batch size for the dataloaders.
        load_cached_data (bool): Whether to attempt loading processed data from cache.
        train_sample_size (int, optional): Number of training samples to use (for debugging).
        val_sample_size (int, optional): Number of validation samples to use.

    Returns:
        tuple: (train_loader, val_loader)
    """
    vocab = load_vocab()

    # Load Metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    # Apply sampling if requested
    if train_sample_size is not None and len(train_meta) > train_sample_size:
        train_meta = train_meta.sample(n=train_sample_size, random_state=Config.SEED)

    if val_sample_size is not None and len(val_meta) > val_sample_size:
        val_meta = val_meta.sample(n=val_sample_size, random_state=Config.SEED)

    # Prepare DataFrames using library functions (handles caching logic internally)
    # Note: If sampling is used, we might be loading a full cached file if it exists.
    # To strictly enforce sampling when a cache exists, one would need to disable loading cache.
    # Here we respect the load_cached_data flag passed by the user.

    train_df = prepare_ranker_data(
        train_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.RANKER_TRAIN_PATH,
    )

    val_df = prepare_ranker_data(
        val_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.RANKER_VAL_PATH,
    )

    # Create Datasets
    train_dataset = NQRankerDataset(train_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)
    val_dataset = NQRankerDataset(val_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def get_reader_loaders(
    batch_size=Config.BATCH_SIZE,
    load_cached_data=True,
    train_sample_size=None,
    val_sample_size=None,
):
    """
    Creates DataLoaders for training and validating the Reader model.
    """
    vocab = load_vocab()

    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)

    if train_sample_size is not None and len(train_meta) > train_sample_size:
        train_meta = train_meta.sample(n=train_sample_size, random_state=Config.SEED)
    if val_sample_size is not None and len(val_meta) > val_sample_size:
        val_meta = val_meta.sample(n=val_sample_size, random_state=Config.SEED)

    train_df = prepare_reader_data(
        train_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.READER_TRAIN_PATH,
    )

    val_df = prepare_reader_data(
        val_meta,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=load_cached_data,
        cache_path=Config.READER_VAL_PATH,
    )

    max_len = Config.Q_MAX_LEN + Config.P_MAX_LEN
    train_dataset = NQReaderDataset(train_df, max_len)
    val_dataset = NQReaderDataset(val_df, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def get_test_loaders(batch_size=Config.BATCH_SIZE):
    """
    Creates the initial DataLoader for the test set (Ranker input).

    Returns:
        tuple: (ranker_loader, ranker_test_df)
    """
    vocab = load_vocab()
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Prepare Ranker Test Data (Candidates)
    # We typically don't cache test intermediate steps to avoid staleness, or use a temp path.
    # Passing cache_path=None prevents saving to disk, keeping it in memory.
    ranker_test_df = prepare_ranker_data(
        test_meta,
        vocab,
        Config.TEST_RAW_FILE,
        is_train=False,
        load_cached_data=False,
        cache_path=None,
    )

    ranker_dataset = NQRankerDataset(ranker_test_df, Config.Q_MAX_LEN, Config.P_MAX_LEN)
    ranker_loader = DataLoader(
        ranker_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return ranker_loader, ranker_test_df


def get_reader_test_loader(ranker_output_path, batch_size=Config.BATCH_SIZE):
    """
    Creates DataLoader for Reader inference based on the selected candidates from the Ranker.

    Args:
        ranker_output_path (str): Path to the parquet file containing selected candidates.
        batch_size (int): Batch size.

    Returns:
        tuple: (reader_loader, reader_test_df) or (None, None) if empty.
    """
    vocab = load_vocab()

    reader_test_df = prepare_reader_test_data(
        ranker_output_path,
        vocab,
        load_cached_data=False,
        cache_path=Config.READER_TEST_PATH,
    )

    if reader_test_df.empty:
        return None, None

    max_len = Config.Q_MAX_LEN + Config.P_MAX_LEN

    # Add dummy targets for dataset compatibility if not present
    if "start_token" not in reader_test_df.columns:
        reader_test_df["start_token"] = 0
    if "end_token" not in reader_test_df.columns:
        reader_test_df["end_token"] = 0

    reader_dataset = NQReaderDataset(reader_test_df, max_len)
    reader_loader = DataLoader(
        reader_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return reader_loader, reader_test_df
