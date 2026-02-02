import pandas as pd
from library.config import Config
from library.sparse_engine import SparseRanker
from library.dense_engine import DenseEngine
from library.data_factory import load_data_factory


def train_sparse_model(
    df_train: pd.DataFrame, df_val: pd.DataFrame = None
) -> SparseRanker:
    """
    Orchestrates the training of the Sparse Stream (TF-IDF + Ridge Regression).

    Args:
        df_train (pd.DataFrame): Training data containing 'source' and 'rank'.
        df_val (pd.DataFrame, optional): Validation data for evaluation.

    Returns:
        SparseRanker: The trained sparse ranker instance.
    """
    print("\n" + "=" * 40)
    print("Starting Sparse Stream Training")
    print("=" * 40)

    # Initialize the SparseRanker (TF-IDF + Ridge)
    ranker = SparseRanker()

    # Fit the model (Vectorization + Regression)
    ranker.fit(df_train, df_val)

    # Save the vectorizer and model artifacts
    ranker.save()

    print("Sparse Stream training completed.")
    return ranker


def train_dense_model(
    df_train: pd.DataFrame, df_val: pd.DataFrame = None
) -> DenseEngine:
    """
    Orchestrates the training of the Dense Stream (Transformer).

    Args:
        df_train (pd.DataFrame): Training data containing 'source' and 'rank'.
        df_val (pd.DataFrame, optional): Validation data for evaluation.

    Returns:
        DenseEngine: The trained dense engine instance.
    """
    print("\n" + "=" * 40)
    print("Starting Dense Stream Training")
    print("=" * 40)

    # Initialize the DenseEngine (Transformer + Linear Head)
    engine = DenseEngine()

    # Execute the training loop
    # The engine handles AMP, Optimizer, Scheduler, Logging, and Checkpointing internally.
    engine.fit(df_train, df_val, patience=1)

    print("Dense Stream training completed.")
    return engine


def run_training(load_cached_data: bool = True):
    """
    Main driver function to load data and execute training for both streams.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache if available.
    """
    # 1. Load Data
    # load_data_factory handles metadata reading and caching of processed content
    print("Loading and processing data...")
    df_train, df_val, _, _ = load_data_factory(load_cached_data=load_cached_data)

    # 2. Train Sparse Stream
    train_sparse_model(df_train, df_val)

    # 3. Train Dense Stream
    train_dense_model(df_train, df_val)
