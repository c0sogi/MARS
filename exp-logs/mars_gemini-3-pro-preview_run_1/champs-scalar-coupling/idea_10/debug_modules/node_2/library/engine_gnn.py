import os
import pandas as pd
from library import config
from library import model_gnn


def train_gnn(epochs=None, batch_size=None, load_cached_data=True):
    """
    Trains the InteractionMPNN model for representation learning.

    Args:
        epochs (int, optional): Number of training epochs. Overrides config if provided.
        batch_size (int, optional): Batch size. Overrides config if provided.
        load_cached_data (bool): Whether to load pre-processed graph data from cache.

    Returns:
        The trained PyTorch model.
    """
    # Update configuration if overrides are provided
    if epochs is not None:
        config.GNN_PARAMS["epochs"] = int(epochs)
    if batch_size is not None:
        config.GNN_PARAMS["batch_size"] = int(batch_size)

    # Ensure working directories exist
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Execute training via the library module
    # The library module handles:
    # - Data loading via graph_dataset
    # - Model initialization
    # - Training loop with Early Stopping
    # - Metric printing
    # - Model checkpointing to config.MODEL_DIR/gnn_best.pt
    model = model_gnn.train_gnn(load_cached_data=load_cached_data)

    return model


def extract_embeddings(split, load_cached_data=True):
    """
    Generates and extracts learned interaction embeddings for a given data split.

    Args:
        split (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): Whether to attempt loading embeddings from parquet cache.
                                 If False or cache missing, runs inference and saves cache.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and the embedding columns (gnn_emb_0, ...).
    """
    # Validate split
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}")

    # Execute embedding generation via the library module
    # The library module handles:
    # - Loading the best model from config.MODEL_DIR/gnn_best.pt
    # - Running inference
    # - Caching the result to config.CACHE_DIR/gnn_embeddings_{split}.parquet
    df_embeddings = model_gnn.generate_embeddings(
        split=split, load_cached_data=load_cached_data
    )

    return df_embeddings
