import os
import numpy as np
import torch
from library.config import Config
from library.model import (
    get_dataloaders,
    process_sequence_features,
    get_structure_pairs,
    RNADataset,
)

# =========================================================================
# Data Interface
# =========================================================================


def get_loaders(load_cached_data=True):
    """
    Retrieves the PyTorch DataLoaders for Train, Validation, and Test sets.

    This function leverages the caching mechanism implemented in library.model.load_data:
    1. If load_cached_data is True, it attempts to load pre-processed .npy files
       from ./working/idea_5/data_cache/.
    2. If files are missing or load_cached_data is False, it processes the metadata
       CSVs from scratch, generates features, saves to cache, and returns the loaders.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    return get_dataloaders(load_cached_data=load_cached_data)


def process_inputs(df):
    """
    Generates the spatially augmented input tensor for a given dataframe.

    Wraps library.model.process_sequence_features to construct:
    - One-Hot encodings (Sequence, Structure, Loop Type)
    - Partner Base Identity features
    - Relative Pair Distance features

    Note: Absolute Positional Encodings are injected dynamically within the
    HybridNet model's forward pass to ensure correct tensor dimensions.

    Args:
        df (pd.DataFrame): Dataframe containing 'sequence', 'structure', etc.

    Returns:
        np.ndarray: Feature tensor of shape (N, Channels, Seq_Len)
    """
    return process_sequence_features(df)


def get_couples(structure):
    """
    Parses a dot-bracket secondary structure string to identify base pairs.

    Args:
        structure (str): Dot-bracket string (e.g., "((..))").

    Returns:
        dict: Mapping where keys and values are indices of paired bases.
    """
    return get_structure_pairs(structure)


def get_positional_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional encodings.

    This utility is provided for completeness regarding the data specification.
    In the training pipeline, the HybridNet model uses its own internal
    PositionalEncoding module (torch.nn.Module) to generate these features
    on the fly on the correct device.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the encoding.

    Returns:
        np.ndarray: Positional encoding of shape (d_model, seq_len).
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    # Transpose to match the (Channels, Seq_Len) format used in this pipeline
    return pe.T.astype(np.float32)
