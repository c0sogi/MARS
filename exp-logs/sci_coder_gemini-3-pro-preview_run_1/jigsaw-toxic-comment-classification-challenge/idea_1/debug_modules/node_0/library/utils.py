import os
import random
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(metadata_path: str, raw_path: str) -> pd.DataFrame:
    """
    Loads the dataset by merging metadata (IDs, labels) with the raw text source.

    Args:
        metadata_path: Path to the metadata CSV file (e.g., train.csv, val.csv).
        raw_path: Path to the raw source CSV file (e.g., input/train.csv).

    Returns:
        pd.DataFrame: Merged DataFrame containing IDs, labels, and comment text.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    # Load metadata containing IDs and Labels
    meta_df = pd.read_csv(metadata_path)

    # Load raw data, selecting only ID and Text to avoid label duplication
    # and reduce memory usage.
    try:
        raw_df = pd.read_csv(raw_path, usecols=["id", "comment_text"])
    except ValueError:
        # Fallback if columns are named differently or file is malformed
        # Reading all and filtering is safer but slower
        raw_df = pd.read_csv(raw_path)
        raw_df = raw_df[["id", "comment_text"]]

    # Merge metadata with raw text on 'id'
    # metadata determines the rows (inner join logic effectively via left join on meta)
    df = pd.merge(meta_df, raw_df, on="id", how="left")

    # Handle missing text values (NaNs become empty strings)
    df["comment_text"] = df["comment_text"].fillna("")

    return df


def load_embeddings(
    embedding_path: str, word_index: dict, embedding_dim: int = 300
) -> np.ndarray:
    """
    Creates an embedding matrix for the model.

    Initializes the matrix with random normal values. If a valid embedding_path
    is provided, it loads pre-trained vectors (e.g., GloVe) and updates the matrix.

    Args:
        embedding_path: Path to the pre-trained embedding file (txt format).
        word_index: Dictionary mapping words to integer indices.
        embedding_dim: Dimension of the embedding vectors.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size + 1, embedding_dim).
    """
    nb_words = len(word_index) + 1
    # Initialize with random normal values
    embedding_matrix = np.random.normal(0, 1, (nb_words, embedding_dim)).astype(
        np.float32
    )

    # If no path provided or file doesn't exist, return random initialization
    if not embedding_path or not os.path.exists(embedding_path):
        return embedding_matrix

    print(f"Loading embeddings from {embedding_path}...")
    try:
        with open(embedding_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                values = line.rstrip().split(" ")
                word = values[0]
                if word in word_index:
                    idx = word_index[word]
                    # Ensure vector size matches expected dimension
                    if len(values) - 1 == embedding_dim:
                        vector = np.asarray(values[1:], dtype="float32")
                        embedding_matrix[idx] = vector
    except Exception as e:
        print(f"Warning: Failed to load embeddings. {e}")

    return embedding_matrix
