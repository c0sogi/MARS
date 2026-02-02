import os
import re
import numpy as np
import torch
from typing import List, Union, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer

from library.config import Config
from library.utils import load_cache_npy, save_cache_npy


def clean_text(text: Union[str, float]) -> str:
    """
    Cleans and normalizes text data.

    Performs the following operations:
    1. Handles NaN/None by returning an empty string.
    2. Normalizes whitespace (replaces newlines and multiple spaces with a single space).
    3. Converts to lowercase.
    4. Strips leading/trailing whitespace.

    Note: The dataset already provides 'request_text_edit_aware' which strips
    explicit 'EDIT:' sections. This function focuses on normalization for
    consistent tokenization and embedding.

    Args:
        text (Union[str, float]): Input text.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text) or text is None:
        return ""

    # Ensure string type
    text = str(text)

    # Replace newlines and tabs with spaces
    text = text.replace("\n", " ").replace("\t", " ")

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # Lowercase and strip
    text = text.lower().strip()

    return text


def generate_embeddings(
    texts: Union[List[str], pd.Series, np.ndarray],
    cache_filename: str,
    load_cached_data: bool = True,
) -> np.ndarray:
    """
    Generates dense vector embeddings for a list of texts using SentenceTransformer.
    Implements caching to avoid re-computation.

    Args:
        texts (Union[List[str], pd.Series, np.ndarray]): Sequence of text strings to embed.
        cache_filename (str): Name of the file to store/load the numpy array (e.g., 'X_train_semantic.npy').
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        np.ndarray: A numpy array of shape (n_samples, embedding_dim).
    """
    # 1. Try to load from cache
    if load_cached_data:
        cached_data = load_cache_npy(cache_filename)
        if cached_data is not None:
            print(f"Loaded embeddings from cache: {cache_filename}")
            return cached_data

    print(
        f"Generating embeddings for {len(texts)} samples (Cache miss or force reload)..."
    )

    # 2. Prepare data
    if isinstance(texts, pd.Series):
        texts_list = texts.tolist()
    elif isinstance(texts, np.ndarray):
        texts_list = texts.tolist()
    else:
        texts_list = list(texts)

    # Ensure all elements are strings (handle potential NaNs in raw data if not cleaned upstream)
    texts_list = [str(t) if not pd.isna(t) else "" for t in texts_list]

    # 3. Load Model
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer model: {Config.SBERT_MODEL_NAME} on {device}")

    model = SentenceTransformer(Config.SBERT_MODEL_NAME, device=device)

    # Set to eval mode just in case (though SentenceTransformer handles this)
    model.eval()

    # 4. Generate Embeddings
    # encode returns a numpy array by default when convert_to_tensor=False
    embeddings = model.encode(
        texts_list,
        batch_size=Config.SBERT_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity stability if needed later
    )

    # 5. Save to cache
    print(f"Saving embeddings to cache: {cache_filename}")
    save_cache_npy(embeddings, cache_filename)

    return embeddings
