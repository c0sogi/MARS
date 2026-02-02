import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from library import config


def tokenize(text):
    """
    Tokenizes text by splitting on whitespace.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokens.
    """
    if not text:
        return []
    return text.split()


def segment_document(tokens):
    """
    Parses tokenized document text to extract candidate paragraphs based on top-level HTML tags.

    In the Simplified NQ dataset, the text contains HTML tags like <P>, <Table>, <Ul>, etc.
    This function identifies spans of text that correspond to these structural elements.

    Args:
        tokens (list): List of document tokens.

    Returns:
        list: A list of dictionaries, each containing:
              - 'start_token': Start index in the token list.
              - 'end_token': End index (exclusive).
              - 'text': The extracted text (joined tokens).
              - 'is_html': Boolean indicating if it started with a tag.
    """
    # Tags that typically denote the start of a candidate long answer in NQ
    # Note: In simplified NQ, tags are tokens.
    TOP_LEVEL_TAGS = {
        "<P>",
        "<Table>",
        "<Tr>",
        "<Ul>",
        "<Ol>",
        "<Dl>",
        "<H1>",
        "<H2>",
        "<H3>",
        "<H4>",
        "<H5>",
        "<H6>",
    }

    candidates = []
    doc_len = len(tokens)

    if doc_len == 0:
        return candidates

    # Simple heuristic:
    # A segment starts at a top-level tag or at index 0.
    # It ends right before the next top-level tag.

    current_start = 0

    for i in range(1, doc_len):
        token = tokens[i]
        if token in TOP_LEVEL_TAGS:
            # End previous segment
            if i > current_start:
                candidates.append(
                    {
                        "start_token": current_start,
                        "end_token": i,
                        "text": " ".join(tokens[current_start:i]),
                    }
                )
            current_start = i

    # Add the last segment
    if current_start < doc_len:
        candidates.append(
            {
                "start_token": current_start,
                "end_token": doc_len,
                "text": " ".join(tokens[current_start:doc_len]),
            }
        )

    return candidates


def build_vocab(load_cached_data=True):
    """
    Builds a vocabulary from the training data.

    Args:
        load_cached_data (bool): If True, tries to load from cache.

    Returns:
        dict: Mapping from token to integer index.
    """
    cache_path = config.VOCAB_CACHE_PATH

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading vocabulary from {cache_path}")
        df = pd.read_parquet(cache_path)
        return dict(zip(df["token"], df["index"]))

    print("Building vocabulary from scratch...")

    # 2. Load Metadata to access training data efficiently
    if not os.path.exists(config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {config.TRAIN_METADATA_PATH}")

    train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)

    # Limit samples for debugging if configured
    if config.DEBUG_SAMPLE_SIZE:
        print(f"Debug mode: Sampling {config.DEBUG_SAMPLE_SIZE} rows for vocab build.")
        train_meta = train_meta.iloc[: config.DEBUG_SAMPLE_SIZE]

    counter = Counter()

    # 3. Iterate through training data
    with open(config.TRAIN_DATA_FILE, "rb") as f:
        for _, row in train_meta.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Add question tokens
                q_text = record.get("question_text", "")
                q_tokens = tokenize(q_text)
                counter.update(q_tokens)

                # Add document tokens (sample first 1000 to save time/memory if doc is huge)
                doc_text = record.get("document_text", "")
                doc_tokens = tokenize(doc_text)
                counter.update(doc_tokens[:1000])

            except json.JSONDecodeError:
                continue

    # 4. Filter and Create Mapping
    # Special tokens
    vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    next_idx = 4

    # Most common tokens
    most_common = counter.most_common(config.MAX_VOCAB_SIZE - len(vocab))
    for token, _ in most_common:
        if token not in vocab:
            vocab[token] = next_idx
            next_idx += 1

    # 5. Save to cache
    print(f"Saving vocabulary to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Convert to DataFrame for Parquet storage
    vocab_df = pd.DataFrame(list(vocab.items()), columns=["token", "index"])
    vocab_df.to_parquet(cache_path, index=False)

    return vocab


def load_embeddings(vocab, load_cached_data=True):
    """
    Loads or initializes an embedding matrix.

    Args:
        vocab (dict): Vocabulary mapping token to index.
        load_cached_data (bool): If True, tries to load from cache.

    Returns:
        np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim).
    """
    cache_path = config.EMBEDDING_MATRIX_CACHE_PATH
    vocab_size = len(vocab)
    embed_dim = config.EMBEDDING_DIM

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading embedding matrix from {cache_path}")
        matrix = np.load(cache_path)
        if matrix.shape == (vocab_size, embed_dim):
            return matrix
        else:
            print(
                f"Cached matrix shape {matrix.shape} mismatch with vocab {vocab_size}. Rebuilding."
            )

    print("Initializing embedding matrix...")

    # 2. Initialize Random Embeddings
    # In a real scenario, we would parse a GloVe file here.
    # Since no GloVe file path is provided in config or input, we use random initialization.
    # We use a normal distribution scaled by 1/sqrt(dim) for stability.
    scale = 1.0 / np.sqrt(embed_dim)
    embedding_matrix = np.random.normal(
        loc=0.0, scale=scale, size=(vocab_size, embed_dim)
    ).astype(np.float32)

    # Zero out PAD token
    if "<PAD>" in vocab:
        pad_idx = vocab["<PAD>"]
        embedding_matrix[pad_idx] = 0.0

    # 3. Save to cache
    print(f"Saving embedding matrix to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embedding_matrix)

    return embedding_matrix


def text_to_indices(text, vocab, max_len=None):
    """
    Converts a text string to a list of vocabulary indices.

    Args:
        text (str): Input text.
        vocab (dict): Vocabulary mapping.
        max_len (int, optional): Maximum length to truncate/pad.

    Returns:
        list: List of integers.
    """
    tokens = tokenize(text)
    unk_idx = vocab.get("<UNK>", 1)
    indices = [vocab.get(t, unk_idx) for t in tokens]

    if max_len is not None:
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            # Pad with 0 (<PAD>)
            indices += [0] * (max_len - len(indices))

    return indices
