import os
import json
import re
import pandas as pd
import numpy as np
from collections import Counter
from library.configuration import Config


def tokenize(text):
    """
    Splits text on whitespace.

    Args:
        text (str): The input string.

    Returns:
        list: A list of tokens.
    """
    if not text:
        return []
    return text.split()


def is_html_tag(token):
    """
    Checks if a token is an HTML tag.

    Args:
        token (str): The token to check.

    Returns:
        bool: True if the token appears to be an HTML tag, False otherwise.
    """
    # NQ simplified tags usually look like <P>, </P>, <Table>, etc.
    return token.startswith("<") and token.endswith(">")


def strip_html_tags(tokens):
    """
    Removes HTML tags from a list of tokens and returns the clean tokens
    along with a mapping from clean indices to original indices.

    Args:
        tokens (list): List of string tokens.

    Returns:
        tuple: (clean_tokens, clean_to_original_indices)
            - clean_tokens (list): List of tokens with tags removed.
            - clean_to_original_indices (list): List where the i-th element is the
              index of the i-th clean token in the original list.
    """
    clean_tokens = []
    clean_to_original_indices = []

    for idx, token in enumerate(tokens):
        if not is_html_tag(token):
            clean_tokens.append(token)
            clean_to_original_indices.append(idx)

    return clean_tokens, clean_to_original_indices


def map_clean_to_raw_span(start_clean, end_clean, mapping):
    """
    Maps a span (start, end) from clean token indices back to raw token indices.

    Args:
        start_clean (int): Start index in clean tokens.
        end_clean (int): End index in clean tokens (exclusive).
        mapping (list): The clean_to_original_indices list.

    Returns:
        tuple: (start_raw, end_raw) indices in the original token list.
               Returns (-1, -1) if indices are invalid.
    """
    if start_clean < 0 or end_clean > len(mapping) or start_clean >= end_clean:
        return -1, -1

    # Map start
    start_raw = mapping[start_clean]

    # Map end
    # The end index in Python slicing is exclusive.
    # If end_clean is within bounds of the mapping, we map it directly.
    # If end_clean is exactly len(mapping), it means the span goes to the very end of the clean text.
    # In that case, we can approximate the raw end as the last mapped index + 1.
    if end_clean < len(mapping):
        end_raw = mapping[end_clean]
    else:
        end_raw = mapping[-1] + 1

    return start_raw, end_raw


def extract_structural_features(candidate_tokens):
    """
    Analyzes tokens to extract structural features based on HTML tags.

    Args:
        candidate_tokens (list): List of tokens for a candidate long answer.

    Returns:
        dict: A dictionary of boolean features.
    """
    features = {
        "is_paragraph": False,
        "is_table": False,
        "is_list": False,
        "is_heading": False,
        "is_other": False,
    }

    if not candidate_tokens:
        features["is_other"] = True
        return features

    # Check the first few tokens for structural indicators
    # We look at the first token primarily, but sometimes tags are nested.
    first_token = candidate_tokens[0]

    if "<P>" in first_token:
        features["is_paragraph"] = True
    elif (
        "<Table>" in first_token
        or "<Tr>" in first_token
        or "<Td>" in first_token
        or "<Th>" in first_token
    ):
        features["is_table"] = True
    elif "<Ul>" in first_token or "<Ol>" in first_token or "<Li>" in first_token:
        features["is_list"] = True
    elif re.match(r"<H\d>", first_token):
        features["is_heading"] = True
    else:
        features["is_other"] = True

    return features


def build_vocab(
    metadata_path=Config.TRAIN_METADATA_PATH,
    data_file=Config.TRAIN_DATA_FILE,
    cache_path=Config.VOCAB_CACHE_PATH,
    vocab_size=Config.VOCAB_SIZE,
    sample_rate=0.1,
    load_cached_data=True,
):
    """
    Builds a vocabulary from the training data with caching.

    Args:
        metadata_path (str): Path to the training metadata CSV.
        data_file (str): Path to the raw JSONL training data.
        cache_path (str): Path to save/load the vocabulary parquet file.
        vocab_size (int): Maximum size of the vocabulary.
        sample_rate (float): Fraction of data to sample for building vocab.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: A dictionary mapping tokens to integer IDs.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading vocabulary from {cache_path}...")
        try:
            df_vocab = pd.read_parquet(cache_path)
            # Convert dataframe back to dict
            vocab = {row["token"]: row["id"] for _, row in df_vocab.iterrows()}
            return vocab
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding vocabulary...")

    # 2. Build from scratch
    print("Building vocabulary from scratch...")

    # Ensure working directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Load metadata to access file offsets
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Sample the data to save time
    if sample_rate < 1.0:
        metadata = metadata.sample(frac=sample_rate, random_state=Config.SEED)

    token_counter = Counter()

    print(f"Processing {len(metadata)} samples for vocabulary...")

    with open(data_file, "rb") as f:
        for _, row in metadata.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                data = json.loads(line.decode("utf-8"))

                # Add question tokens
                q_text = data.get("question_text", "")
                q_tokens = tokenize(q_text)
                token_counter.update(q_tokens)

                # Add document tokens (sample first 500 to capture common words without processing full text)
                doc_text = data.get("document_text", "")
                doc_tokens = tokenize(doc_text)
                # Filter HTML tags from vocab to keep it clean for the Reader
                clean_doc_tokens, _ = strip_html_tags(doc_tokens)
                token_counter.update(clean_doc_tokens[:500])

            except json.JSONDecodeError:
                continue

    # 3. Create Vocabulary
    # Start with special tokens
    special_tokens = [
        Config.PAD_TOKEN,
        Config.SOS_TOKEN,
        Config.EOS_TOKEN,
        Config.UNK_TOKEN,
    ]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    current_idx = len(special_tokens)

    # Add most common tokens up to vocab_size
    # Subtract special tokens count from limit
    limit = vocab_size - len(special_tokens)
    most_common = token_counter.most_common(limit)

    for token, _ in most_common:
        if token not in vocab:
            vocab[token] = current_idx
            current_idx += 1

    print(f"Vocabulary built with {len(vocab)} tokens.")

    # 4. Save to cache
    # Save as DataFrame for Parquet compatibility
    vocab_list = [{"token": k, "id": v} for k, v in vocab.items()]
    df_vocab = pd.DataFrame(vocab_list)
    df_vocab.to_parquet(cache_path, index=False)
    print(f"Vocabulary saved to {cache_path}")

    return vocab


def text_to_indices(text, vocab, max_len=None):
    """
    Converts a string of text to a list of vocabulary indices.

    Args:
        text (str): Input text.
        vocab (dict): Vocabulary dictionary.
        max_len (int, optional): Max length to truncate/pad.

    Returns:
        list: List of integer indices.
    """
    tokens = tokenize(text)
    # Strip tags if present, assuming vocab is built on clean text
    tokens, _ = strip_html_tags(tokens)

    indices = [vocab.get(t, vocab[Config.UNK_TOKEN]) for t in tokens]

    if max_len is not None:
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            # Pad with PAD_TOKEN
            pad_idx = vocab[Config.PAD_TOKEN]
            indices += [pad_idx] * (max_len - len(indices))

    return indices
