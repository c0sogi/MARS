import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from library.config import Config


def tokenize_text(text):
    """
    Splits a string into a list of tokens based on whitespace.

    Args:
        text (str): Input text string.

    Returns:
        list: List of string tokens.
    """
    if not text:
        return []
    return text.split()


def split_document_by_html(document_tokens):
    """
    Splits the document tokens into candidate blocks based on HTML tags defined in Config.

    Args:
        document_tokens (list): List of tokens from the document.

    Returns:
        list: A list of dictionaries, each representing a candidate block.
              Format: {'tokens': list, 'start_token_idx': int, 'end_token_idx': int}
    """
    split_tags = set(Config.SPLIT_TAGS)
    candidates = []

    current_tokens = []
    current_start_idx = 0

    for i, token in enumerate(document_tokens):
        # If token is a split tag and we have accumulated content, save current block
        # We start a new block with the tag itself
        if token in split_tags and current_tokens:
            candidates.append(
                {
                    "tokens": current_tokens,
                    "start_token_idx": current_start_idx,
                    "end_token_idx": i,  # Exclusive end index
                }
            )
            current_tokens = []
            current_start_idx = i

        current_tokens.append(token)

    # Append the last block
    if current_tokens:
        candidates.append(
            {
                "tokens": current_tokens,
                "start_token_idx": current_start_idx,
                "end_token_idx": len(document_tokens),
            }
        )

    return candidates


class TextEncoder:
    """
    Encodes text tokens into indices and decodes indices back to tokens.
    """

    def __init__(self, vocab_list):
        self.idx2word = vocab_list
        self.word2idx = {w: i for i, w in enumerate(vocab_list)}

        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # Default to 0 and 1 if not found, though build_vocab ensures they are at 0 and 1
        self.pad_idx = self.word2idx.get(self.pad_token, 0)
        self.unk_idx = self.word2idx.get(self.unk_token, 1)

    def encode(self, tokens):
        """Converts a list of tokens to a list of indices."""
        return [self.word2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices):
        """Converts a list of indices to a list of tokens."""
        return [self.idx2word[i] for i in indices if 0 <= i < len(self.idx2word)]

    def __len__(self):
        return len(self.idx2word)


def build_vocab(metadata_df, load_cached_data=True):
    """
    Builds a vocabulary from the training data or loads it from cache.

    Args:
        metadata_df (pd.DataFrame): Metadata containing file paths and byte offsets.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        TextEncoder: An instantiated TextEncoder object.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, "vocab.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading vocabulary from cache: {cache_path}")
        try:
            vocab_df = pd.read_parquet(cache_path)
            vocab_list = vocab_df["token"].tolist()
            print(f"Vocabulary loaded. Size: {len(vocab_list)}")
            return TextEncoder(vocab_list)
        except Exception as e:
            print(f"Failed to load cache: {e}. Rebuilding from scratch...")

    # 2. Compute from scratch
    print("Building vocabulary from training data...")

    # Determine sample size for vocabulary building
    if Config.MAX_TRAIN_SAMPLES is not None:
        sample_size = min(len(metadata_df), Config.MAX_TRAIN_SAMPLES)
        df_subset = metadata_df.iloc[:sample_size]
        print(f"Using subset of {sample_size} samples for vocabulary building.")
    else:
        df_subset = metadata_df
        print(f"Using all {len(df_subset)} samples for vocabulary building.")

    token_counter = Counter()

    # Group by file path to optimize file opening
    grouped = df_subset.groupby("file_path")

    for file_name, group in grouped:
        file_path = os.path.join(Config.INPUT_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        with open(file_path, "rb") as f:
            for _, row in group.iterrows():
                offset = row["byte_offset"]
                f.seek(offset)
                line = f.readline()
                if not line:
                    continue

                try:
                    data = json.loads(line.decode("utf-8"))

                    # Add question tokens
                    q_text = data.get("question_text", "")
                    q_tokens = tokenize_text(q_text)
                    token_counter.update(q_tokens)

                    # Add document tokens
                    # Note: Documents can be very long. We process the full text here
                    # to ensure good vocabulary coverage, but one could sample if needed.
                    doc_text = data.get("document_text", "")
                    doc_tokens = tokenize_text(doc_text)
                    token_counter.update(doc_tokens)

                except json.JSONDecodeError:
                    continue

    print(f"Total unique tokens found: {len(token_counter)}")

    # Get most common tokens
    # most_common returns elements sorted by count descending
    most_common = token_counter.most_common()

    # Filter by minimum frequency
    filtered_tokens = [t for t, c in most_common if c >= Config.MIN_FREQ]

    # Truncate to MAX_VOCAB_SIZE, reserving 2 spots for special tokens
    # We take the top frequent tokens
    vocab_list = filtered_tokens[: Config.MAX_VOCAB_SIZE - 2]

    # Add Special Tokens at the beginning
    final_vocab = [Config.PAD_TOKEN, Config.UNK_TOKEN] + vocab_list

    print(f"Final vocabulary size: {len(final_vocab)}")

    # 3. Save to cache
    print(f"Saving vocabulary to {cache_path}")
    vocab_df = pd.DataFrame({"token": final_vocab})
    vocab_df.to_parquet(cache_path, index=False)

    return TextEncoder(final_vocab)
