import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from library.config import Config


class HTMLParser:
    """
    Handles parsing of raw document text into candidate paragraphs based on
    provided candidate annotations.
    """

    def __init__(self):
        pass

    def extract_candidates(self, document_text, candidates_data):
        """
        Extracts candidate paragraphs from the document text.

        Args:
            document_text (str): The raw document text (space-separated tokens).
            candidates_data (list): List of dicts containing 'start_token', 'end_token'.

        Returns:
            list: A list of dictionaries, each containing:
                  - 'text': The string content of the paragraph.
                  - 'start_token': The starting index in the document.
                  - 'end_token': The ending index in the document.
                  - 'top_level': Boolean indicating if it's a top-level HTML tag.
        """
        if not document_text:
            return []

        # The document_text in NQ is pre-tokenized by spaces
        tokens = document_text.split()
        candidates = []

        for cand in candidates_data:
            start = cand.get("start_token", -1)
            end = cand.get("end_token", -1)

            # Validate indices
            if start != -1 and end != -1 and start < end and end <= len(tokens):
                # Extract the span
                span_tokens = tokens[start:end]
                text = " ".join(span_tokens)

                candidates.append(
                    {
                        "text": text,
                        "start_token": start,
                        "end_token": end,
                        "top_level": cand.get("top_level", False),
                    }
                )

        return candidates


class Tokenizer:
    """
    Converts text to integer sequences based on a fixed vocabulary.
    """

    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.vocab_size = 0

        # Initialize special tokens from Config
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # 0 is reserved for padding, 1 for unknown
        self.word_index[self.pad_token] = 0
        self.index_word[0] = self.pad_token
        self.word_index[self.unk_token] = 1
        self.index_word[1] = self.unk_token

        self.next_idx = 2

    def fit_from_vocab_df(self, vocab_df):
        """
        Populates the internal vocabulary maps from a pandas DataFrame.

        Args:
            vocab_df (pd.DataFrame): DataFrame with a 'token' column.
        """
        # Reset to initial state
        self.word_index = {self.pad_token: 0, self.unk_token: 1}
        self.index_word = {0: self.pad_token, 1: self.unk_token}
        self.next_idx = 2

        # Determine how many tokens to load (Config limit - 2 special tokens)
        limit = Config.VOCAB_SIZE - 2

        # Assume vocab_df is already sorted by frequency if that matters,
        # otherwise we just take the top rows.
        if "token" not in vocab_df.columns:
            raise ValueError("vocab_df must contain a 'token' column")

        tokens = vocab_df["token"].astype(str).tolist()[:limit]

        for token in tokens:
            if token not in self.word_index:
                self.word_index[token] = self.next_idx
                self.index_word[self.next_idx] = token
                self.next_idx += 1

        self.vocab_size = len(self.word_index)

    def text_to_sequence(self, text):
        """
        Converts a string to a list of integers.
        """
        if not text:
            return []

        tokens = text.split()
        seq = []
        for t in tokens:
            # Look up token, default to UNK index (1)
            idx = self.word_index.get(t, 1)
            seq.append(idx)
        return seq

    def pad_sequence(self, seq, max_len):
        """
        Pads or truncates a sequence to a fixed length.
        Padding is applied at the end (post-padding).
        """
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            # Pad with 0
            return seq + [0] * (max_len - len(seq))


def build_vocab(load_cached_data=True):
    """
    Constructs the vocabulary from the training corpus or loads it from cache.

    Args:
        load_cached_data (bool): If True, attempts to load the vocabulary from disk.
                                 If False or load fails, rebuilds from scratch.

    Returns:
        Tokenizer: A fitted Tokenizer instance.
    """
    # Ensure working directory exists
    Config.setup_directories()

    cache_path = Config.VOCAB_CACHE_PATH

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading vocabulary from {cache_path}...")
        try:
            vocab_df = pd.read_parquet(cache_path)
            tokenizer = Tokenizer()
            tokenizer.fit_from_vocab_df(vocab_df)
            print(f"Vocabulary loaded successfully. Size: {tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            print(f"Failed to load cached vocabulary: {e}. Rebuilding...")

    # 2. Build from Scratch
    print("Building vocabulary from training corpus...")

    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Training metadata not found at {Config.TRAIN_METADATA_PATH}"
        )

    metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(
            f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows for vocabulary building."
        )
        metadata = metadata.head(Config.DEBUG_SAMPLE_SIZE)

    train_file_path = os.path.join(Config.INPUT_DIR, Config.TRAIN_FILE)

    token_counter = Counter()

    # Iterate through training data using offsets
    with open(train_file_path, "rb") as f:
        for _, row in metadata.iterrows():
            offset = row["byte_offset"]
            f.seek(offset)
            line = f.readline()
            if not line:
                continue

            try:
                entry = json.loads(line.decode("utf-8"))

                # Add Question Tokens
                q_text = entry.get("question_text", "")
                if q_text:
                    token_counter.update(q_text.split())

                # Add Document Tokens
                # Optimization: Documents can be very long. We sample the first 2000 tokens
                # to capture common vocabulary without excessive processing time.
                doc_text = entry.get("document_text", "")
                if doc_text:
                    doc_tokens = doc_text.split()
                    token_counter.update(doc_tokens[:2000])

            except json.JSONDecodeError:
                continue

    # Select top N tokens
    # Config.VOCAB_SIZE includes special tokens, so we take a bit less from the counter
    # to leave room, though fit_from_vocab_df handles the exact slicing.
    most_common = token_counter.most_common(Config.VOCAB_SIZE)

    # Create DataFrame
    vocab_data = [{"token": t, "count": c} for t, c in most_common]
    vocab_df = pd.DataFrame(vocab_data)

    # 3. Save to Cache
    print(f"Saving vocabulary to {cache_path}...")
    vocab_df.to_parquet(cache_path, index=False)

    # Initialize Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_from_vocab_df(vocab_df)
    print(f"Vocabulary build complete. Size: {tokenizer.vocab_size}")

    return tokenizer
