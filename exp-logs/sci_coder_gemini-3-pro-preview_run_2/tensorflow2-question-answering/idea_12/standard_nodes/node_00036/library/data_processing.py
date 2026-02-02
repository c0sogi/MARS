import os
import json
import numpy as np
from collections import Counter
from library.config import Config


class Tokenizer:
    """
    Simple whitespace tokenizer.
    """

    def tokenize(self, text):
        """
        Splits text into tokens by whitespace.

        Args:
            text (str): Input text.

        Returns:
            list: List of string tokens.
        """
        if not text:
            return []
        return text.split()


class Vocabulary:
    """
    Manages mapping between tokens and integer indices.
    """

    def __init__(self):
        self.stoi = {}  # String to Index
        self.itos = []  # Index to String
        self.unk_index = -1
        self.pad_index = -1

    def build(
        self,
        token_counts,
        max_size,
        unk_token=Config.UNK_TOKEN,
        pad_token=Config.PAD_TOKEN,
    ):
        """
        Builds vocabulary from token counts.

        Args:
            token_counts (Counter): Frequency count of tokens.
            max_size (int): Maximum vocabulary size.
            unk_token (str): Token for unknown words.
            pad_token (str): Token for padding.
        """
        # Start with special tokens
        self.itos = [pad_token, unk_token]

        # Add most frequent tokens up to max_size
        # Subtract 2 for the special tokens already added
        for token, _ in token_counts.most_common(max_size - 2):
            self.itos.append(token)

        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_index = self.stoi[pad_token]
        self.unk_index = self.stoi[unk_token]

    def save(self, path):
        """
        Saves the vocabulary list to a .npy file.
        """
        # Save as a numpy array of strings
        np.save(path, np.array(self.itos))

    def load(self, path):
        """
        Loads the vocabulary list from a .npy file.
        """
        # Load numpy array and convert to list
        self.itos = np.load(path, allow_pickle=True).tolist()
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        # Re-assign special indices
        if Config.PAD_TOKEN in self.stoi:
            self.pad_index = self.stoi[Config.PAD_TOKEN]
        if Config.UNK_TOKEN in self.stoi:
            self.unk_index = self.stoi[Config.UNK_TOKEN]

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        """
        Converts a list of tokens to a list of indices.
        """
        return [self.stoi.get(t, self.unk_index) for t in tokens]

    def lookup_tokens(self, indices):
        """
        Converts a list of indices to a list of tokens.
        """
        return [
            self.itos[i] if 0 <= i < len(self.itos) else Config.UNK_TOKEN
            for i in indices
        ]


def build_vocab(load_cached_data=True):
    """
    Constructs or loads the vocabulary.

    Args:
        load_cached_data (bool): If True, attempts to load from cache.

    Returns:
        Vocabulary: The initialized vocabulary object.
    """
    vocab = Vocabulary()

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(Config.VOCAB_FILE), exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(Config.VOCAB_FILE):
        print(f"Loading vocabulary from {Config.VOCAB_FILE}...")
        try:
            vocab.load(Config.VOCAB_FILE)
            print(f"Vocabulary loaded. Size: {len(vocab)}")
            return vocab
        except Exception as e:
            print(f"Failed to load vocabulary: {e}. Rebuilding from scratch.")

    # 2. Build from scratch
    print("Building vocabulary from training corpus...")
    tokenizer = Tokenizer()
    counter = Counter()

    # Stream the training file to avoid memory issues
    try:
        with open(Config.TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)

                # Tokenize Question
                q_tokens = tokenizer.tokenize(entry.get("question_text", ""))
                counter.update(q_tokens)

                # Tokenize Document
                # Note: Document text can be very long, processing it all might be slow.
                # For this baseline, we include document tokens in vocab.
                doc_tokens = tokenizer.tokenize(entry.get("document_text", ""))
                counter.update(doc_tokens)

    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found at {Config.TRAIN_FILE}")

    vocab.build(counter, Config.VOCAB_SIZE)

    # 3. Save to cache
    print(f"Saving vocabulary to {Config.VOCAB_FILE}...")
    vocab.save(Config.VOCAB_FILE)
    print(f"Vocabulary built and saved. Size: {len(vocab)}")

    return vocab


def pad_sequence(sequence, max_len, pad_value=0):
    """
    Pads or truncates a sequence of integers to a fixed length.

    Args:
        sequence (list): List of integers.
        max_len (int): Target length.
        pad_value (int): Value to use for padding.

    Returns:
        numpy.ndarray: Padded sequence of shape (max_len,).
    """
    seq_len = len(sequence)

    if seq_len >= max_len:
        # Truncate
        return np.array(sequence[:max_len], dtype=np.int64)
    else:
        # Pad
        padded = np.full(max_len, pad_value, dtype=np.int64)
        padded[:seq_len] = sequence
        return padded


def text_to_indices(text, tokenizer, vocab, max_len):
    """
    Helper function to tokenize, map to indices, and pad a text string.

    Args:
        text (str): Input text string.
        tokenizer (Tokenizer): Tokenizer instance.
        vocab (Vocabulary): Vocabulary instance.
        max_len (int): Target sequence length.

    Returns:
        numpy.ndarray: Processed index array.
    """
    tokens = tokenizer.tokenize(text)
    indices = vocab.lookup_indices(tokens)
    padded = pad_sequence(indices, max_len, pad_value=vocab.pad_index)
    return padded
