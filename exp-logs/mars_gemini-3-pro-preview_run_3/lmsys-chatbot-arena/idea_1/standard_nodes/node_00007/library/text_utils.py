import os
import json
import re
import collections
import numpy as np
from library.config import Config


class Tokenizer:
    """
    A simple tokenizer that builds a vocabulary from text and converts text to sequences of integers.
    Supports saving and loading the vocabulary state.
    """

    def __init__(self, config: Config):
        self.vocab_size = config.VOCAB_SIZE
        self.lowercase = config.TOKENIZER_LOWERCASE

        # Special Tokens
        self.pad_token = "<PAD>"
        self.oov_token = "<UNK>"
        self.sep_token = "<SEP>"
        self.pad_index = 0
        self.oov_index = 1
        self.sep_index = 2

        # Mappings
        self.word_index = {}
        self.index_word = {}

        # Initialize mappings with special tokens
        self.word_index[self.pad_token] = self.pad_index
        self.word_index[self.oov_token] = self.oov_index
        self.word_index[self.sep_token] = self.sep_index
        self.index_word[self.pad_index] = self.pad_token
        self.index_word[self.oov_index] = self.oov_token
        self.index_word[self.sep_index] = self.sep_token

    def _tokenize(self, text):
        """Helper to clean and split text."""
        if self.lowercase:
            text = text.lower()
        # Simple regex to keep alphanumeric words
        return re.findall(r"\b\w+\b", text)

    def fit_on_texts(self, texts):
        """
        Builds the vocabulary based on the frequency of words in the given texts.

        Args:
            texts: List of strings.
        """
        word_counts = collections.Counter()

        for text in texts:
            if isinstance(text, str):
                tokens = self._tokenize(text)
                word_counts.update(tokens)

        # We reserve 3 indices (PAD, UNK, SEP), so we pick vocab_size - 3 top words
        # If vocab_size is larger than unique words, we take all of them.
        num_words = self.vocab_size - 3
        most_common = word_counts.most_common(num_words)

        # Start indexing from 3
        current_idx = 3
        for word, _ in most_common:
            self.word_index[word] = current_idx
            self.index_word[current_idx] = word
            current_idx += 1

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts into a list of integer sequences using the fitted vocabulary.

        Args:
            texts: List of strings.

        Returns:
            List of lists of integers.
        """
        sequences = []
        for text in texts:
            if not isinstance(text, str):
                sequences.append([])
                continue

            tokens = self._tokenize(text)
            seq = []
            for token in tokens:
                # Use get() with default OOV_INDEX
                idx = self.word_index.get(token, self.oov_index)
                seq.append(idx)
            sequences.append(seq)
        return sequences

    def save(self, path):
        """
        Saves the tokenizer state (vocabulary and config) to a JSON file.

        Args:
            path: File path to save the JSON.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "word_index": self.word_index,
            "config": {"vocab_size": self.vocab_size, "lowercase": self.lowercase},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """
        Loads the tokenizer state from a JSON file.

        Args:
            path: File path to the JSON file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.word_index = data["word_index"]
            # JSON keys are strings, convert back to int for index_word
            self.index_word = {int(v): k for k, v in self.word_index.items()}

            if "config" in data:
                self.vocab_size = data["config"].get("vocab_size", self.vocab_size)
                self.lowercase = data["config"].get("lowercase", self.lowercase)
            return True
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            return False


def pad_sequences(sequences, maxlen, padding="post", truncating="post", value=0):
    """
    Pads sequences to the same length.

    Args:
        sequences: List of lists of integers.
        maxlen: Int, maximum length of all sequences.
        padding: 'pre' or 'post' (pad either before or after each sequence).
        truncating: 'pre' or 'post' (remove values from sequences larger than maxlen).
        value: Float or String, padding value.

    Returns:
        Numpy array of shape (len(sequences), maxlen).
    """
    num_samples = len(sequences)

    # Initialize output array with padding value
    x = np.full((num_samples, maxlen), value, dtype=np.int32)

    for idx, seq in enumerate(sequences):
        if not seq:
            continue

        # Truncate
        if len(seq) > maxlen:
            if truncating == "pre":
                trunc = seq[-maxlen:]
            else:
                trunc = seq[:maxlen]
        else:
            trunc = seq

        # Pad
        if len(trunc) < maxlen:
            if padding == "pre":
                x[idx, -len(trunc) :] = trunc
            else:
                x[idx, : len(trunc)] = trunc
        else:
            x[idx, :] = trunc

    return x
