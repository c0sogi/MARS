import os
import re
import pandas as pd
from collections import Counter
from typing import List, Optional, Dict

from library.config import CACHE_DIR, MAX_VOCAB_SIZE, MIN_FREQ, PAD_TOKEN, UNK_TOKEN
from library.utils import clean_text


class Vocabulary:
    def __init__(self):
        self.stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def tokenize(self, text: str) -> List[str]:
        """
        Cleans and tokenizes text into a list of words.
        Splits on non-alphanumeric characters.
        """
        # Clean text (handles unicode, quotes, lowercase)
        text = clean_text(text)
        # Find all alphanumeric sequences (letters, numbers, underscores)
        return re.findall(r"\w+", text)

    def fit(
        self, texts: List[str], max_size: int = MAX_VOCAB_SIZE, min_freq: int = MIN_FREQ
    ):
        """
        Builds the vocabulary from a list of text strings.

        Args:
            texts: List of raw text strings.
            max_size: Maximum size of the vocabulary.
            min_freq: Minimum frequency for a token to be included.
        """
        print(f"Fitting vocabulary on {len(texts)} texts...")
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Filter by minimum frequency
        valid_tokens = [token for token, count in counter.items() if count >= min_freq]

        # Sort by frequency (descending) and then alphabetically for determinism
        valid_tokens.sort(key=lambda t: (-counter[t], t))

        # Truncate to max size, accounting for existing special tokens
        # We currently have 2 special tokens (PAD, UNK)
        num_special = len(self.stoi)
        num_to_keep = max_size - num_special

        valid_tokens = valid_tokens[:num_to_keep]

        # Add to vocabulary
        for token in valid_tokens:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

        print(f"Vocabulary fitted. Total size: {len(self.stoi)}")

    def transform(self, text: str) -> List[int]:
        """
        Converts a text string into a list of token indices.
        """
        tokens = self.tokenize(text)
        unk_idx = self.stoi[UNK_TOKEN]
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def save(self, directory: str):
        """
        Saves the vocabulary mapping to a parquet file.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "vocab.parquet")

        # Convert dictionary to DataFrame for Parquet storage
        data = [{"token": token, "index": idx} for token, idx in self.stoi.items()]
        df = pd.DataFrame(data)

        df.to_parquet(path, index=False)
        print(f"Vocabulary saved to {path}")

    def load(self, directory: str):
        """
        Loads the vocabulary mapping from a parquet file.
        """
        path = os.path.join(directory, "vocab.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        df = pd.read_parquet(path)

        self.stoi = {}
        self.itos = {}

        for _, row in df.iterrows():
            token = row["token"]
            idx = row["index"]
            self.stoi[token] = idx
            self.itos[idx] = token

        print(f"Vocabulary loaded from {path}. Total size: {len(self.stoi)}")


def build_vocabulary(
    texts: Optional[List[str]] = None, load_cached_data: bool = True
) -> Vocabulary:
    """
    Factory function to create or load a Vocabulary instance.

    Args:
        texts: List of strings to fit on if cache is not used/found.
        load_cached_data: Whether to attempt loading from cache.

    Returns:
        A ready-to-use Vocabulary instance.
    """
    vocab = Vocabulary()

    # Try to load from cache
    if load_cached_data:
        try:
            vocab.load(CACHE_DIR)
            return vocab
        except (FileNotFoundError, OSError):
            print("Cached vocabulary not found or unreadable. Rebuilding...")

    # If we reach here, we need to rebuild. Ensure texts are provided.
    if texts is None:
        raise ValueError("Vocabulary cache not found and no texts provided to fit.")

    vocab.fit(texts)
    vocab.save(CACHE_DIR)

    return vocab
