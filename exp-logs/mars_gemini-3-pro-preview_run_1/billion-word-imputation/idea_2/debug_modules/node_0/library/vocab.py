import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("vocab")


class Vocabulary:
    def __init__(self):
        self.stoi = {}
        self.itos = []
        self.special_tokens = [
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.NO_INSERT_TOKEN,
        ]

    def __len__(self):
        return len(self.itos)

    def build_from_corpus(self, csv_path: str, max_vocab_size: int, min_freq: int):
        """
        Builds the vocabulary from the provided CSV file.
        Assumes the CSV has a 'sentence' column.
        """
        logger.info(f"Building vocabulary from {csv_path}...")
        counter = Counter()

        # Read in chunks to handle large files memory-efficiently
        chunk_size = 100_000
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Training metadata not found at {csv_path}")

            reader = pd.read_csv(csv_path, chunksize=chunk_size)

            for i, chunk in enumerate(reader):
                if "sentence" not in chunk.columns:
                    continue

                # Drop NaNs and convert to string
                sentences = chunk["sentence"].dropna().astype(str).tolist()

                # Simple whitespace tokenization as per dataset analysis
                for sentence in sentences:
                    tokens = sentence.split()
                    counter.update(tokens)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {(i + 1) * chunk_size} rows...")

        except Exception as e:
            logger.error(f"Error reading corpus: {e}")
            raise e

        logger.info(f"Total unique tokens found: {len(counter)}")

        # Filter by frequency
        # most_common returns a list of (elem, count) sorted by count
        # We take the top (max_vocab_size - len(special_tokens))
        limit = max_vocab_size - len(self.special_tokens)
        most_common = counter.most_common(limit)

        # Filter by min_freq
        valid_tokens = [token for token, count in most_common if count >= min_freq]

        logger.info(
            f"Tokens kept after filtering (max_size={max_vocab_size}, min_freq={min_freq}): {len(valid_tokens)}"
        )

        # Construct vocab
        self.itos = self.special_tokens + valid_tokens
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        logger.info(f"Final vocabulary size: {len(self.itos)}")

    def save(self, path: str):
        """
        Saves the vocabulary (itos list) to a numpy file.
        """
        logger.info(f"Saving vocabulary to {path}...")
        try:
            np.save(path, np.array(self.itos))
        except Exception as e:
            logger.error(f"Failed to save vocabulary: {e}")
            raise e

    def load(self, path: str):
        """
        Loads the vocabulary from a numpy file.
        """
        logger.info(f"Loading vocabulary from {path}...")
        try:
            # allow_pickle=True is required for loading arrays of strings/objects
            self.itos = np.load(path, allow_pickle=True).tolist()
            self.stoi = {token: i for i, token in enumerate(self.itos)}
            logger.info(f"Loaded vocabulary of size {len(self.itos)}")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            raise e

    def numericalize(self, tokens: list) -> list:
        """
        Converts a list of tokens (strings) to indices.
        """
        unk_idx = self.stoi.get(Config.UNK_TOKEN)
        return [self.stoi.get(token, unk_idx) for token in tokens]

    def denumericalize(self, indices: list) -> list:
        """
        Converts a list of indices to tokens (strings).
        """
        return [
            self.itos[idx] if 0 <= idx < len(self.itos) else Config.UNK_TOKEN
            for idx in indices
        ]

    def get_pad_index(self):
        return self.stoi.get(Config.PAD_TOKEN)

    def get_unk_index(self):
        return self.stoi.get(Config.UNK_TOKEN)

    def get_no_insert_index(self):
        return self.stoi.get(Config.NO_INSERT_TOKEN)


def get_vocabulary(load_cached_data: bool = True) -> Vocabulary:
    """
    Factory function to get the vocabulary object.
    Implements the strict caching logic required.
    """
    vocab = Vocabulary()

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORK_DIR, "vocab.npy")

    # Logic Flow:
    # 1. IF load_cached_data is True: Try to load.
    # 2. IF loading fails OR load_cached_data is False: Build and Save.

    loaded = False
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                vocab.load(cache_path)
                loaded = True
            except Exception as e:
                logger.warning(f"Could not load cached vocabulary: {e}. Rebuilding...")
                loaded = False
        else:
            logger.info("Cached vocabulary not found. Rebuilding...")
            loaded = False

    if not loaded:
        logger.info("Building vocabulary from scratch...")
        vocab.build_from_corpus(
            csv_path=Config.TRAIN_METADATA,
            max_vocab_size=Config.VOCAB_SIZE,
            min_freq=Config.MIN_FREQ,
        )
        vocab.save(cache_path)

    return vocab
