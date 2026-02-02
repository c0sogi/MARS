import os
import json
import collections
import pandas as pd
from library.config import Config


class WordTokenizer:
    """
    A simple word-level tokenizer that handles vocabulary building,
    encoding, and decoding for the word insertion task.
    """

    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_token_id = 0
        self.unk_token_id = 1

        # Initialize with special tokens
        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
        }
        self.idx2word = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
        }

    def build_vocab(self, sentences, vocab_size):
        """
        Builds vocabulary from a list of sentences.

        Args:
            sentences (list): List of sentence strings.
            vocab_size (int): Maximum size of the vocabulary.
        """
        print(f"Building vocabulary from {len(sentences)} sentences...")
        counter = collections.Counter()

        for sentence in sentences:
            # Simple whitespace tokenization as per dataset format
            tokens = sentence.split()
            counter.update(tokens)

        # Reserve spots for PAD and UNK
        num_special_tokens = 2
        most_common = counter.most_common(vocab_size - num_special_tokens)

        print(f"Found {len(counter)} unique words. Keeping top {len(most_common)}.")

        # Reset vocab to ensure clean state
        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
        }
        self.idx2word = {
            self.pad_token_id: self.pad_token,
            self.unk_token_id: self.unk_token,
        }

        for i, (word, count) in enumerate(most_common):
            idx = i + num_special_tokens
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built. Size: {len(self.word2idx)}")

    def encode(self, text, max_len=None):
        """
        Converts a sentence string to a list of token IDs.

        Args:
            text (str): Input sentence.
            max_len (int, optional): Max sequence length for padding/truncation.

        Returns:
            list: List of integer token IDs.
        """
        tokens = text.split()
        token_ids = [self.word2idx.get(token, self.unk_token_id) for token in tokens]

        if max_len is not None:
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids = token_ids + [self.pad_token_id] * (max_len - len(token_ids))

        return token_ids

    def decode(self, token_ids):
        """
        Converts a list of token IDs back to a string.

        Args:
            token_ids (list): List of integer token IDs.

        Returns:
            str: Reconstructed sentence.
        """
        tokens = []
        for idx in token_ids:
            # Skip padding
            if idx == self.pad_token_id:
                continue

            # Retrieve word, default to UNK if somehow index is missing (shouldn't happen)
            word = self.idx2word.get(idx, self.unk_token)
            tokens.append(word)

        return " ".join(tokens)

    def save(self, path):
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        # Reconstruct idx2word
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        print(f"Tokenizer loaded from {path}. Vocab size: {len(self.word2idx)}")

    def get_vocab_size(self):
        return len(self.word2idx)


def get_tokenizer(load_cached_data=True):
    """
    Factory function to get a tokenizer instance.
    Implements caching logic to avoid rebuilding vocab if it exists.

    Args:
        load_cached_data (bool): If True, attempts to load from disk.

    Returns:
        WordTokenizer: Ready-to-use tokenizer instance.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    tokenizer_path = Config.TOKENIZER_PATH
    tokenizer = WordTokenizer()

    # 1. Try to load cached
    if load_cached_data and os.path.exists(tokenizer_path):
        try:
            tokenizer.load(tokenizer_path)
            return tokenizer
        except Exception as e:
            print(f"Failed to load cached tokenizer: {e}. Rebuilding...")

    # 2. Rebuild if missing or forced
    print("Building tokenizer from training data...")

    # Load training data
    # We use the metadata parquet file as per Config
    df_train = pd.read_parquet(Config.TRAIN_DATA_PATH, columns=["sentence"])

    # Handle debug sampling
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(
            f"DEBUG: Sampling {Config.DEBUG_SAMPLE_SIZE} sentences for vocab building."
        )
        if len(df_train) > Config.DEBUG_SAMPLE_SIZE:
            df_train = df_train.sample(
                n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
            )

    sentences = df_train["sentence"].tolist()

    # Build vocab
    tokenizer.build_vocab(sentences, Config.VOCAB_SIZE)

    # Save to cache
    tokenizer.save(tokenizer_path)

    return tokenizer
