import os
import pandas as pd
from collections import Counter
from library.config import Config


class Vocabulary:
    def __init__(self, name, specials=None):
        """
        Args:
            name (str): A name for the vocabulary (e.g., 'tokens', 'chars').
            specials (list): List of special tokens to include at the beginning.
        """
        self.name = name
        self.specials = specials if specials else []
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)

    def build_from_corpus(self, iterator, max_size=None, min_freq=1):
        """
        Builds the vocabulary from an iterator of tokens/strings.

        Args:
            iterator: Iterable of strings.
            max_size (int): Maximum size of the vocabulary (excluding specials).
            min_freq (int): Minimum frequency to include a token.
        """
        counter = Counter(iterator)

        # Sort by frequency (descending) then alphabetically
        # This ensures deterministic ordering
        sorted_by_freq = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # Start with specials
        self.stoi = {tok: i for i, tok in enumerate(self.specials)}
        self.itos = {i: tok for i, tok in enumerate(self.specials)}
        idx = len(self.specials)

        for token, freq in sorted_by_freq:
            if freq < min_freq:
                break

            if (
                max_size is not None
                and (len(self.stoi) - len(self.specials)) >= max_size
            ):
                break

            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1

    def numericalize(self, tokens):
        """
        Converts a list of tokens to indices. Handles UNK if present in specials.
        """
        ids = []
        unk_idx = self.stoi.get(Config.UNK_TOKEN)

        for token in tokens:
            if token in self.stoi:
                ids.append(self.stoi[token])
            else:
                if unk_idx is not None:
                    ids.append(unk_idx)
                else:
                    # If no UNK token, we might ignore or raise error.
                    # For this task, we assume UNK exists if needed, or we map to something safe.
                    # If strictly no UNK defined, we skip.
                    pass
        return ids

    def denumericalize(self, indices):
        """
        Converts a list of indices to tokens.
        """
        return [self.itos.get(idx, Config.UNK_TOKEN) for idx in indices]

    def save(self, path):
        """
        Saves the vocabulary to a Parquet file.
        """
        data = [{"token": token, "index": idx} for token, idx in self.stoi.items()]
        df = pd.DataFrame(data)
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Saved {self.name} vocabulary to {path} ({len(self)} items)")

    def load(self, path):
        """
        Loads the vocabulary from a Parquet file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        df = pd.read_parquet(path)
        self.stoi = {row["token"]: row["index"] for _, row in df.iterrows()}
        self.itos = {row["index"]: row["token"] for _, row in df.iterrows()}
        print(f"Loaded {self.name} vocabulary from {path} ({len(self)} items)")

    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos


def build_vocabularies(load_cached=True):
    """
    Orchestrates the creation or loading of Token, Char, and Class vocabularies.

    Args:
        load_cached (bool): If True, attempts to load from disk first.

    Returns:
        tuple: (vocab_tokens, vocab_chars, vocab_classes)
    """

    # Initialize Vocabulary objects with appropriate specials
    vocab_tokens = Vocabulary(
        name="tokens", specials=[Config.PAD_TOKEN, Config.UNK_TOKEN]
    )

    vocab_chars = Vocabulary(
        name="chars",
        specials=[
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
        ],
    )

    # Classes usually don't need UNK/SOS/EOS, but PAD might be useful for batching if needed.
    # We'll map classes purely.
    vocab_classes = Vocabulary(name="classes", specials=[])

    # Check paths
    paths = {
        "tokens": Config.VOCAB_TOKENS_PATH,
        "chars": Config.VOCAB_CHARS_PATH,
        "classes": Config.VOCAB_CLASSES_PATH,
    }

    all_exist = all(os.path.exists(p) for p in paths.values())

    if load_cached and all_exist:
        print("Loading vocabularies from cache...")
        vocab_tokens.load(paths["tokens"])
        vocab_chars.load(paths["chars"])
        vocab_classes.load(paths["classes"])
    else:
        print("Building vocabularies from training corpus...")

        # Load training data
        # We use keep_default_na=False to ensure "NaN" or "null" strings are treated as text
        df_train = pd.read_csv(Config.TRAIN_DATA_PATH, dtype=str, keep_default_na=False)

        # 1. Build Token Vocabulary
        # We use the 'before' column for input embeddings
        print("Building Token Vocabulary...")
        tokens = df_train["before"].tolist()
        vocab_tokens.build_from_corpus(
            tokens, max_size=Config.MAX_TOKEN_VOCAB_SIZE, min_freq=Config.MIN_TOKEN_FREQ
        )
        vocab_tokens.save(paths["tokens"])

        # 2. Build Character Vocabulary
        # We need characters from both 'before' (input) and 'after' (target for seq2seq)
        print("Building Character Vocabulary...")
        # Concatenate all text to extract unique characters
        all_text = "".join(df_train["before"].tolist()) + "".join(
            df_train["after"].tolist()
        )
        vocab_chars.build_from_corpus(
            all_text, max_size=None, min_freq=1  # Keep all characters
        )
        vocab_chars.save(paths["chars"])

        # 3. Build Class Vocabulary
        print("Building Class Vocabulary...")
        classes = df_train["class"].tolist()
        vocab_classes.build_from_corpus(classes, max_size=None, min_freq=1)
        vocab_classes.save(paths["classes"])

    return vocab_tokens, vocab_chars, vocab_classes
