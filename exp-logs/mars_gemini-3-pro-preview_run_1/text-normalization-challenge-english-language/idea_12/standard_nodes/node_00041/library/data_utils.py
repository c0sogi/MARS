import os
import json
import pandas as pd
from collections import Counter
from library.config import Config


class Vocabulary:
    """
    Generic Vocabulary class for mapping tokens to indices and vice versa.
    """

    def __init__(self, specials=None):
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

        # Initialize with special tokens
        for i, token in enumerate(self.specials):
            self.stoi[token] = i
            self.itos[i] = token

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        """
        Returns the index of the token. Returns <UNK> index if not found and <UNK> exists.
        """
        if token in self.stoi:
            return self.stoi[token]

        if "<UNK>" in self.stoi:
            return self.stoi["<UNK>"]

        raise KeyError(
            f"Token '{token}' not found in vocabulary and no <UNK> token defined."
        )

    def lookup_token(self, idx):
        """
        Returns the token for a given index.
        """
        if idx in self.itos:
            return self.itos[idx]
        raise KeyError(f"Index {idx} not found in vocabulary.")

    def add_token(self, token):
        """
        Adds a token to the vocabulary if it doesn't exist.
        """
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

    def save(self, path):
        """
        Saves the vocabulary to a JSON file.
        """
        data = {
            "stoi": self.stoi,
            "itos": {
                int(k): v for k, v in self.itos.items()
            },  # Ensure keys are ints for JSON
            "specials": self.specials,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        """
        Loads the vocabulary from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(specials=data.get("specials"))
        vocab.stoi = data["stoi"]
        # Convert keys back to integers
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        return vocab


def build_vocabularies(load_cached_data=True):
    """
    Builds or loads the Word, Character, and Class vocabularies.

    Args:
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        tuple: (vocab_words, vocab_chars, vocab_classes)
    """
    # Paths
    word_path = Config.VOCAB_WORDS_PATH
    char_path = Config.VOCAB_CHARS_PATH
    class_path = Config.VOCAB_CLASSES_PATH

    # Check if all exist
    all_exist = (
        os.path.exists(word_path)
        and os.path.exists(char_path)
        and os.path.exists(class_path)
    )

    if load_cached_data and all_exist:
        print("Loading vocabularies from cache...")
        vocab_words = Vocabulary.load(word_path)
        vocab_chars = Vocabulary.load(char_path)
        vocab_classes = Vocabulary.load(class_path)
        return vocab_words, vocab_chars, vocab_classes

    print("Building vocabularies from training data...")

    # Load training data
    # We use the training metadata.
    # keep_default_na=False is critical for text data to preserve "null", "nan" as strings.
    df = pd.read_csv(Config.TRAIN_DATA, dtype=str, keep_default_na=False)

    # 1. Build Word Vocabulary
    # ------------------------
    print("Building Word Vocabulary...")
    word_counter = Counter(df["before"].tolist())

    # Specials for words: Padding, Unknown
    vocab_words = Vocabulary(specials=["<PAD>", "<UNK>"])

    # Filter by frequency and max size
    # most_common returns a list of (elem, count)
    for word, count in word_counter.most_common(Config.MAX_VOCAB_SIZE):
        if count >= Config.MIN_FREQ:
            vocab_words.add_token(word)

    # 2. Build Character Vocabulary
    # -----------------------------
    print("Building Character Vocabulary...")
    # We need characters from both 'before' (input) and 'after' (target for seq2seq)
    unique_chars = set()

    # Use a set comprehension for speed on unique chars
    # We sample if dataset is massive, but for 7M tokens, iterating unique words is faster
    unique_before = df["before"].unique()
    unique_after = df["after"].unique()

    for text in unique_before:
        unique_chars.update(str(text))
    for text in unique_after:
        unique_chars.update(str(text))

    # Specials for chars: Pad, Unk, Start of Seq, End of Seq
    vocab_chars = Vocabulary(specials=["<PAD>", "<UNK>", "<SOS>", "<EOS>"])

    # Sort for determinism
    for char in sorted(list(unique_chars)):
        # Limit char vocab if necessary, though usually small enough
        if len(vocab_chars) < Config.MAX_CHAR_VOCAB:
            vocab_chars.add_token(char)

    # 3. Build Class Vocabulary
    # -------------------------
    print("Building Class Vocabulary...")
    unique_classes = sorted(df["class"].unique().tolist())

    # Specials: Just PAD usually enough for classes if using padding in loss
    vocab_classes = Vocabulary(specials=["<PAD>"])

    for cls_name in unique_classes:
        vocab_classes.add_token(cls_name)

    # Save all
    print(f"Saving vocabularies to {Config.WORKING_DIR}...")
    vocab_words.save(word_path)
    vocab_chars.save(char_path)
    vocab_classes.save(class_path)

    print(
        f"Vocab Sizes - Words: {len(vocab_words)}, Chars: {len(vocab_chars)}, Classes: {len(vocab_classes)}"
    )

    return vocab_words, vocab_chars, vocab_classes


class KnowledgeBase:
    """
    A deterministic lookup table mapping (raw_token, class) -> normalized_text.
    """

    def __init__(self):
        self.lookup_table = {}

    def build(self, load_cached_data=True):
        """
        Builds the knowledge base from training data or loads from cache.

        Args:
            load_cached_data (bool): Whether to try loading from cache.
        """
        cache_path = Config.KNOWLEDGE_BASE_PATH

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading Knowledge Base from {cache_path}...")
            try:
                df_kb = pd.read_parquet(cache_path)
                # Convert DataFrame back to dictionary for O(1) lookup
                # Keys are tuples (before, class)
                self.lookup_table = dict(
                    zip(zip(df_kb["before"], df_kb["class"]), df_kb["after"])
                )
                print(f"Knowledge Base loaded with {len(self.lookup_table)} entries.")
                return
            except Exception as e:
                print(f"Failed to load KB cache: {e}. Rebuilding...")

        # 2. Build from scratch
        print("Building Knowledge Base from training data...")

        # Load data
        df = pd.read_csv(
            Config.TRAIN_DATA,
            usecols=["before", "class", "after"],
            dtype=str,
            keep_default_na=False,
        )

        # We want the most frequent mapping for each (before, class) pair.
        # While usually 1:1, there might be noise.
        # Group by keys and take the mode (most frequent 'after')
        # Using a specialized aggregation is faster than generic apply

        # Count occurrences of each triplet
        counts = (
            df.groupby(["before", "class", "after"]).size().reset_index(name="count")
        )

        # Sort by count descending so the first occurrence is the most frequent
        counts = counts.sort_values(
            ["before", "class", "count"], ascending=[True, True, False]
        )

        # Drop duplicates keeping the first (most frequent)
        kb_df = counts.drop_duplicates(subset=["before", "class"], keep="first")

        # Select only necessary columns
        kb_df = kb_df[["before", "class", "after"]]

        # Create dictionary
        self.lookup_table = dict(
            zip(zip(kb_df["before"], kb_df["class"]), kb_df["after"])
        )

        # 3. Save to cache
        print(f"Saving Knowledge Base to {cache_path}...")
        kb_df.to_parquet(cache_path, index=False)

        print(f"Knowledge Base built with {len(self.lookup_table)} entries.")

    def query(self, token, class_label):
        """
        Retrieve the normalized text for a given token and class.

        Args:
            token (str): The raw text token.
            class_label (str): The predicted class.

        Returns:
            str or None: The normalized text if found, else None.
        """
        return self.lookup_table.get((token, class_label))
