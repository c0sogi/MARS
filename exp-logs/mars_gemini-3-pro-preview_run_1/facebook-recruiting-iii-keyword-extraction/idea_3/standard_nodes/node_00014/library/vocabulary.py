import os
import json
import pandas as pd
from collections import Counter
from library.config import Config


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.tag2idx = {}
        self.idx2tag = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.pad_idx = 0
        self.unk_idx = 1

        # Initialize with special tokens
        self.word2idx[self.pad_token] = self.pad_idx
        self.word2idx[self.unk_token] = self.unk_idx
        self.idx2word[self.pad_idx] = self.pad_token
        self.idx2word[self.unk_idx] = self.unk_token

    def build(self, train_ids_set):
        """
        Builds vocabulary and tag maps from the training data.
        Reads train.csv in chunks, filters by train_ids_set.

        Args:
            train_ids_set (set): A set of integer IDs belonging to the training split.
        """
        print("Building vocabulary from scratch...")

        word_counter = Counter()
        tag_counter = Counter()

        # Ensure input file exists
        if not os.path.exists(Config.TRAIN_CSV):
            raise FileNotFoundError(f"{Config.TRAIN_CSV} not found.")

        # Read train.csv in chunks to handle large file size
        chunk_size = 100000

        # We read Id, Title, Body, and Tags
        # Title and Body are for vocabulary, Tags are for labels
        for chunk in pd.read_csv(
            Config.TRAIN_CSV,
            chunksize=chunk_size,
            usecols=["Id", "Title", "Body", "Tags"],
            dtype={
                "Id": "int64",
                "Title": "object",
                "Body": "object",
                "Tags": "object",
            },
        ):

            # Filter for training set
            chunk = chunk[chunk["Id"].isin(train_ids_set)]

            if chunk.empty:
                continue

            # Process Tags
            # Tags are space delimited strings
            chunk_tags = chunk["Tags"].fillna("").astype(str).tolist()
            for tag_str in chunk_tags:
                tags = tag_str.lower().split()
                tag_counter.update(tags)

            # Process Text (Title + Body)
            # Simple whitespace tokenization as per instructions
            titles = chunk["Title"].fillna("").astype(str).str.lower()
            bodies = chunk["Body"].fillna("").astype(str).str.lower()

            # Iterate through the chunk to update word counts
            # Combining title and body for tokenization
            for t, b in zip(titles, bodies):
                tokens = t.split() + b.split()
                word_counter.update(tokens)

        # Finalize Vocabulary
        print(f"Total unique words found: {len(word_counter)}")

        # Sort by frequency (descending) then alphabetically
        most_common = word_counter.most_common()

        # Filter by MIN_FREQ
        valid_words = []
        for word, count in most_common:
            if count >= Config.MIN_FREQ:
                valid_words.append(word)

        # Truncate to VOCAB_SIZE (accounting for PAD and UNK)
        max_vocab_words = Config.VOCAB_SIZE - 2
        if len(valid_words) > max_vocab_words:
            valid_words = valid_words[:max_vocab_words]

        # Add to mapping
        for i, word in enumerate(valid_words):
            idx = i + 2  # 0 and 1 are taken
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Final Vocabulary Size: {len(self.word2idx)}")

        # Finalize Tags
        # We keep all unique tags found in the training set
        # Sort alphabetically for deterministic mapping
        unique_tags = sorted(tag_counter.keys())

        for i, tag in enumerate(unique_tags):
            self.tag2idx[tag] = i
            self.idx2tag[i] = tag

        print(f"Total Tags: {len(self.tag2idx)}")

    def save(self):
        """Saves the vocabulary and tag mappings to JSON files."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(Config.TOKENIZER_PATH), exist_ok=True)

        # Save Tokenizer (Vocabulary)
        with open(Config.TOKENIZER_PATH, "w") as f:
            json.dump(self.word2idx, f)

        # Save Tag Map
        with open(Config.TAG_MAP_PATH, "w") as f:
            json.dump(self.tag2idx, f)

        print(f"Saved tokenizer to {Config.TOKENIZER_PATH}")
        print(f"Saved tag map to {Config.TAG_MAP_PATH}")

    def load(self):
        """Loads the vocabulary and tag mappings from JSON files."""
        if not os.path.exists(Config.TOKENIZER_PATH) or not os.path.exists(
            Config.TAG_MAP_PATH
        ):
            raise FileNotFoundError("Vocab or Tag files not found.")

        with open(Config.TOKENIZER_PATH, "r") as f:
            self.word2idx = json.load(f)

        # Reconstruct idx2word (JSON keys are always strings, need to convert to int)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}

        with open(Config.TAG_MAP_PATH, "r") as f:
            self.tag2idx = json.load(f)

        # Reconstruct idx2tag
        self.idx2tag = {int(v): k for k, v in self.tag2idx.items()}

        print(f"Loaded vocabulary: {len(self.word2idx)} tokens")
        print(f"Loaded tag map: {len(self.tag2idx)} tags")

    def text_to_indices(self, text):
        """
        Converts a string to a list of integer indices.

        Args:
            text (str): Input text.

        Returns:
            list[int]: List of token indices.
        """
        if not isinstance(text, str):
            return []

        tokens = text.lower().split()
        indices = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        return indices

    def tags_to_indices(self, tags_str):
        """
        Converts a space-delimited tag string to a list of tag indices.

        Args:
            tags_str (str): Space-delimited tags.

        Returns:
            list[int]: List of tag indices.
        """
        if not isinstance(tags_str, str):
            return []

        tags = tags_str.lower().split()
        indices = [self.tag2idx[tag] for tag in tags if tag in self.tag2idx]
        return indices

    def indices_to_tags(self, indices):
        """
        Converts a list of indices to a space-delimited tag string.

        Args:
            indices (list[int]): Tag indices.

        Returns:
            str: Space-delimited tags.
        """
        tags = [self.idx2tag.get(int(idx), "") for idx in indices]
        # Filter out empty strings if any index was not found
        tags = [t for t in tags if t]
        return " ".join(tags)

    def get_vocab_size(self):
        return len(self.word2idx)

    def get_num_tags(self):
        return len(self.tag2idx)


def get_or_build_vocabulary(load_cached_data=True):
    """
    Factory function to get the Vocabulary object.
    Implements caching logic: checks for existing JSON files.
    If not found or forced to rebuild, reads metadata and train.csv to build from scratch.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        Vocabulary: The initialized vocabulary object.
    """
    vocab = Vocabulary()

    # Check if files exist
    files_exist = os.path.exists(Config.TOKENIZER_PATH) and os.path.exists(
        Config.TAG_MAP_PATH
    )

    if load_cached_data and files_exist:
        print("Loading vocabulary from cache...")
        vocab.load()
    else:
        # Build from scratch
        # 1. Load Train Metadata to identify training samples
        print(f"Loading metadata from {Config.TRAIN_META}...")
        if not os.path.exists(Config.TRAIN_META):
            raise FileNotFoundError(
                f"Metadata file {Config.TRAIN_META} not found. Please run metadata generation first."
            )

        df_meta = pd.read_csv(Config.TRAIN_META, usecols=["Id"])
        train_ids_set = set(df_meta["Id"].values)

        # 2. Build
        vocab.build(train_ids_set)

        # 3. Save
        vocab.save()

    return vocab
