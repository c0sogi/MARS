import os
import json
import pandas as pd
from collections import Counter
from library.config import Config


class WordVocabulary:
    """
    Manages the target word-level vocabulary for the dual-head model.
    Handles building from corpus, saving/loading, and token-ID conversion.
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initializes the vocabulary with special tokens."""
        self.token2id = {self.pad_token: 0, self.unk_token: 1}
        self.id2token = {0: self.pad_token, 1: self.unk_token}

    def __len__(self):
        return len(self.token2id)

    def build_from_corpus(
        self,
        corpus_path,
        vocab_size=Config.TARGET_VOCAB_SIZE,
        save_path=Config.TARGET_VOCAB_PATH,
        load_cached=True,
    ):
        """
        Builds the vocabulary from the training corpus.

        Args:
            corpus_path (str): Path to the training parquet file.
            vocab_size (int): Maximum size of the vocabulary (excluding special tokens).
            save_path (str): Path to save/load the vocabulary JSON.
            load_cached (bool): Whether to attempt loading from cache first.
        """
        # 1. Check Cache
        if load_cached and os.path.exists(save_path):
            print(f"Loading vocabulary from cache: {save_path}")
            self.load(save_path)
            return

        print(f"Building vocabulary from corpus: {corpus_path}")

        # 2. Load Data
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        # We only need the sentence column.
        df = pd.read_parquet(corpus_path, columns=["sentence"])

        # 3. Count Frequencies
        # We assume space-separated tokens as per the dataset description/analysis
        word_counts = Counter()

        print("Counting word frequencies...")
        total = len(df)

        # Iterate through sentences to count words
        # Using simple split() which handles whitespace
        for i, sentence in enumerate(df["sentence"]):
            if sentence:
                word_counts.update(sentence.split())

            if i % 2000000 == 0 and i > 0:
                print(f"Processed {i}/{total} sentences...")

        # 4. Select Top K
        print(f"Selecting top {vocab_size} frequent words...")
        most_common = word_counts.most_common(vocab_size)

        # 5. Construct Mappings
        self._init_special_tokens()
        current_id = len(self.token2id)

        for word, count in most_common:
            if word not in self.token2id:
                self.token2id[word] = current_id
                self.id2token[current_id] = word
                current_id += 1

        print(f"Vocabulary built. Total size: {len(self.token2id)}")

        # 6. Save to Cache
        self.save(save_path)

    def save(self, path):
        """Saves the vocabulary mappings to a JSON file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "token2id": self.token2id,
            "id2token": {
                str(k): v for k, v in self.id2token.items()
            },  # keys must be strings for json
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """Loads the vocabulary mappings from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token2id = data["token2id"]
        # Convert keys back to integers for id2token
        self.id2token = {int(k): v for k, v in data["id2token"].items()}
        print(f"Vocabulary loaded. Size: {len(self.token2id)}")

    def token_to_id(self, token):
        """
        Converts a token to its integer ID.
        Returns UNK_TOKEN ID if token is not found.
        """
        return self.token2id.get(token, self.token2id[self.unk_token])

    def id_to_token(self, idx):
        """
        Converts an integer ID to its corresponding token.
        Returns UNK_TOKEN if ID is not found.
        """
        return self.id2token.get(idx, self.unk_token)

    def get_vocab_size(self):
        """Returns the total size of the vocabulary."""
        return len(self.token2id)
