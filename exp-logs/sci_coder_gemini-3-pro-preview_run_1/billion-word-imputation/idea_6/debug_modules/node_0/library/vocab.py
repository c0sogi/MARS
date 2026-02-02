import os
import numpy as np
import pandas as pd
from collections import Counter
from library.config import Config


class Vocabulary:
    """
    Vocabulary class to handle tokenization and numericalization for the
    Bifurcated Interleaved Transformer. Manages special tokens and
    frequency-based filtering.
    """

    def __init__(self):
        self.stoi = {}
        self.itos = {}

        # Special tokens
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN
        self.gap_token = Config.GAP_TOKEN
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"

        # Order ensures fixed indices for specials: PAD=0, UNK=1, GAP=2, SOS=3, EOS=4
        self.specials = [
            self.pad_token,
            self.unk_token,
            self.gap_token,
            self.sos_token,
            self.eos_token,
        ]

    def build(
        self,
        corpus_path=Config.TRAIN_METADATA,
        max_size=Config.VOCAB_SIZE,
        min_freq=Config.MIN_FREQ,
        load_cached_data=True,
    ):
        """
        Builds the vocabulary from the corpus or loads it from cache.

        Args:
            corpus_path (str): Path to the training metadata CSV.
            max_size (int): Maximum size of the vocabulary.
            min_freq (int): Minimum frequency for a word to be included.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Check Cache
        if load_cached_data and os.path.exists(Config.VOCAB_PATH):
            print(f"Loading vocabulary from cache: {Config.VOCAB_PATH}")
            try:
                self.load(Config.VOCAB_PATH)
                return
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding vocabulary.")

        # 2. Build from Scratch
        print(f"Building vocabulary from {corpus_path}...")

        # Load training data
        # We use pandas to read the CSV efficiently
        try:
            df = pd.read_csv(corpus_path)
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {corpus_path}")
            return

        # Initialize counter
        counter = Counter()

        # Process sentences
        # Assuming space-separated tokens as per dataset description
        if "sentence" in df.columns:
            sentences = df["sentence"].dropna().astype(str).tolist()
            for sentence in sentences:
                counter.update(sentence.split())
        else:
            print("Error: 'sentence' column not found in metadata.")
            return

        # Initialize mappings with special tokens
        self.stoi = {token: i for i, token in enumerate(self.specials)}
        self.itos = {i: token for i, token in enumerate(self.specials)}

        current_idx = len(self.specials)

        # Add most common words
        # most_common() returns a list of (word, count) sorted by count desc
        for word, count in counter.most_common():
            if len(self.stoi) >= max_size:
                break
            if count < min_freq:
                break

            if word not in self.stoi:
                self.stoi[word] = current_idx
                self.itos[current_idx] = word
                current_idx += 1

        print(f"Vocabulary built. Total tokens: {len(self.stoi)}")

        # 3. Save to Cache
        self.save(Config.VOCAB_PATH)

    def save(self, path):
        """
        Saves the vocabulary to a .npy file.
        """
        # We save the list of tokens ordered by index.
        # stoi can be reconstructed from this list.
        vocab_list = [self.itos[i] for i in range(len(self.itos))]
        np.save(path, np.array(vocab_list))
        print(f"Vocabulary saved to {path}")

    def load(self, path):
        """
        Loads the vocabulary from a .npy file.
        """
        vocab_array = np.load(path, allow_pickle=True)
        self.itos = {i: str(token) for i, token in enumerate(vocab_array)}
        self.stoi = {str(token): i for i, token in enumerate(vocab_array)}
        print(f"Vocabulary loaded. Size: {len(self.stoi)}")

    def encode(self, text, add_special_tokens=False):
        """
        Converts a text string or list of tokens into a list of indices.

        Args:
            text (str or list): Input sentence or list of tokens.
            add_special_tokens (bool): If True, wraps with [SOS] and [EOS].

        Returns:
            list[int]: List of token indices.
        """
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text

        indices = []

        if add_special_tokens:
            indices.append(self.stoi[self.sos_token])

        for token in tokens:
            indices.append(self.stoi.get(token, self.stoi[self.unk_token]))

        if add_special_tokens:
            indices.append(self.stoi[self.eos_token])

        return indices

    def decode(self, indices, skip_special_tokens=True):
        """
        Converts a list of indices back into a string.

        Args:
            indices (list[int]): List of token indices.
            skip_special_tokens (bool): If True, removes special tokens from output.

        Returns:
            str: Reconstructed sentence.
        """
        tokens = []
        for idx in indices:
            token = self.itos.get(int(idx), self.unk_token)
            if skip_special_tokens and token in self.specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi[self.unk_token])

    def get_token(self, idx):
        return self.itos.get(idx, self.unk_token)
