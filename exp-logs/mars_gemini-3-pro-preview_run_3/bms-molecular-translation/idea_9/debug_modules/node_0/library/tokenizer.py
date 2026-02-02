import os
import json
import pandas as pd
import torch


class InchiTokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary construction, encoding, and decoding with special tokens.
    """

    def __init__(
        self, metadata_path="./metadata/train_metadata.csv", load_cached_data=True
    ):
        """
        Initialize the tokenizer.

        Args:
            metadata_path (str): Path to the training metadata CSV file.
            load_cached_data (bool): Whether to try loading the vocabulary from cache.
        """
        self.special_tokens = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.stoi = {}
        self.itos = {}

        # Ensure cache directory exists
        self.cache_dir = "./working/idea_9/"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "vocab.json")

        self._load_or_build_vocab(metadata_path, load_cached_data)

    def _load_or_build_vocab(self, metadata_path, load_cached_data):
        """
        Loads vocabulary from cache or builds it from the dataset.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading vocabulary from {self.cache_path}")
            try:
                with open(self.cache_path, "r") as f:
                    vocab_data = json.load(f)
                self.stoi = vocab_data["stoi"]
                self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
                return
            except Exception as e:
                print(f"Failed to load vocab cache: {e}. Rebuilding...")

        print(f"Building vocabulary from {metadata_path}...")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Extract all unique characters from the InChI column
        # Using a set for efficiency
        unique_chars = set()

        # Iterate in chunks to handle memory efficiently if needed,
        # though InChI strings are short enough that full column iteration is fine.
        # We assume 'InChI' column exists based on metadata generation script.
        if "InChI" not in df.columns:
            raise ValueError(f"Column 'InChI' not found in {metadata_path}")

        # Collect all characters
        # We can simply join a sample or iterate. Given the scale, iterating is safer.
        # However, pandas str.cat might be memory intensive.
        # Let's use a counter approach or direct set update.
        text_blob = "".join(df["InChI"].astype(str).unique())
        unique_chars.update(text_blob)

        # Sort characters for determinism
        sorted_chars = sorted(list(unique_chars))

        # Build mappings
        # Start with special tokens
        self.stoi = self.special_tokens.copy()
        self.itos = {v: k for k, v in self.special_tokens.items()}

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.stoi[char] = idx
            self.itos[idx] = char

        # Save to cache
        print(f"Saving vocabulary to {self.cache_path}")
        with open(self.cache_path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, indent=4)

    def encode(self, text, max_length=None, padding=False):
        """
        Converts an InChI string into a list of integers.
        Adds <SOS> at the start and <EOS> at the end.

        Args:
            text (str): The InChI string.
            max_length (int, optional): Max length for padding/truncation.
            padding (bool): If True, pads to max_length.

        Returns:
            torch.Tensor: Tensor of token indices.
        """
        tokens = [self.stoi["<SOS>"]]

        for char in text:
            tokens.append(self.stoi.get(char, self.stoi["<UNK>"]))

        tokens.append(self.stoi["<EOS>"])

        if max_length is not None:
            if len(tokens) > max_length:
                # Truncate, but keep EOS if possible?
                # Usually for InChI we want the full string.
                # Standard truncation:
                tokens = tokens[:max_length]
                # Ensure EOS is at the end if truncated?
                # For autoregressive generation, simply truncating is standard data loader behavior.
                tokens[-1] = self.stoi["<EOS>"]

            if padding:
                pad_len = max_length - len(tokens)
                if pad_len > 0:
                    tokens.extend([self.stoi["<PAD>"]] * pad_len)

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, indices):
        """
        Converts a sequence of integers back to an InChI string.
        Stops at <EOS>. Ignores <SOS> and <PAD>.

        Args:
            indices (list or torch.Tensor): Sequence of token indices.

        Returns:
            str: Decoded InChI string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        chars = []
        for idx in indices:
            if idx == self.stoi["<EOS>"]:
                break
            if idx == self.stoi["<SOS>"]:
                continue
            if idx == self.stoi["<PAD>"]:
                continue

            chars.append(self.itos.get(idx, ""))

        return "".join(chars)

    def __len__(self):
        return len(self.stoi)

    @property
    def pad_token_id(self):
        return self.stoi["<PAD>"]

    @property
    def sos_token_id(self):
        return self.stoi["<SOS>"]

    @property
    def eos_token_id(self):
        return self.stoi["<EOS>"]

    @property
    def unk_token_id(self):
        return self.stoi["<UNK>"]
