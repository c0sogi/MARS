import os
import json
import pandas as pd


class Tokenizer:
    """
    Handles character-level tokenization for InChI strings.
    Manages vocabulary creation, caching, and encoding/decoding.
    """

    def __init__(
        self,
        metadata_path="./metadata/train_metadata.csv",
        load_cached_data=True,
        cache_dir="./working/idea_6/",
    ):
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>"]
        self.stoi = {}
        self.itos = {}

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        vocab_path = os.path.join(cache_dir, "vocab.json")

        loaded = False
        if load_cached_data and os.path.exists(vocab_path):
            try:
                print(f"Loading vocabulary from {vocab_path}...")
                with open(vocab_path, "r") as f:
                    self.stoi = json.load(f)
                loaded = True
            except Exception as e:
                print(f"Failed to load vocabulary from cache: {e}")

        if not loaded:
            print("Building vocabulary from metadata...")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

            df = pd.read_csv(metadata_path)

            # Collect unique characters from the InChI column
            # Using a set to store unique characters
            chars = set()
            # Iterate to avoid memory issues with joining massive columns
            for text in df["InChI"]:
                chars.update(text)

            # Sort characters to ensure deterministic order
            sorted_chars = sorted(list(chars))

            # Build string-to-index mapping
            # Indices 0, 1, 2 are reserved for special tokens
            self.stoi = {token: i for i, token in enumerate(self.special_tokens)}
            for i, char in enumerate(sorted_chars):
                self.stoi[char] = i + len(self.special_tokens)

            # Save to cache
            print(f"Saving vocabulary to {vocab_path}...")
            with open(vocab_path, "w") as f:
                json.dump(self.stoi, f, indent=4)

        # Build index-to-string mapping
        self.itos = {int(i): token for token, i in self.stoi.items()}

    def text_to_sequence(self, text):
        """
        Converts a string to a sequence of indices with <SOS> and <EOS>.

        Args:
            text (str): Input InChI string.

        Returns:
            list[int]: List of token indices.
        """
        sequence = [self.stoi["<SOS>"]]
        for char in text:
            if char in self.stoi:
                sequence.append(self.stoi[char])
            else:
                # In strict mode we might raise an error, but for robustness we skip unknown chars
                pass
        sequence.append(self.stoi["<EOS>"])
        return sequence

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of indices back to a string.
        Stops at <EOS>. Ignores <PAD> and <SOS>.

        Args:
            sequence (list[int] or torch.Tensor): Sequence of token indices.

        Returns:
            str: Decoded string.
        """
        result = []
        for idx in sequence:
            # Handle tensor or numpy scalar
            if hasattr(idx, "item"):
                idx = idx.item()

            # Ensure idx is int
            idx = int(idx)

            if idx == self.stoi["<EOS>"]:
                break
            if idx == self.stoi["<PAD>"]:
                continue
            if idx == self.stoi["<SOS>"]:
                continue

            if idx in self.itos:
                result.append(self.itos[idx])
        return "".join(result)

    def __len__(self):
        """Returns the size of the vocabulary."""
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
