import os
import json
import pandas as pd
from library.config import Config


class Tokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary building, caching, and special tokens.
    """

    def __init__(self, load_cached_data=True):
        self.stoi = {}  # String to Integer mapping
        self.itos = {}  # Integer to String mapping
        self.vocab_path = Config.VOCAB_PATH

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self.build_vocab(load_cached_data)

    def build_vocab(self, load_cached_data):
        """
        Builds the vocabulary from the training metadata or loads it from cache.
        Strictly follows the caching logic requirement.
        """
        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(self.vocab_path):
            try:
                with open(self.vocab_path, "r") as f:
                    vocab_data = json.load(f)
                self.stoi = vocab_data["stoi"]
                # JSON keys are always strings, convert back to int for itos
                self.itos = {int(k): v for k, v in vocab_data["itos"].items()}
                print(
                    f"Loaded vocabulary from {self.vocab_path}. Size: {len(self.stoi)}"
                )
                return
            except Exception as e:
                print(f"Failed to load cached vocabulary: {e}. Rebuilding...")

        # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
        print("Building vocabulary from training metadata...")

        # Load training metadata
        if not os.path.exists(Config.TRAIN_METADATA):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_METADATA}"
            )

        df = pd.read_csv(Config.TRAIN_METADATA)

        # Extract unique characters from the InChI column
        # We use the unique values of the column first to reduce the size of text to process
        unique_inchis = df["InChI"].unique()
        text_blob = "".join(unique_inchis)
        unique_chars = sorted(list(set(text_blob)))

        # Define special tokens order
        # PAD usually 0 for embedding layers padding_idx=0
        specials = [
            Config.PAD_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
            Config.UNK_TOKEN,
        ]

        # Create mappings
        self.stoi = {token: i for i, token in enumerate(specials)}
        start_idx = len(specials)

        for i, char in enumerate(unique_chars):
            self.stoi[char] = start_idx + i

        self.itos = {i: token for token, i in self.stoi.items()}

        # 3. Save the result to the cache directory
        save_data = {"stoi": self.stoi, "itos": self.itos}
        with open(self.vocab_path, "w") as f:
            json.dump(save_data, f, indent=4)

        print(
            f"Vocabulary built and saved to {self.vocab_path}. Size: {len(self.stoi)}"
        )

    def text_to_sequence(self, text, add_special_tokens=True, max_length=None):
        """
        Converts a text string to a sequence of integers.

        Args:
            text (str): The input InChI string.
            add_special_tokens (bool): Whether to add SOS and EOS tokens.
            max_length (int, optional): If provided, pads or truncates the sequence.

        Returns:
            list[int]: The sequence of token indices.
        """
        sequence = []
        if add_special_tokens:
            sequence.append(self.stoi[Config.SOS_TOKEN])

        for char in text:
            sequence.append(self.stoi.get(char, self.stoi[Config.UNK_TOKEN]))

        if add_special_tokens:
            sequence.append(self.stoi[Config.EOS_TOKEN])

        if max_length is not None:
            # Truncate if necessary
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
                # Note: Truncation might remove EOS, but strictly enforcing length is priority here

            # Pad if necessary
            if len(sequence) < max_length:
                pad_count = max_length - len(sequence)
                sequence.extend([self.stoi[Config.PAD_TOKEN]] * pad_count)

        return sequence

    def sequence_to_text(self, sequence, remove_special_tokens=True):
        """
        Converts a sequence of integers back to a text string.

        Args:
            sequence (list[int] or torch.Tensor): The sequence of token indices.
            remove_special_tokens (bool): Whether to stop at EOS and skip SOS/PAD.

        Returns:
            str: The decoded string.
        """
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()

        result = []
        for idx in sequence:
            idx = int(idx)
            token = self.itos.get(idx, Config.UNK_TOKEN)

            if remove_special_tokens:
                if token == Config.SOS_TOKEN:
                    continue
                if token == Config.PAD_TOKEN:
                    continue
                if token == Config.EOS_TOKEN:
                    break

            result.append(token)

        return "".join(result)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.stoi)
