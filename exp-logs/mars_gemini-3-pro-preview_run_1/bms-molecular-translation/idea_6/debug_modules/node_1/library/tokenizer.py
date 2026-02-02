import json
import os
import torch
from library.config import Config


class InChITokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Relies on the vocabulary defined in Config.
    """

    def __init__(self):
        self.vocab = Config.VOCAB
        self.stoi = {char: idx for idx, char in enumerate(self.vocab)}
        self.itos = {idx: char for idx, char in enumerate(self.vocab)}

    def text_to_sequence(self, text, max_len=None, padding=False):
        """
        Converts a string to a sequence of integers.

        Args:
            text (str): The InChI string.
            max_len (int, optional): Maximum length for padding/truncation.
            padding (bool): Whether to pad the sequence to max_len.

        Returns:
            list[int]: The sequence of indices.
        """
        sequence = [self.stoi[Config.SOS_TOKEN]]

        for char in text:
            if char in self.stoi:
                sequence.append(self.stoi[char])
            else:
                sequence.append(self.stoi[Config.UNK_TOKEN])

        sequence.append(self.stoi[Config.EOS_TOKEN])

        if padding and max_len is not None:
            if len(sequence) < max_len:
                sequence += [self.stoi[Config.PAD_TOKEN]] * (max_len - len(sequence))
            else:
                # Truncate if strictly necessary, preserving EOS if possible
                # In practice, MAX_LEN is chosen to cover the dataset
                sequence = sequence[: max_len - 1] + [self.stoi[Config.EOS_TOKEN]]

        return sequence

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of integers back to a string.

        Args:
            sequence (list[int] or torch.Tensor): The sequence of indices.

        Returns:
            str: The decoded InChI string.
        """
        text = []
        for idx in sequence:
            # Handle tensor input
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            # Stop at EOS
            if idx == self.stoi[Config.EOS_TOKEN]:
                break

            # Skip SOS and PAD
            if idx == self.stoi[Config.SOS_TOKEN] or idx == self.stoi[Config.PAD_TOKEN]:
                continue

            # Convert to char
            if idx in self.itos:
                text.append(self.itos[idx])
            else:
                # Unknown index (should not happen with correct vocab)
                pass

        return "".join(text)

    def save_tokenizer(self, path=None):
        """
        Saves the vocabulary mapping to a JSON file.
        """
        if path is None:
            path = Config.TOKENIZER_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "stoi": self.stoi,
            "itos": {
                str(k): v for k, v in self.itos.items()
            },  # Convert int keys to str for JSON
            "vocab": self.vocab,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Tokenizer saved to {path}")

    def load_tokenizer(self, path=None):
        """
        Loads the vocabulary mapping from a JSON file.
        """
        if path is None:
            path = Config.TOKENIZER_PATH

        if not os.path.exists(path):
            print(f"Tokenizer file not found at {path}. Using Config defaults.")
            return

        with open(path, "r") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        # Convert string keys back to int for itos
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab = data["vocab"]
        print(f"Tokenizer loaded from {path}")


def get_tokenizer(load_cached_data=True):
    """
    Factory function to get a tokenizer instance.
    Implements the required caching logic.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        InChITokenizer: The initialized tokenizer.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    tokenizer = InChITokenizer()

    if load_cached_data and os.path.exists(Config.TOKENIZER_PATH):
        try:
            tokenizer.load_tokenizer()
        except Exception as e:
            print(f"Failed to load tokenizer cache: {e}. Re-saving defaults.")
            tokenizer.save_tokenizer()
    else:
        # If loading is disabled or file doesn't exist, save the current (Config-based) state
        tokenizer.save_tokenizer()

    return tokenizer
