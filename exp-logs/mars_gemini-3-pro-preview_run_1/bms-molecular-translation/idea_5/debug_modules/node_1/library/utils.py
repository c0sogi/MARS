import re
import json
import numpy as np
import nltk
import os
from library.config import Config


class Tokenizer:
    """
    Character-level tokenizer for InChI strings.
    Handles vocabulary building, encoding, decoding, and special tokens.
    """

    def __init__(self, config: Config):
        self.config = config
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

        # Initialize with special tokens
        self.special_tokens = [
            config.PAD_TOKEN,
            config.SOS_TOKEN,
            config.EOS_TOKEN,
            config.UNK_TOKEN,
        ]

        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.char2idx:
            self.char2idx[token] = self.vocab_size
            self.idx2char[self.vocab_size] = token
            self.vocab_size += 1

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        for char in sorted_chars:
            self._add_token(char)

    def text_to_sequence(self, text, add_special_tokens=True):
        """
        Converts text to a list of indices.
        """
        sequence = []
        if add_special_tokens:
            sequence.append(self.char2idx[self.config.SOS_TOKEN])

        for char in text:
            sequence.append(
                self.char2idx.get(char, self.char2idx[self.config.UNK_TOKEN])
            )

        if add_special_tokens:
            sequence.append(self.char2idx[self.config.EOS_TOKEN])

        return sequence

    def sequence_to_text(self, sequence, remove_special_tokens=True):
        """
        Converts a list of indices back to text.
        Stops at EOS token if present.
        """
        result = []
        for idx in sequence:
            # Handle tensor or int input
            if hasattr(idx, "item"):
                idx = idx.item()

            char = self.idx2char.get(idx, self.config.UNK_TOKEN)

            if remove_special_tokens:
                if char == self.config.SOS_TOKEN:
                    continue
                if char == self.config.EOS_TOKEN:
                    break
                if char == self.config.PAD_TOKEN:
                    continue

            result.append(char)

        return "".join(result)

    def save(self, path):
        """Saves vocabulary to a JSON file."""
        data = {
            "char2idx": self.char2idx,
            "idx2char": {k: v for k, v in self.idx2char.items()},  # keys as ints/strs
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, path):
        """Loads vocabulary from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        # JSON keys are strings, convert back to int
        self.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        self.vocab_size = len(self.char2idx)


class AttributeNormalizer:
    """
    Handles Z-score normalization for regression targets (atom counts + sequence length).
    """

    def __init__(self, config: Config):
        self.config = config
        self.mean = None
        self.std = None
        self.epsilon = 1e-8

    def fit(self, attributes: np.ndarray):
        """
        Computes mean and std from the attribute matrix.
        Shape of attributes: (N, num_attributes)
        """
        self.mean = np.mean(attributes, axis=0)
        self.std = np.std(attributes, axis=0)
        # Avoid division by zero for constant columns
        self.std[self.std < self.epsilon] = 1.0

    def transform(self, attributes: np.ndarray) -> np.ndarray:
        """
        Applies Z-score normalization: (x - mean) / std
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before transform.")
        return (attributes - self.mean) / self.std

    def inverse_transform(self, normalized_attributes: np.ndarray) -> np.ndarray:
        """
        Reverses normalization: x * std + mean
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before inverse_transform.")
        return (normalized_attributes * self.std) + self.mean

    def save(self, path):
        """Saves mean and std to a numpy file."""
        if self.mean is None or self.std is None:
            return  # Nothing to save
        np.save(path, {"mean": self.mean, "std": self.std})

    def load(self, path):
        """Loads mean and std from a numpy file."""
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True).item()
        self.mean = data["mean"]
        self.std = data["std"]
        return True


def parse_inchi_attributes(inchi_str: str, tracked_atoms: list) -> np.ndarray:
    """
    Parses an InChI string to extract atom counts and sequence length.

    Args:
        inchi_str: The InChI string (e.g., "InChI=1S/C2H5OH/c1-2-3/h3H,2H2,1H3")
        tracked_atoms: List of atom symbols to track (e.g., ['C', 'H', 'O', ...])

    Returns:
        np.ndarray: Vector of shape (len(tracked_atoms) + 1,).
                    Last element is the sequence length.
    """
    # Initialize counts
    counts = {atom: 0.0 for atom in tracked_atoms}

    # Extract Formula Layer
    # Standard format: InChI=1S/<formula>/...
    parts = inchi_str.split("/")
    if len(parts) > 1:
        formula = parts[1]

        # Regex to find Element and Count pairs (e.g., C13, H10, Cl4)
        # Matches an uppercase letter optionally followed by a lowercase letter,
        # followed by optional digits.
        matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula)

        for element, count_str in matches:
            if element in counts:
                # If count is empty string, it implies 1
                count = float(count_str) if count_str else 1.0
                counts[element] += count

    # Create attribute vector
    attr_vector = [counts[atom] for atom in tracked_atoms]

    # Append Sequence Length
    attr_vector.append(float(len(inchi_str)))

    return np.array(attr_vector, dtype=np.float32)


def compute_levenshtein(predictions: list, references: list) -> float:
    """
    Computes the mean Levenshtein distance between predictions and references.

    Args:
        predictions: List of predicted strings.
        references: List of ground truth strings.

    Returns:
        float: Mean Levenshtein distance.
    """
    if not predictions or not references:
        return 0.0

    total_distance = 0
    count = 0

    for pred, ref in zip(predictions, references):
        # NLTK's edit_distance is equivalent to Levenshtein distance
        dist = nltk.edit_distance(pred, ref)
        total_distance += dist
        count += 1

    return total_distance / count if count > 0 else 0.0
