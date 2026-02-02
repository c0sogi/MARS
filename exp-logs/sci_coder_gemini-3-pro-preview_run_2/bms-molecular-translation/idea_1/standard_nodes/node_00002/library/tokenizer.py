import torch
import numpy as np
from typing import List, Union
from library.config import Config


class Tokenizer:
    """
    Tokenizer class responsible for converting InChI text to sequences of indices
    and decoding sequences of indices back to text using CTC greedy decoding logic.
    """

    def __init__(self):
        """
        Initialize the Tokenizer with vocabulary and mappings from the Config.
        """
        self.char2idx = Config.CHAR2IDX
        self.idx2char = Config.IDX2CHAR
        self.blank_idx = Config.BLANK_IDX
        self.vocab = Config.VOCAB

    def text_to_sequence(self, text: str) -> torch.Tensor:
        """
        Convert a raw InChI string into a sequence of integer indices.

        Args:
            text (str): The InChI string to encode.

        Returns:
            torch.Tensor: A 1D LongTensor containing the indices.
        """
        indices = []
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                # In a real scenario, we might handle unknown chars.
                # Given the closed vocabulary derived from training data, we skip or ignore.
                continue
        return torch.tensor(indices, dtype=torch.long)

    def sequence_to_text(
        self, sequence: Union[torch.Tensor, np.ndarray, List[int]]
    ) -> str:
        """
        Decode a sequence of indices into a string using CTC greedy decoding.

        CTC Greedy Decoding Logic:
        1. Collapse consecutive duplicate indices (e.g., [1, 1, 2] -> [1, 2]).
        2. Discard blank tokens.

        Args:
            sequence: A 1D iterable (Tensor, array, or list) of indices.

        Returns:
            str: The decoded InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.detach().cpu().numpy()
        elif isinstance(sequence, list):
            sequence = np.array(sequence)

        decoded_chars = []
        prev_idx = -1

        for idx in sequence:
            # Ensure idx is a standard python int for dict lookup if needed,
            # though numpy scalars usually work.
            idx = int(idx)

            # CTC Logic: Only append if it's not a repeat of the immediate previous
            # token. Note: The blank token acts as a separator for repeats.
            if idx != prev_idx:
                if idx != self.blank_idx:
                    # Map index to character.
                    # We use .get() to handle potential out-of-bound indices gracefully, though unlikely.
                    char = self.idx2char.get(idx, "")
                    decoded_chars.append(char)
                prev_idx = idx

        return "".join(decoded_chars)

    def decode_batch(self, sequences: torch.Tensor) -> List[str]:
        """
        Decode a batch of sequences.

        Args:
            sequences (torch.Tensor): A 2D tensor of shape (batch_size, sequence_length)
                                      containing class indices (usually after argmax).

        Returns:
            List[str]: A list of decoded InChI strings.
        """
        # Ensure tensor is on CPU before iteration
        sequences = sequences.detach().cpu()

        decoded_batch = []
        for seq in sequences:
            decoded_string = self.sequence_to_text(seq)
            decoded_batch.append(decoded_string)

        return decoded_batch
