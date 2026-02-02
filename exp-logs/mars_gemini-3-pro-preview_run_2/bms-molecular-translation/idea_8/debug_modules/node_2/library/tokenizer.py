import torch
import numpy as np
from library.config import Config


class InChiTokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa
    using CTC greedy decoding logic. Depends on the vocabulary defined in Config.
    """

    def __init__(self):
        self.char2idx = Config.CHAR2IDX
        self.idx2char = Config.IDX2CHAR
        self.vocab = Config.VOCAB
        # CTC blank token is defined as the first element in Config.VOCAB
        self.blank_idx = 0

    def text_to_sequence(self, text):
        """
        Converts a text string into a sequence of class indices.

        Args:
            text (str): The input InChI string.

        Returns:
            torch.LongTensor: Tensor of indices corresponding to the text.
        """
        sequence = []
        for char in text:
            # Only encode characters that exist in our vocabulary
            if char in self.char2idx:
                sequence.append(self.char2idx[char])
        return torch.LongTensor(sequence)

    def decode_ctc_greedy(self, logits, batch_first=True):
        """
        Decodes model output logits using CTC greedy decoding.

        Logic:
        1. Argmax over the vocabulary dimension to get the most likely index at each step.
        2. Iterate through the sequence:
           - If the current index is the same as the previous index, it is considered a repeat
             (unless separated by a blank) and is ignored.
           - If the current index is the blank token, it is ignored but resets the 'previous' state.
           - Otherwise, map index to character and append to result.

        Args:
            logits (torch.Tensor): Logits or probabilities from the model.
                                   Shape (N, T, C) if batch_first=True, else (T, N, C).
            batch_first (bool): Whether the first dimension is batch size. Defaults to True.

        Returns:
            List[str]: List of decoded InChI strings.
        """
        # Ensure logits are on CPU for numpy processing
        if logits.is_cuda:
            logits = logits.cpu()

        # Handle time-first format by permuting to (N, T, C)
        if not batch_first:
            logits = logits.permute(1, 0, 2)

        # Get the index of the max probability at each time step
        # shape: (Batch Size, Sequence Length)
        predictions = torch.argmax(logits, dim=-1)

        predictions_np = predictions.numpy()
        decoded_strings = []

        for seq in predictions_np:
            decoded_chars = []
            prev_idx = -1

            for idx in seq:
                # CTC Greedy Decoding Logic:
                # We only append a character if it is distinct from the immediately
                # preceding character (collapsing repeats).
                # The blank token (0) acts as a separator but is not added to the string.
                # Note: If we have A, blank, A -> both As are kept.
                # If we have A, A -> one A is kept.

                if idx != prev_idx:
                    if idx != self.blank_idx:
                        decoded_chars.append(self.idx2char[idx])
                    prev_idx = idx

            decoded_strings.append("".join(decoded_chars))

        return decoded_strings
