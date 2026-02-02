import numpy as np
import torch
from library.config import Config


class InChITokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    Handles vocabulary mapping, special tokens, padding, and truncation.
    """

    def __init__(self):
        self.token2idx = Config.TOKEN2IDX
        self.idx2token = Config.IDX2TOKEN
        self.pad_idx = Config.PAD_IDX
        self.sos_idx = Config.SOS_IDX
        self.eos_idx = Config.EOS_IDX
        self.max_len = Config.MAX_TEXT_LEN
        self.vocab_size = Config.VOCAB_SIZE

    def text_to_sequence(self, text):
        """
        Converts a text string to a sequence of indices.
        Adds SOS at the beginning and EOS at the end.
        Pads with PAD token to max_len.

        Args:
            text (str): The InChI string to convert.

        Returns:
            np.ndarray: Array of integer indices with shape (max_len,).
        """
        sequence = [self.sos_idx]

        for char in text:
            if char in self.token2idx:
                sequence.append(self.token2idx[char])
            # Unknown characters are skipped based on EDA showing full coverage

        sequence.append(self.eos_idx)

        # Truncate if necessary
        if len(sequence) > self.max_len:
            sequence = sequence[: self.max_len]
            # Ensure the sequence ends with EOS if truncated
            sequence[-1] = self.eos_idx

        # Pad
        if len(sequence) < self.max_len:
            pad_len = self.max_len - len(sequence)
            sequence.extend([self.pad_idx] * pad_len)

        return np.array(sequence, dtype=np.int64)

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of indices back to a text string.
        Stops decoding when EOS or PAD token is encountered.
        Ignores SOS token.

        Args:
            sequence (iterable): List or array of integer indices.

        Returns:
            str: The decoded InChI string.
        """
        chars = []
        for idx in sequence:
            idx = int(idx)
            if idx == self.sos_idx:
                continue
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx:
                break

            if idx in self.idx2token:
                chars.append(self.idx2token[idx])

        return "".join(chars)

    def batch_decode(self, sequences):
        """
        Decodes a batch of sequences into a list of strings.

        Args:
            sequences (torch.Tensor or np.ndarray): Batch of sequences.

        Returns:
            List[str]: List of decoded strings.
        """
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.detach().cpu().numpy()

        texts = []
        for seq in sequences:
            texts.append(self.sequence_to_text(seq))
        return texts
