import torch
from library.config import Config


class InChITokenizer:
    """
    Tokenizer for converting InChI strings to integer sequences and vice versa.
    """

    def __init__(self):
        self.char2idx = Config.CHAR2IDX
        self.idx2char = Config.IDX2CHAR
        self.pad_idx = Config.PAD_IDX
        self.sos_idx = Config.SOS_IDX
        self.eos_idx = Config.EOS_IDX
        self.max_len = Config.MAX_LEN
        self.vocab_size = Config.VOCAB_SIZE

    def encode(self, text: str, max_len: int = None) -> torch.Tensor:
        """
        Encodes a text string into a tensor of indices with <SOS> and <EOS> tokens,
        padded to max_len.

        Args:
            text (str): The InChI string to encode.
            max_len (int, optional): The maximum length of the sequence.
                                     Defaults to Config.MAX_LEN.

        Returns:
            torch.Tensor: A long tensor of shape (max_len,).
        """
        if max_len is None:
            max_len = self.max_len

        # Convert characters to indices
        # We assume all characters in text are present in char2idx based on EDA
        indices = [self.char2idx[c] for c in text]

        # Add <SOS> and <EOS>
        indices = [self.sos_idx] + indices + [self.eos_idx]

        # Handle padding or truncation
        current_len = len(indices)

        if current_len < max_len:
            # Pad with <PAD> token
            padding = [self.pad_idx] * (max_len - current_len)
            indices.extend(padding)
        else:
            # Truncate if longer than max_len
            # Ensure the last token is EOS if truncated
            indices = indices[:max_len]
            indices[-1] = self.eos_idx

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> str:
        """
        Decodes a sequence of indices back into a string.
        Stops decoding when <EOS> or <PAD> is encountered.
        Ignores <SOS> token.

        Args:
            indices (torch.Tensor or list): The sequence of indices to decode.

        Returns:
            str: The decoded InChI string.
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        decoded_chars = []
        for idx in indices:
            # Skip <SOS>
            if idx == self.sos_idx:
                continue

            # Stop at <EOS> or <PAD>
            if idx == self.eos_idx or idx == self.pad_idx:
                break

            # Retrieve character
            char = self.idx2char.get(idx, "")
            decoded_chars.append(char)

        return "".join(decoded_chars)
