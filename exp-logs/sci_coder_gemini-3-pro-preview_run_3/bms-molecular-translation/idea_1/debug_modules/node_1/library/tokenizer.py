import torch
from library.config import Config


class InchiTokenizer:
    """
    Tokenizer for converting InChI strings to sequences of integers and vice versa.
    Handles character-level tokenization with special tokens <PAD>, <SOS>, <EOS>, <UNK>.
    """

    def __init__(self):
        # Vocabulary from Config
        self.vocab_string = Config.VOCAB_STRING

        # Special tokens
        self.special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

        # Build mappings
        self.char2idx = {}
        self.idx2char = {}

        # Add special tokens first
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

        # Add vocabulary characters
        # The vocab_string contains the unique characters found in the dataset
        next_idx = len(self.special_tokens)
        for char in self.vocab_string:
            if char not in self.char2idx:
                self.char2idx[char] = next_idx
                self.idx2char[next_idx] = char
                next_idx += 1

        # Store IDs for easy external access
        self.pad_token_id = self.char2idx["<PAD>"]
        self.sos_token_id = self.char2idx["<SOS>"]
        self.eos_token_id = self.char2idx["<EOS>"]
        self.unk_token_id = self.char2idx["<UNK>"]

    def text_to_sequence(self, text, max_len=None, padding=True):
        """
        Converts an InChI string to a tensor sequence of indices.
        Adds <SOS> at the start and <EOS> at the end.

        Args:
            text (str): The InChI string.
            max_len (int, optional): Maximum length of the sequence.
                                     If None, no truncation/padding to specific length is performed
                                     (unless padding=True without max_len, which implies no padding logic).
            padding (bool): Whether to pad the sequence to max_len with <PAD> tokens.

        Returns:
            torch.Tensor: Sequence of integer indices (dtype=torch.long).
        """
        sequence = [self.sos_token_id]

        for char in text:
            # Map char to index, default to UNK if not found
            idx = self.char2idx.get(char, self.unk_token_id)
            sequence.append(idx)

        sequence.append(self.eos_token_id)

        # Handle truncation and padding
        if max_len is not None:
            # Truncate if longer than max_len
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
                # Note: We simply truncate. In some implementations, one might force the last token to be EOS,
                # but standard practice for fixed-length batches often just cuts off.

            # Pad if shorter than max_len
            if padding:
                pad_len = max_len - len(sequence)
                if pad_len > 0:
                    sequence.extend([self.pad_token_id] * pad_len)

        return torch.tensor(sequence, dtype=torch.long)

    def sequence_to_text(self, sequence):
        """
        Converts a sequence of indices back to an InChI string.
        Stops decoding at <EOS>. Ignores <SOS> and <PAD>.

        Args:
            sequence (torch.Tensor or list): Sequence of integer indices.

        Returns:
            str: The reconstructed InChI string.
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()

        chars = []
        for idx in sequence:
            # Handle numpy scalars or ints
            if hasattr(idx, "item"):
                idx = idx.item()

            if idx == self.eos_token_id:
                break

            if idx == self.sos_token_id:
                continue

            if idx == self.pad_token_id:
                continue

            if idx == self.unk_token_id:
                # Skip unknown tokens in reconstruction
                continue

            char = self.idx2char.get(idx, "")
            chars.append(char)

        return "".join(chars)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.char2idx)
