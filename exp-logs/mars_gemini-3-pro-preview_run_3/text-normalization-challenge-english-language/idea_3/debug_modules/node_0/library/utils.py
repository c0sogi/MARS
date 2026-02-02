import os
import random
import json
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class CharTokenizer:
    """
    Character-level tokenizer for the Transformer model.
    Handles vocabulary creation, encoding, decoding, and persistence.
    """

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.special_tokens = Config.SPECIAL_TOKENS

        # Initialize vocabulary with special tokens
        for i, token in enumerate(self.special_tokens):
            self.char2id[token] = i
            self.id2char[i] = token

    def fit_on_texts(self, texts):
        """
        Builds the vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            # Ensure text is string and handle potential NaNs
            text_str = str(text) if text is not None else ""
            unique_chars.update(text_str)

        # Sort characters to ensure deterministic vocabulary order
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2id[char] = idx
            self.id2char[idx] = char

    def encode(self, text, add_special_tokens=False):
        """
        Converts a string to a list of token IDs.
        """
        text = str(text) if text is not None else ""
        ids = []
        unk_id = self.char2id[Config.TOKEN_UNK]

        for char in text:
            ids.append(self.char2id.get(char, unk_id))

        if add_special_tokens:
            sos_id = self.char2id[Config.TOKEN_SOS]
            eos_id = self.char2id[Config.TOKEN_EOS]
            ids = [sos_id] + ids + [eos_id]

        return ids

    def decode(self, ids, remove_special_tokens=True):
        """
        Converts a list of token IDs back to a string.
        """
        chars = []
        for idx in ids:
            # Handle PyTorch tensors
            if isinstance(idx, torch.Tensor):
                idx = idx.item()

            token = self.id2char.get(idx, "")

            if remove_special_tokens:
                if token in self.special_tokens:
                    continue

            chars.append(token)

        return "".join(chars)

    def save(self, path):
        """
        Saves the vocabulary to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "char2id": self.char2id,
                    # Convert int keys to strings for JSON compatibility
                    "id2char": {str(k): v for k, v in self.id2char.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path):
        """
        Loads the vocabulary from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char2id = data["char2id"]
            # Convert keys back to integers
            self.id2char = {int(k): v for k, v in data["id2char"].items()}

    def __len__(self):
        return len(self.char2id)

    @property
    def pad_token_id(self):
        return self.char2id[Config.TOKEN_PAD]

    @property
    def sos_token_id(self):
        return self.char2id[Config.TOKEN_SOS]

    @property
    def eos_token_id(self):
        return self.char2id[Config.TOKEN_EOS]

    @property
    def sep_token_id(self):
        return self.char2id[Config.TOKEN_SEP]


def build_or_load_tokenizer(texts, vocab_path, load_cached_data=True):
    """
    Deterministic caching logic for the tokenizer.

    Args:
        texts: List of strings to fit on if cache is missing.
        vocab_path: Path to save/load the tokenizer JSON.
        load_cached_data: Boolean flag to enable loading from cache.

    Returns:
        A fitted CharTokenizer instance.
    """
    # Ensure working directory exists
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    tokenizer = CharTokenizer()

    if load_cached_data and os.path.exists(vocab_path):
        print(f"Loading tokenizer from {vocab_path}...")
        try:
            tokenizer.load(vocab_path)
            return tokenizer
        except Exception as e:
            print(f"Failed to load tokenizer: {e}. Rebuilding...")

    print("Building tokenizer from scratch...")
    tokenizer.fit_on_texts(texts)
    tokenizer.save(vocab_path)
    print(f"Tokenizer saved to {vocab_path}. Vocab size: {len(tokenizer)}")

    return tokenizer


def encode_context_window(tokenizer, prev_tok, curr_tok, next_tok):
    """
    Formats and encodes the input context window for the neural model.
    Structure: encode(prev) + [SEP] + encode(curr) + [SEP] + encode(next)

    Args:
        tokenizer: Instance of CharTokenizer.
        prev_tok: String for the previous token (or None).
        curr_tok: String for the current token (target input).
        next_tok: String for the next token (or None).

    Returns:
        List of integer IDs.
    """
    sep_id = tokenizer.sep_token_id

    # Handle None or NaN values
    p = str(prev_tok) if prev_tok is not None and str(prev_tok) != "nan" else ""
    c = str(curr_tok) if curr_tok is not None and str(curr_tok) != "nan" else ""
    n = str(next_token) if next_tok is not None and str(next_tok) != "nan" else ""

    # Encode parts without adding SOS/EOS individually
    ids_p = tokenizer.encode(p, add_special_tokens=False)
    ids_c = tokenizer.encode(c, add_special_tokens=False)
    ids_n = tokenizer.encode(n, add_special_tokens=False)

    # Construct sequence: [prev] <SEP> [curr] <SEP> [next]
    # Note: SOS and EOS are typically added at the boundaries of the FULL sequence
    # by the Dataset class or Collator, or here if we treat this as the full input.
    # Based on standard Seq2Seq, the Encoder input doesn't strictly need SOS/EOS,
    # but the Decoder target does. We return the raw ID sequence here.
    return ids_p + [sep_id] + ids_c + [sep_id] + ids_n
