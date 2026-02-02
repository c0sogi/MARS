import json
import os
from typing import List, Dict, Optional, Union
import pandas as pd
from library.config import Config


class CharTokenizer:
    """
    A character-level tokenizer for text normalization.

    Handles the mapping between characters and integer IDs, including special tokens
    required for the Retrieval-Augmented Transformer (e.g., <sep> for separating
    context and retrieval results).
    """

    def __init__(self):
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}
        # Special tokens:
        # <pad>: Padding
        # <sos>: Start of Sequence
        # <eos>: End of Sequence
        # <unk>: Unknown character
        # <sep>: Separator for RAG inputs
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", "<sep>"]

        # Initialize vocab with special tokens
        for i, token in enumerate(self.special_tokens):
            self.char2id[token] = i
            self.id2char[i] = token

    @property
    def pad_token_id(self) -> int:
        return self.char2id["<pad>"]

    @property
    def sos_token_id(self) -> int:
        return self.char2id["<sos>"]

    @property
    def eos_token_id(self) -> int:
        return self.char2id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.char2id["<unk>"]

    @property
    def sep_token_id(self) -> int:
        return self.char2id["<sep>"]

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def train(self, texts: List[str]):
        """
        Builds the vocabulary from a list of text strings.

        Args:
            texts: List of strings to learn characters from.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for deterministic vocabulary generation
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2id[char] = idx
            self.id2char[idx] = char

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts a string into a list of token IDs.

        Args:
            text: Input string.
            add_special_tokens: If True, wraps the sequence with <sos> and <eos>.
        """
        text = str(text)
        ids = [self.char2id.get(c, self.unk_token_id) for c in text]
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of token IDs back into a string.

        Args:
            ids: List of integer IDs.
            skip_special_tokens: If True, omits special tokens from the output.
        """
        chars = []
        for i in ids:
            if i not in self.id2char:
                continue
            token = self.id2char[i]
            if skip_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str):
        """
        Saves the vocabulary to a JSON file.
        """
        data = {
            "char2id": self.char2id,
            # Convert int keys to str for JSON compatibility
            "id2char": {str(k): v for k, v in self.id2char.items()},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """
        Loads the vocabulary from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char2id = data["char2id"]
        # Convert str keys back to int
        self.id2char = {int(k): v for k, v in data["id2char"].items()}


def is_hard_token(token_class: str, token_text: str) -> bool:
    """
    Determines if a token requires neural normalization (is 'Hard').

    Used to filter the training dataset. A token is considered 'Hard' if:
    1. It is NOT classified as 'PLAIN' or 'PUNCT'.
    2. It is NOT purely alphabetic (to exclude simple proper nouns or words).

    Args:
        token_class: The class label of the token (e.g., 'DATE', 'PLAIN').
        token_text: The raw text of the token.

    Returns:
        bool: True if the token is a candidate for the neural model.
    """
    # 1. Class-based filtering
    if token_class in ["PLAIN", "PUNCT"]:
        return False

    # 2. Content-based filtering
    # Purely alphabetic tokens (e.g., "Mountain") are usually identity mappings
    # or handled by simple lookups, even if labeled differently.
    if str(token_text).isalpha():
        return False

    return True


def get_context_window(df: pd.DataFrame, index: int, window_size: int = 2) -> str:
    """
    Retrieves the surrounding context for a token at a specific index.
    Ensures that the context does not cross sentence boundaries.

    Args:
        df: DataFrame containing 'sentence_id' and 'before' columns.
        index: The row index of the target token.
        window_size: Number of tokens to include on the left and right.

    Returns:
        str: A space-separated string containing the left context, target token,
             and right context (e.g., "The quick brown fox").
    """
    try:
        row = df.iloc[index]
        sent_id = row["sentence_id"]
        target_text = str(row["before"])
    except IndexError:
        return ""

    # Collect Left Context
    left_context = []
    for i in range(1, window_size + 1):
        prev_idx = index - i
        if prev_idx < 0:
            break

        # Check if previous token is in the same sentence
        if df.iloc[prev_idx]["sentence_id"] == sent_id:
            left_context.append(str(df.iloc[prev_idx]["before"]))
        else:
            break
    # Reverse to maintain correct order (far-left -> near-left)
    left_context.reverse()

    # Collect Right Context
    right_context = []
    for i in range(1, window_size + 1):
        next_idx = index + i
        if next_idx >= len(df):
            break

        # Check if next token is in the same sentence
        if df.iloc[next_idx]["sentence_id"] == sent_id:
            right_context.append(str(df.iloc[next_idx]["before"]))
        else:
            break

    # Construct full context string
    full_context_list = left_context + [target_text] + right_context
    return " ".join(full_context_list)
