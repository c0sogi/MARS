import os
import json
import sentencepiece as spm
import pandas as pd
from collections import Counter
from library.config import Config
from library.utils import clean_text


class CharTokenizer:
    """
    Character-level tokenizer for the input encoder.
    Maps characters to integer IDs, preserving exact spelling for digits and symbols.
    """

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.sep_token = "<sep>"  # Used to separate context tokens

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token,
            self.sep_token,
        ]

        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.char2id[token] = idx
            self.id2char[idx] = token

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.

        Args:
            texts (list[str]): List of input strings.
        """
        counter = Counter()
        for text in texts:
            if not isinstance(text, str):
                continue
            counter.update(text)

        # Sort characters by frequency for deterministic ordering
        sorted_chars = sorted(counter.keys())

        start_idx = len(self.special_tokens)
        for idx, char in enumerate(sorted_chars):
            real_idx = start_idx + idx
            self.char2id[char] = real_idx
            self.id2char[real_idx] = char

    def save(self, path):
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)

        self.id2char = {int(v): k for k, v in self.char2id.items()}

    def encode(self, text, max_len=None, add_special_tokens=False):
        """
        Converts a string to a list of IDs.

        Args:
            text (str): Input string.
            max_len (int, optional): Maximum length for truncation/padding.
            add_special_tokens (bool): Whether to add SOS/EOS.

        Returns:
            list[int]: List of token IDs.
        """
        if not isinstance(text, str):
            text = str(text)

        ids = []
        if add_special_tokens:
            ids.append(self.char2id[self.sos_token])

        # We need to handle the <sep> token if it appears in the input string
        # (e.g. if the input string is pre-formatted with separators)
        # However, usually the input text is raw chars.
        # If the caller passes "prev <sep> target", we treat <sep> as a special token, not chars.
        # Simple heuristic: split by <sep> if present, encode parts, join by sep_id.

        if self.sep_token in text:
            parts = text.split(self.sep_token)
            sep_id = self.char2id[self.sep_token]
            unk_id = self.char2id[self.unk_token]

            encoded_parts = []
            for i, part in enumerate(parts):
                part_ids = [self.char2id.get(c, unk_id) for c in part]
                encoded_parts.extend(part_ids)
                if i < len(parts) - 1:
                    encoded_parts.append(sep_id)
            ids.extend(encoded_parts)
        else:
            unk_id = self.char2id[self.unk_token]
            ids.extend([self.char2id.get(c, unk_id) for c in text])

        if add_special_tokens:
            ids.append(self.char2id[self.eos_token])

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.char2id[self.pad_token]] * (max_len - len(ids))

        return ids

    def decode(self, ids, remove_special_tokens=True):
        """Converts a list of IDs back to a string."""
        chars = []
        for i in ids:
            token = self.id2char.get(i, "")
            if remove_special_tokens and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def get_vocab_size(self):
        return len(self.char2id)


class TargetBPETokenizer:
    """
    Subword-level tokenizer (BPE) for the output decoder.
    Uses SentencePiece to handle Russian morphology efficiently.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.model_loaded = False

    def train(self, texts, model_prefix, vocab_size):
        """
        Trains a SentencePiece BPE model.

        Args:
            texts (list[str]): List of target strings.
            model_prefix (str): Prefix for the output model file.
            vocab_size (int): Desired vocabulary size.
        """
        # Create a temporary file for training data
        temp_file = os.path.join(os.path.dirname(model_prefix), "temp_sp_train.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                if isinstance(text, str) and text.strip():
                    f.write(text + "\n")

        # Train SentencePiece
        # We use unigram or bpe. BPE is standard for this type of task.
        # pad_id=0, sos_id=1, eos_id=2, unk_id=3 to match typical PyTorch conventions (though SP defaults differ)
        # We align with Config defaults implicitly or explicit args.
        # Config doesn't specify IDs, so we stick to SP defaults but ensure we know them.
        # SP defaults: <unk>=0, <s>=1, </s>=2.
        # We will map them: pad=0, sos=1, eos=2, unk=3 via user_defined_symbols or control symbols?
        # Simpler: Use SP defaults and handle mapping in dataset or just use SP IDs.
        # Let's use SP defaults for simplicity: unk=0, bos=1, eos=2. We need a pad token.
        # We will add <pad> as a control symbol (ID 3).

        cmd = (
            f"--input={temp_file} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--model_type=bpe "
            f"--character_coverage=1.0 "
            f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 "  # Aligning with common practice
        )

        spm.SentencePieceTrainer.Train(cmd)

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.load(model_prefix + ".model")

    def load(self, model_path):
        """Loads a trained SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.sp.Load(model_path)
        self.model_loaded = True

    def encode(self, text, max_len=None, add_special_tokens=True):
        """
        Encodes text to IDs.

        Args:
            text (str): Input text.
            max_len (int, optional): Max length.
            add_special_tokens (bool): Add BOS/EOS.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded.")

        if not isinstance(text, str):
            text = str(text)

        if add_special_tokens:
            ids = self.sp.EncodeAsIds(text)
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        else:
            ids = self.sp.EncodeAsIds(text)

        if max_len is not None:
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.sp.pad_id()] * (max_len - len(ids))
        return ids

    def decode(self, ids, remove_special_tokens=True):
        """Decodes IDs to text."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded.")

        # Filter out pad tokens
        valid_ids = [i for i in ids if i != self.sp.pad_id()]

        if remove_special_tokens:
            # SP decode handles BOS/EOS usually, but explicit filtering is safer if manual addition
            valid_ids = [
                i for i in valid_ids if i not in [self.sp.bos_id(), self.sp.eos_id()]
            ]

        return self.sp.DecodeIds(valid_ids)

    def get_vocab_size(self):
        if not self.model_loaded:
            return 0
        return self.sp.GetPieceSize()

    @property
    def pad_token_id(self):
        return self.sp.pad_id()

    @property
    def bos_token_id(self):
        return self.sp.bos_id()

    @property
    def eos_token_id(self):
        return self.sp.eos_id()


def train_tokenizers(load_cached_data=True):
    """
    Orchestrates the training or loading of both tokenizers.

    Args:
        load_cached_data (bool): If True, attempts to load existing models from disk.

    Returns:
        tuple: (CharTokenizer, TargetBPETokenizer)
    """
    char_tokenizer = CharTokenizer()
    target_tokenizer = TargetBPETokenizer()

    # Check if artifacts exist
    char_vocab_exists = os.path.exists(Config.CHAR_VOCAB_PATH)
    target_model_exists = os.path.exists(Config.TARGET_TOKENIZER_MODEL)

    if load_cached_data and char_vocab_exists and target_model_exists:
        print("Loading cached tokenizers...")
        char_tokenizer.load(Config.CHAR_VOCAB_PATH)
        target_tokenizer.load(Config.TARGET_TOKENIZER_MODEL)
        return char_tokenizer, target_tokenizer

    print("Training tokenizers from scratch...")

    # Load training data
    # We only need 'before' for CharTokenizer and 'after' for TargetBPETokenizer
    # Loading full CSV might be heavy, but necessary.
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {Config.TRAIN_DATA_PATH}")

    df = pd.read_csv(Config.TRAIN_DATA_PATH)

    # 1. Train CharTokenizer
    # It needs to see all characters that might appear in input.
    # The input is 'before' column.
    print("Fitting CharTokenizer...")
    input_texts = df["before"].dropna().astype(str).tolist()
    char_tokenizer.fit_on_texts(input_texts)
    char_tokenizer.save(Config.CHAR_VOCAB_PATH)
    print(f"CharTokenizer saved. Vocab size: {char_tokenizer.get_vocab_size()}")

    # 2. Train TargetBPETokenizer
    # It needs to learn subwords from the 'after' column (normalized text).
    print("Training TargetBPETokenizer...")
    target_texts = df["after"].dropna().astype(str).tolist()

    # Remove file extension for prefix as SP adds .model
    model_prefix = Config.TARGET_TOKENIZER_MODEL.replace(".model", "")

    target_tokenizer.train(
        texts=target_texts,
        model_prefix=model_prefix,
        vocab_size=Config.TARGET_VOCAB_SIZE,
    )
    print(f"TargetBPETokenizer saved. Vocab size: {target_tokenizer.get_vocab_size()}")

    return char_tokenizer, target_tokenizer
