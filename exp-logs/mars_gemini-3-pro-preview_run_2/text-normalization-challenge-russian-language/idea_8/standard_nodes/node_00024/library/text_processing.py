import os
import json
import re
import sentencepiece as spm
from library.config import Config


class CharTokenizer:
    """
    Character-level tokenizer for the input text (raw tokens).
    Handles mapping characters to integer indices and vice versa.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    SEP_TOKEN = "<SEP>"

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.SEP_TOKEN,
        ]
        # Initialize with special tokens
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

    @property
    def vocab_size(self):
        return len(self.char2idx)

    @property
    def pad_token_id(self):
        return self.char2idx[self.PAD_TOKEN]

    @property
    def unk_token_id(self):
        return self.char2idx[self.UNK_TOKEN]

    @property
    def sos_token_id(self):
        return self.char2idx[self.SOS_TOKEN]

    @property
    def eos_token_id(self):
        return self.char2idx[self.EOS_TOKEN]

    @property
    def sep_token_id(self):
        return self.char2idx[self.SEP_TOKEN]

    def fit(self, texts):
        """
        Builds vocabulary from a list of text strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def encode(self, text, max_len=None, add_special_tokens=False):
        """
        Converts a string to a list of indices.
        """
        text = str(text)
        indices = [self.char2idx.get(c, self.unk_token_id) for c in text]

        if add_special_tokens:
            indices = [self.sos_token_id] + indices + [self.eos_token_id]

        if max_len is not None:
            indices = indices[:max_len]
            if len(indices) < max_len:
                indices += [self.pad_token_id] * (max_len - len(indices))

        return indices

    def decode(self, indices, remove_special_tokens=True):
        """
        Converts a list of indices back to a string.
        """
        chars = []
        for idx in indices:
            if remove_special_tokens and idx < len(self.special_tokens):
                continue
            chars.append(self.idx2char.get(idx, ""))
        return "".join(chars)

    def save(self, path):
        """Saves the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Loads the vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}


class TargetTokenizer:
    """
    BPE Tokenizer for the target text (normalized tokens) using SentencePiece.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.model_path = None

    def train(self, texts, vocab_size, model_prefix):
        """
        Trains a SentencePiece model.

        Args:
            texts: List of strings to train on.
            vocab_size: Desired vocabulary size.
            model_prefix: Path prefix for saving the model (without extension).
        """
        # SentencePiece requires a text file as input
        temp_file = f"{model_prefix}_train.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(str(text) + "\n")

        # Train model
        # user_defined_symbols can be added if needed
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.load(f"{model_prefix}.model")

    def load(self, model_path):
        """Loads a trained SentencePiece model."""
        self.model_path = model_path
        self.sp.Load(model_path)

    def encode(self, text, add_bos=True, add_eos=True):
        """Encodes text to IDs."""
        return self.sp.EncodeAsIds(str(text))
        # Note: BOS/EOS handling is often done manually in dataset or by SP depending on flags.
        # Here we rely on the caller or SP defaults.
        # For seq2seq, we usually manually prepend BOS and append EOS in the Dataset class.

    def decode(self, ids):
        """Decodes IDs to text."""
        return self.sp.DecodeIds(ids)

    def encode_as_pieces(self, text):
        return self.sp.EncodeAsPieces(str(text))

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def unk_id(self):
        return self.sp.unk_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()


def is_semiotic(text):
    """
    Determines if a token is semiotic (contains digits or Latin characters).
    Used for routing logic.
    """
    if not isinstance(text, str):
        text = str(text)
    # Check for digits or Latin characters
    return bool(re.search(r"[\d|a-zA-Z]", text))


def build_tokenizers(train_df=None, load_cached_data=True):
    """
    Orchestrates the creation or loading of tokenizers.

    Args:
        train_df: Pandas DataFrame containing 'before' and 'after' columns.
                  Required if load_cached_data is False or cache is missing.
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        tuple: (CharTokenizer, TargetTokenizer)
    """
    Config.setup_dirs()

    char_tokenizer = CharTokenizer()
    target_tokenizer = TargetTokenizer()

    char_vocab_exists = os.path.exists(Config.CHAR_VOCAB_PATH)
    bpe_model_exists = os.path.exists(Config.BPE_MODEL_PATH)

    # 1. Try to load from cache
    if load_cached_data and char_vocab_exists and bpe_model_exists:
        print(f"Loading tokenizers from cache...")
        char_tokenizer.load(Config.CHAR_VOCAB_PATH)
        target_tokenizer.load(Config.BPE_MODEL_PATH)
        print(f"Loaded CharTokenizer (vocab={char_tokenizer.vocab_size})")
        print(f"Loaded TargetTokenizer (vocab={target_tokenizer.vocab_size})")
        return char_tokenizer, target_tokenizer

    # 2. Train from scratch
    if train_df is None:
        raise ValueError(
            "Training data (train_df) is required to build tokenizers when cache is missing or ignored."
        )

    print("Building tokenizers from scratch...")

    # Train Char Tokenizer on input text ('before')
    print("Fitting CharTokenizer...")
    # Ensure all inputs are strings
    inputs = train_df["before"].fillna("").astype(str).tolist()
    char_tokenizer.fit(inputs)
    char_tokenizer.save(Config.CHAR_VOCAB_PATH)
    print(
        f"Saved CharTokenizer to {Config.CHAR_VOCAB_PATH} (vocab={char_tokenizer.vocab_size})"
    )

    # Train Target Tokenizer on output text ('after')
    print("Training TargetTokenizer (BPE)...")
    outputs = train_df["after"].fillna("").astype(str).tolist()
    # Remove extension for model_prefix as SP adds .model and .vocab
    model_prefix = Config.BPE_MODEL_PREFIX
    target_tokenizer.train(outputs, Config.BPE_VOCAB_SIZE, model_prefix)
    print(
        f"Saved TargetTokenizer to {Config.BPE_MODEL_PATH} (vocab={target_tokenizer.vocab_size})"
    )

    return char_tokenizer, target_tokenizer
