import os
import json
import pandas as pd
import sentencepiece as spm
from library import config
from library import utils


class CharTokenizer:
    """
    Character-level tokenizer for input sequences.
    Handles mapping between characters and integer IDs.
    """

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<SEP>"]
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.sep_token = "<SEP>"

        # Initialize with specials
        for idx, token in enumerate(self.specials):
            self.char2id[token] = idx
            self.id2char[idx] = token

    @property
    def vocab_size(self):
        return len(self.char2id)

    @property
    def pad_id(self):
        return self.char2id[self.pad_token]

    @property
    def sos_id(self):
        return self.char2id[self.sos_token]

    @property
    def eos_id(self):
        return self.char2id[self.eos_token]

    @property
    def unk_id(self):
        return self.char2id[self.unk_token]

    @property
    def sep_id(self):
        return self.char2id[self.sep_token]

    def train(self, texts):
        """
        Builds vocabulary from a list of strings.
        Args:
            texts (iterable): List of strings to extract characters from.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Add to vocab, starting after specials
        start_idx = len(self.specials)
        for i, char in enumerate(sorted_chars):
            if char not in self.char2id:
                idx = start_idx + i
                self.char2id[char] = idx
                self.id2char[idx] = char

    def save(self, path):
        """Saves vocab to JSON."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Loads vocab from JSON."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)
        self.id2char = {int(v): k for k, v in self.char2id.items()}

    def encode(self, text, add_special_tokens=False):
        """
        Converts string to list of IDs.
        Args:
            text (str): Input string.
            add_special_tokens (bool): If True, adds SOS and EOS.
        """
        text = str(text)
        ids = [self.char2id.get(c, self.unk_id) for c in text]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Converts list of IDs to string.
        """
        chars = []
        for i in ids:
            token = self.id2char.get(i, self.unk_token)
            if skip_special_tokens and token in self.specials:
                continue
            chars.append(token)
        return "".join(chars)


class BPETokenizer:
    """
    BPE Tokenizer using SentencePiece for output sequences.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.loaded = False

    def train(self, texts, model_prefix, vocab_size):
        """
        Trains SentencePiece model.
        Args:
            texts (iterable): List of strings to train on.
            model_prefix (str): Prefix for model file output.
            vocab_size (int): Target vocabulary size.
        """
        # Create directory if needed
        directory = os.path.dirname(model_prefix)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Write texts to temp file
        temp_file = os.path.join(directory, "temp_bpe_train.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(str(text) + "\n")

        # Train SentencePiece
        # We enforce PAD=0, BOS=1, EOS=2, UNK=3 to match common PyTorch conventions
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.load(model_prefix + ".model")

    def load(self, model_path):
        """Loads the SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.sp.Load(model_path)
        self.loaded = True

    def encode(self, text, add_special_tokens=False):
        """
        Encodes text to IDs.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        ids = self.sp.EncodeAsIds(str(text))
        if add_special_tokens:
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """
        Decodes IDs to text.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded.")

        if skip_special_tokens:
            # Filter out PAD, BOS, EOS
            specials = {self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id()}
            ids = [i for i in ids if i not in specials]

        return self.sp.DecodeIds(ids)

    @property
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def sos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()


def build_tokenizers(load_cached_data=True):
    """
    Builds or loads the CharTokenizer and BPETokenizer.

    Args:
        load_cached_data (bool): If True, attempts to load existing artifacts.

    Returns:
        tuple: (CharTokenizer, BPETokenizer)
    """
    char_tokenizer = CharTokenizer()
    bpe_tokenizer = BPETokenizer()

    char_exists = os.path.exists(config.CHAR_VOCAB_PATH)
    bpe_exists = os.path.exists(config.BPE_MODEL_PATH)

    if load_cached_data and char_exists and bpe_exists:
        print("Tokenizers: Loading from cache...")
        char_tokenizer.load(config.CHAR_VOCAB_PATH)
        bpe_tokenizer.load(config.BPE_MODEL_PATH)
    else:
        print("Tokenizers: Building from training data...")
        # Load training data
        try:
            df = pd.read_csv(config.TRAIN_FILE)
            # Ensure string types
            df["before"] = df["before"].fillna("").astype(str)
            df["after"] = df["after"].fillna("").astype(str)
        except FileNotFoundError:
            print(f"Error: Training file {config.TRAIN_FILE} not found.")
            return char_tokenizer, bpe_tokenizer

        # 1. Train Char Tokenizer on Input Text ('before')
        print("Tokenizers: Training CharTokenizer...")
        char_tokenizer.train(df["before"].tolist())
        char_tokenizer.save(config.CHAR_VOCAB_PATH)

        # 2. Train BPE Tokenizer on Target Text ('after')
        print("Tokenizers: Training BPETokenizer...")
        bpe_tokenizer.train(
            texts=df["after"].tolist(),
            model_prefix=config.BPE_MODEL_PREFIX,
            vocab_size=config.VOCAB_SIZE,
        )

    return char_tokenizer, bpe_tokenizer
