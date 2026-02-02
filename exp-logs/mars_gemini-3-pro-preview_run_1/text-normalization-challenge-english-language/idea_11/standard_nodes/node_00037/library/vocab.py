import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import sentencepiece as spm
from library.config import Config
from library.utils import setup_logger, ensure_dir

logger = setup_logger("vocab")


class Vocabulary:
    """
    Generic Vocabulary class to map tokens to indices and vice versa.
    Handles special tokens like <pad>, <unk>, <sos>, <eos>.
    """

    def __init__(self, name, specials=None):
        self.name = name
        self.specials = specials if specials is not None else []
        self.token2idx = {}
        self.idx2token = {}
        self.counts = Counter()

        # Initialize specials
        for token in self.specials:
            self.add_token(token)

    def add_token(self, token):
        """Adds a token to the vocabulary if it doesn't exist."""
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build(self, tokens, max_size=None, min_freq=1):
        """
        Builds vocabulary from a list/iterator of tokens.

        Args:
            tokens: Iterable of tokens.
            max_size: Maximum size of vocabulary (excluding specials).
            min_freq: Minimum frequency for a token to be included.
        """
        logger.info(f"Building {self.name} vocabulary...")
        self.counts = Counter(tokens)

        # Sort by frequency (descending) then alphabetically
        sorted_tokens = sorted(self.counts.items(), key=lambda x: (-x[1], x[0]))

        added_count = 0
        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size is not None and added_count >= max_size:
                break

            if token not in self.token2idx:
                self.add_token(token)
                added_count += 1

        logger.info(f"Vocabulary {self.name} built. Size: {len(self)}")

    def __len__(self):
        return len(self.token2idx)

    def __getitem__(self, token):
        """Returns index of token. Returns <unk> index if not found and <unk> exists."""
        if token in self.token2idx:
            return self.token2idx[token]
        if "<unk>" in self.token2idx:
            return self.token2idx["<unk>"]
        raise KeyError(
            f"Token '{token}' not found in vocabulary {self.name} and no <unk> token defined."
        )

    def lookup_token(self, idx):
        """Returns token for a given index."""
        if idx in self.idx2token:
            return self.idx2token[idx]
        raise KeyError(f"Index {idx} not found in vocabulary {self.name}.")

    def save(self, path):
        """Saves vocabulary to a JSON file."""
        ensure_dir(path)
        data = {"token2idx": self.token2idx, "specials": self.specials}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {self.name} vocabulary to {path}")

    def load(self, path):
        """Loads vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token2idx = data["token2idx"]
        self.specials = data.get("specials", [])
        self.idx2token = {int(v): k for k, v in self.token2idx.items()}
        logger.info(f"Loaded {self.name} vocabulary from {path}. Size: {len(self)}")


class BPETokenizer:
    """
    Wrapper around SentencePiece for BPE tokenization.
    """

    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.model_prefix = Config.BPE_MODEL_PREFIX
        self.model_path = f"{self.model_prefix}.model"
        self.vocab_path = f"{self.model_prefix}.vocab"

    def train(self, text_iterator, vocab_size=Config.MAX_VOCAB_SIZE_BPE):
        """
        Trains the BPE model.

        Args:
            text_iterator: Iterable of strings (sentences or tokens).
            vocab_size: Target vocabulary size.
        """
        logger.info("Training BPE Tokenizer...")
        ensure_dir(self.model_prefix)

        # SentencePiece needs a file input. We write to a temp file.
        temp_file = os.path.join(Config.WORK_DIR, "temp_bpe_train.txt")

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for text in text_iterator:
                    f.write(str(text) + "\n")

            # Train model
            # user_defined_symbols can be added if needed
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=self.model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,  # Cover all characters
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece="<pad>",
                unk_piece="<unk>",
                bos_piece="<sos>",
                eos_piece="<eos>",
            )
            logger.info(f"BPE training complete. Model saved to {self.model_path}")

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        self.load()

    def load(self):
        """Loads the trained BPE model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"BPE model not found at {self.model_path}")
        self.sp.Load(self.model_path)
        logger.info(f"Loaded BPE model. Vocab size: {self.sp.GetPieceSize()}")

    def encode(self, text):
        """Encodes text into subword indices."""
        return self.sp.EncodeAsIds(str(text))

    def decode(self, ids):
        """Decodes subword indices back to text."""
        return self.sp.DecodeIds(ids)

    def __len__(self):
        return self.sp.GetPieceSize()


class VocabManager:
    """
    Manages all vocabularies (Word, Char, Class, BPE).
    Handles building, caching, and loading.
    """

    def __init__(self):
        self.word_vocab = Vocabulary("words", specials=["<pad>", "<unk>"])
        self.char_vocab = Vocabulary(
            "chars", specials=["<pad>", "<unk>", "<sos>", "<eos>"]
        )
        self.class_vocab = Vocabulary(
            "classes", specials=[]
        )  # Classes usually don't need unk/pad for classification target
        self.bpe_tokenizer = BPETokenizer()

        self.paths = {
            "words": os.path.join(Config.VOCAB_DIR, "vocab_words.json"),
            "chars": os.path.join(Config.VOCAB_DIR, "vocab_chars.json"),
            "classes": os.path.join(Config.VOCAB_DIR, "vocab_classes.json"),
        }

    def build_or_load(self, load_cached_data=True):
        """
        Main entry point. Loads from cache if available and requested,
        otherwise builds from training data.
        """
        # Check if all artifacts exist
        all_exist = (
            os.path.exists(self.paths["words"])
            and os.path.exists(self.paths["chars"])
            and os.path.exists(self.paths["classes"])
            and os.path.exists(self.bpe_tokenizer.model_path)
        )

        if load_cached_data and all_exist:
            logger.info("Loading vocabularies from cache...")
            self.word_vocab.load(self.paths["words"])
            self.char_vocab.load(self.paths["chars"])
            self.class_vocab.load(self.paths["classes"])
            self.bpe_tokenizer.load()
        else:
            logger.info(
                "Cache missing or rebuild requested. Building vocabularies from scratch..."
            )
            self._build_from_scratch()

    def _build_from_scratch(self):
        """
        Reads training data and builds all vocabularies.
        """
        # Load training data
        logger.info(f"Reading training data from {Config.TRAIN_FILE}...")
        df = pd.read_csv(Config.TRAIN_FILE)

        # Ensure string types
        df["before"] = df["before"].astype(str)
        df["after"] = df["after"].astype(str)
        df["class"] = df["class"].astype(str)

        # 1. Build Word Vocabulary
        # We use 'before' column for input words
        logger.info("Building Word Vocabulary...")
        self.word_vocab.build(
            df["before"].tolist(),
            max_size=Config.MAX_VOCAB_SIZE_WORD,
            min_freq=2,  # Ignore singletons to save space/robustness
        )
        self.word_vocab.save(self.paths["words"])

        # 2. Build Character Vocabulary
        # We need chars from both 'before' (for CharCNN) and 'after' (for Seq2Seq generation)
        logger.info("Building Character Vocabulary...")
        all_chars = set()
        # Sampling or iterating all? Iterating all chars in 7M tokens is heavy but doable.
        # To be efficient, we can use a set comprehension on unique tokens.
        unique_before = df["before"].unique()
        unique_after = df["after"].unique()

        # Collect all characters
        chars_counter = Counter()
        for s in unique_before:
            chars_counter.update(s)
        for s in unique_after:
            chars_counter.update(s)

        # Build vocab manually from counter to ensure we capture everything
        # We don't set a max size for chars usually, as it's small (~100-500)
        self.char_vocab.build(chars_counter.elements())
        self.char_vocab.save(self.paths["chars"])

        # 3. Build Class Vocabulary
        logger.info("Building Class Vocabulary...")
        self.class_vocab.build(df["class"].tolist())
        self.class_vocab.save(self.paths["classes"])

        # 4. Train BPE Tokenizer
        logger.info("Training BPE Tokenizer...")
        # We train on 'before' tokens.
        # Using unique tokens is much faster and usually sufficient for subword modeling
        # provided we weight them or just assume the distribution of unique tokens covers the morphology.
        # However, for true frequency based BPE, we should use the full corpus or a large sample.
        # Given 7M rows, we can pass the full column iterator.
        self.bpe_tokenizer.train(
            df["before"].astype(str).tolist(), vocab_size=Config.MAX_VOCAB_SIZE_BPE
        )

        logger.info("All vocabularies built and saved.")

    def get_word_vocab(self):
        return self.word_vocab

    def get_char_vocab(self):
        return self.char_vocab

    def get_class_vocab(self):
        return self.class_vocab

    def get_bpe_tokenizer(self):
        return self.bpe_tokenizer
