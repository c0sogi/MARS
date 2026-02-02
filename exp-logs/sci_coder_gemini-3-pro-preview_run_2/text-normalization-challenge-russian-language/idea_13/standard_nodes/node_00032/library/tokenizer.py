import os
import json
import sentencepiece as spm
from collections import Counter
from library.config import Config
from library.utils import load_metadata, ensure_dir


class HybridTokenizer:
    """
    Handles Heterogeneous Tokenization for the Hybrid Cascade Transformer.

    1. Encoder: Character-Level Tokenizer
       - Parses internal structure of numbers, dates, and symbols.
       - Vocabulary built from 'before' (raw) text.

    2. Decoder: Subword-Level (BPE) Tokenizer
       - Handles Russian morphology and grammar.
       - Trained on 'after' (normalized) text using SentencePiece.
    """

    # Special Token Constants for Character Tokenizer
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    SEP_TOKEN = "<SEP>"

    PAD_ID = 0
    UNK_ID = 1
    START_ID = 2
    END_ID = 3
    SEP_ID = 4

    def __init__(self, config: Config):
        self.config = config

        # Define artifact paths using config hash to ensure consistency
        self.char_vocab_path = self.config.get_artifact_path("char_vocab.json")
        self.bpe_model_prefix = self.config.get_artifact_path("bpe_ru_target")
        self.bpe_model_path = self.bpe_model_prefix + ".model"
        self.bpe_vocab_path = self.bpe_model_prefix + ".vocab"

        # In-memory storage
        self.char2id = {}
        self.id2char = {}
        self.sp_processor = None

    def fit(self, load_cached_data=True):
        """
        Builds or loads the tokenizers.

        Args:
            load_cached_data (bool): If True, attempts to load existing artifacts.
        """
        # Check if artifacts exist
        artifacts_exist = os.path.exists(self.char_vocab_path) and os.path.exists(
            self.bpe_model_path
        )

        if load_cached_data and artifacts_exist:
            print(
                f"Loading tokenizers from cache: {os.path.dirname(self.char_vocab_path)}"
            )
            self._load_char_vocab()
            self._load_bpe_model()
            return

        print("Building tokenizers from scratch...")

        # Load full training data to ensure coverage
        # We need 'before' for chars and 'after' for BPE
        df = load_metadata("train")

        # 1. Build Character Vocab (Encoder)
        print("Building Character Vocabulary...")
        # We use the raw input text ('before')
        raw_texts = df["before"].astype(str).tolist()
        self._build_char_vocab(raw_texts)

        # 2. Train BPE Model (Decoder)
        print("Training BPE Model...")
        # We use the normalized target text ('after')
        target_texts = df["after"].astype(str).tolist()
        self._train_bpe(target_texts)

        print("Tokenizers ready.")

    def _build_char_vocab(self, texts):
        """
        Scans texts to build a character-to-id mapping.
        """
        counter = Counter()
        for text in texts:
            counter.update(text)

        # Get most common characters up to a reasonable limit to avoid noise
        # Reserve spots for special tokens
        vocab_size_limit = self.config.char_vocab_size - 5
        most_common = counter.most_common(vocab_size_limit)

        # Initialize with special tokens
        self.char2id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.START_TOKEN: self.START_ID,
            self.END_TOKEN: self.END_ID,
            self.SEP_TOKEN: self.SEP_ID,
        }

        # Add characters
        current_id = 5
        for char, _ in most_common:
            if char not in self.char2id:
                self.char2id[char] = current_id
                current_id += 1

        self.id2char = {v: k for k, v in self.char2id.items()}

        # Save to JSON
        ensure_dir(self.char_vocab_path)
        with open(self.char_vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def _load_char_vocab(self):
        """Loads character vocab from JSON."""
        with open(self.char_vocab_path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)
        self.id2char = {v: k for k, v in self.char2id.items()}

    def _train_bpe(self, texts):
        """
        Trains a SentencePiece BPE model on the target texts.
        """
        # SentencePiece requires a file input
        temp_file = os.path.join(self.config.base_working_dir, "temp_bpe_train.txt")
        ensure_dir(temp_file)

        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        # Train model
        # We use bos_id=1, eos_id=2, unk_id=3, pad_id=0 to align with common practices
        # though SP defaults might differ. We explicitly set them.
        cmd = (
            f"--input={temp_file} "
            f"--model_prefix={self.bpe_model_prefix} "
            f"--vocab_size={self.config.bpe_vocab_size} "
            f"--model_type=bpe "
            f"--character_coverage=1.0 "
            f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3 "
            f"--pad_piece=<pad> --bos_piece=<s> --eos_piece=</s> --unk_piece=<unk>"
        )

        # Execute training (suppress output unless debug)
        try:
            spm.SentencePieceTrainer.train(cmd)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        self._load_bpe_model()

    def _load_bpe_model(self):
        """Loads the SentencePiece processor."""
        self.sp_processor = spm.SentencePieceProcessor()
        self.sp_processor.load(self.bpe_model_path)

    def encode_char(self, text):
        """
        Encodes a string into a list of character IDs.

        Args:
            text (str): Input text.

        Returns:
            list[int]: List of character IDs.
        """
        if not isinstance(text, str):
            text = str(text)
        return [self.char2id.get(c, self.UNK_ID) for c in text]

    def encode_bpe(self, text):
        """
        Encodes a string into a list of BPE subword IDs.

        Args:
            text (str): Input text.

        Returns:
            list[int]: List of BPE IDs.
        """
        if not isinstance(text, str):
            text = str(text)
        if self.sp_processor is None:
            raise RuntimeError("BPE model not loaded. Call fit() first.")
        return self.sp_processor.encode_as_ids(text)

    def decode_bpe(self, ids):
        """
        Decodes a list of BPE IDs back into a string.

        Args:
            ids (list[int]): List of BPE IDs.

        Returns:
            str: Decoded text.
        """
        if self.sp_processor is None:
            raise RuntimeError("BPE model not loaded. Call fit() first.")
        return self.sp_processor.decode_ids(ids)

    @property
    def bpe_vocab_size(self):
        if self.sp_processor:
            return self.sp_processor.get_piece_size()
        return self.config.bpe_vocab_size

    @property
    def char_vocab_size_actual(self):
        return len(self.char2id)

    # Expose BPE special token IDs
    @property
    def bpe_pad_id(self):
        return self.sp_processor.pad_id() if self.sp_processor else 0

    @property
    def bpe_bos_id(self):
        return self.sp_processor.bos_id() if self.sp_processor else 1

    @property
    def bpe_eos_id(self):
        return self.sp_processor.eos_id() if self.sp_processor else 2
