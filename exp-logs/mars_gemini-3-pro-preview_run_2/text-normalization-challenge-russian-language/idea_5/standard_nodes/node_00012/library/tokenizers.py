import os
import json
import sentencepiece as spm
import pandas as pd
from library.config import Config
from library.utils import set_seed


class HeterogeneousTokenizer:
    """
    Implements a heterogeneous tokenization strategy:
    - Source (Input): Character-level tokenization for precise symbol/digit parsing.
    - Target (Output): BPE (Subword) tokenization for valid Russian morphology generation.
    """

    def __init__(self):
        self.source_vocab = {}
        self.id2source = {}
        self.sp_model = spm.SentencePieceProcessor()
        self.is_fitted = False

        # Special tokens from Config
        self.pad_token = Config.PAD_TOKEN
        self.unk_token = Config.UNK_TOKEN
        self.sos_token = Config.SOS_TOKEN
        self.eos_token = Config.EOS_TOKEN
        self.sep_token = Config.SEP_TOKEN

        # Special token IDs (will be populated after fit/load)
        self.pad_id = 0
        self.unk_id = 1
        self.sos_id = 2
        self.eos_id = 3
        self.sep_id = 4

    def fit(self, train_df, load_cached_data=True):
        """
        Trains or loads the tokenizers.

        Args:
            train_df (pd.DataFrame): Training dataframe containing 'before' and 'after' columns.
            load_cached_data (bool): If True, attempts to load from disk.
        """
        Config.setup_directories()

        # Paths
        vocab_path = Config.VOCAB_PATH
        bpe_prefix = Config.BPE_MODEL_PREFIX
        bpe_model_path = f"{bpe_prefix}.model"

        # Check if artifacts exist
        source_exists = os.path.exists(vocab_path)
        target_exists = os.path.exists(bpe_model_path)

        if load_cached_data and source_exists and target_exists:
            print("Tokenizer: Loading cached vocabularies...")
            self._load_source_vocab(vocab_path)
            self._load_target_model(bpe_model_path)
        else:
            print("Tokenizer: Building vocabularies from scratch...")
            # 1. Build Source Vocab (Character level)
            self._build_source_vocab(train_df, vocab_path)

            # 2. Train Target BPE (Subword level)
            self._train_target_bpe(train_df, bpe_prefix)
            self._load_target_model(bpe_model_path)

        self.is_fitted = True
        print(
            f"Tokenizer: Ready. Source Vocab Size: {len(self.source_vocab)}, Target Vocab Size: {self.sp_model.get_piece_size()}"
        )

    def _build_source_vocab(self, df, save_path):
        """
        Builds character-level vocabulary from the 'before' column.
        """
        print("Tokenizer: Extracting characters from source text...")
        unique_chars = set()

        # Iterate over all source tokens to find unique characters
        # Using a set comprehension or loop. For large datasets, chunking might be safer,
        # but unique chars fit in memory easily.
        # We ensure 'before' is string
        texts = df["before"].astype(str).tolist()
        for text in texts:
            unique_chars.update(text)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Define special tokens first
        vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sos_token: 2,
            self.eos_token: 3,
            self.sep_token: 4,
        }

        current_id = 5
        for char in sorted_chars:
            if char not in vocab:
                vocab[char] = current_id
                current_id += 1

        # Save to JSON
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        self.source_vocab = vocab
        self.id2source = {v: k for k, v in vocab.items()}

        # Update special IDs based on generated vocab
        self.pad_id = vocab[self.pad_token]
        self.unk_id = vocab[self.unk_token]
        self.sos_id = vocab[self.sos_token]
        self.eos_id = vocab[self.eos_token]
        self.sep_id = vocab[self.sep_token]

    def _load_source_vocab(self, path):
        """Loads character vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            self.source_vocab = json.load(f)
        self.id2source = {v: k for k, v in self.source_vocab.items()}

        self.pad_id = self.source_vocab.get(self.pad_token, 0)
        self.unk_id = self.source_vocab.get(self.unk_token, 1)
        self.sos_id = self.source_vocab.get(self.sos_token, 2)
        self.eos_id = self.source_vocab.get(self.eos_token, 3)
        self.sep_id = self.source_vocab.get(self.sep_token, 4)

    def _train_target_bpe(self, df, model_prefix):
        """
        Trains SentencePiece BPE model on 'after' column.
        """
        print("Tokenizer: Training SentencePiece BPE on target text...")

        # Create a temporary file for training data
        temp_file = os.path.join(Config.WORKING_DIR, "temp_bpe_train.txt")

        # Filter out empty strings just in case
        texts = df["after"].dropna().astype(str).tolist()

        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        # Train SentencePiece
        # We map our special tokens to SP's control symbols or user defined symbols
        # pad_id=0, unk_id=1, bos_id=2, eos_id=3 to match our convention if possible,
        # but SP has its own defaults. We will align them via arguments.
        # Config.BPE_VOCAB_SIZE usually 8000

        spm.SentencePieceTrainer.Train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=Config.BPE_VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=self.pad_token,
            unk_piece=self.unk_token,
            bos_piece=self.sos_token,
            eos_piece=self.eos_token,
            user_defined_symbols=[self.sep_token],
        )

        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def _load_target_model(self, model_path):
        """Loads the trained SentencePiece model."""
        self.sp_model.Load(model_path)

    def encode_source(self, text):
        """
        Encodes source text into character IDs.
        Args:
            text (str): Input text.
        Returns:
            list[int]: List of character IDs.
        """
        if not isinstance(text, str):
            text = str(text)

        return [self.source_vocab.get(char, self.unk_id) for char in text]

    def encode_target(self, text):
        """
        Encodes target text into BPE subword IDs.
        Args:
            text (str): Target text.
        Returns:
            list[int]: List of token IDs.
        """
        if not isinstance(text, str):
            text = str(text)
        # EncodeAsIds returns a list of ints
        return self.sp_model.EncodeAsIds(text)

    def decode_target(self, ids):
        """
        Decodes BPE IDs back to string.
        Args:
            ids (list[int]): List of token IDs.
        Returns:
            str: Decoded text.
        """
        # Ensure ids are standard python ints
        ids = [int(i) for i in ids]
        return self.sp_model.DecodeIds(ids)

    def get_source_vocab_size(self):
        return len(self.source_vocab)

    def get_target_vocab_size(self):
        return self.sp_model.get_piece_size()
