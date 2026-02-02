import os
import json
import itertools
from typing import List, Dict, Union, Optional
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.implementations import ByteLevelBPETokenizer

from library.config import Config, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN
from library.utils import ensure_dir


class CharTokenizer:
    """
    A simple character-level tokenizer for the target token.
    Preserves precise orthography (digits, symbols) which BPE might fragment inconsistently.
    """

    def __init__(self, config: Config):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Pre-define special tokens
        self.specials = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
        for idx, token in enumerate(self.specials):
            self.vocab[token] = idx
            self.id_to_char[idx] = token

    def train(self, texts: List[str]):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Add to vocab, respecting max_char_vocab_size
        current_idx = len(self.vocab)
        max_size = self.config.max_char_vocab_size

        for char in sorted_chars:
            if char in self.vocab:
                continue
            if current_idx >= max_size:
                break
            self.vocab[char] = current_idx
            self.id_to_char[current_idx] = char
            current_idx += 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts a string to a list of character IDs.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[SOS_TOKEN])

        for char in str(text):
            ids.append(self.vocab.get(char, self.vocab[UNK_TOKEN]))

        if add_special_tokens:
            ids.append(self.vocab[EOS_TOKEN])
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Converts a list of IDs back to a string.
        """
        chars = []
        for i in ids:
            token = self.id_to_char.get(i, UNK_TOKEN)
            if skip_special_tokens and token in self.specials:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str):
        ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Char tokenizer file not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_char = {int(v): k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab[PAD_TOKEN]


class HybridTokenizer:
    """
    Manages two distinct tokenization strategies:
    1. BPE for Context words and Output (captures morphology).
    2. Character for Input Target token (preserves orthography).
    """

    def __init__(self, config: Config):
        self.config = config
        self.char_tokenizer = CharTokenizer(config)
        self.bpe_tokenizer = None  # Initialized in train or load

    def train_tokenizers(self, train_df: pd.DataFrame, load_cached_data: bool = True):
        """
        Trains the BPE tokenizer and builds the Character vocabulary.
        Uses caching based on the config paths.
        """
        bpe_path = self.config.bpe_tokenizer_path
        char_path = self.config.char_tokenizer_path

        # 1. Check Cache
        if load_cached_data and os.path.exists(bpe_path) and os.path.exists(char_path):
            print(f"Loading cached tokenizers from {self.config.working_dir}...")
            self.load(bpe_path, char_path)
            print(f"Loaded BPE Vocab Size: {self.bpe_tokenizer.get_vocab_size()}")
            print(f"Loaded Char Vocab Size: {self.char_tokenizer.vocab_size}")
            return

        print("Training tokenizers from scratch...")

        # 2. Prepare Data
        # For BPE: Use both 'before' and 'after' to ensure decoder can generate normalized text
        # We save to a temp file because tokenizers library trains from files efficiently
        temp_corpus_path = os.path.join(self.config.working_dir, "temp_bpe_corpus.txt")
        ensure_dir(temp_corpus_path)

        print("Extracting text for BPE training...")
        # Sample if dataset is massive to save time, but for best results use all
        # Given 24h limit, we can afford to write all unique tokens
        unique_before = train_df["before"].dropna().unique()
        unique_after = train_df["after"].dropna().unique()

        with open(temp_corpus_path, "w", encoding="utf-8") as f:
            for text in unique_before:
                f.write(str(text) + "\n")
            for text in unique_after:
                f.write(str(text) + "\n")

        # 3. Train BPE Tokenizer
        print("Training BPE Tokenizer...")
        # Initialize ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()

        # Customize special tokens
        special_tokens = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

        tokenizer.train(
            files=[temp_corpus_path],
            vocab_size=self.config.bpe_vocab_size,
            min_frequency=2,
            show_progress=False,  # Silent as requested
            special_tokens=special_tokens,
        )
        self.bpe_tokenizer = tokenizer

        # 4. Train Char Tokenizer
        print("Training Char Tokenizer...")
        # We only strictly need chars from 'before' for the input encoder
        # but adding 'after' doesn't hurt and keeps indices consistent if used elsewhere
        all_texts = list(unique_before) + list(unique_after)
        self.char_tokenizer.train(all_texts)

        # 5. Save Artifacts
        print(f"Saving tokenizers to {self.config.working_dir}...")
        self.save(bpe_path, char_path)

        # Cleanup
        if os.path.exists(temp_corpus_path):
            os.remove(temp_corpus_path)

        print(f"BPE Vocab Size: {self.bpe_tokenizer.get_vocab_size()}")
        print(f"Char Vocab Size: {self.char_tokenizer.vocab_size}")

    def encode(
        self, context_left: List[str], target_token: str, context_right: List[str]
    ) -> Dict[str, List[int]]:
        """
        Encodes the input components.
        Returns a dictionary with separate ID lists.
        """
        if self.bpe_tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        # Encode Contexts (BPE)
        # We join context words with space for BPE, or encode individually?
        # Encoding individually is safer to preserve word boundaries explicitly if needed,
        # but BPE handles spaces. Let's join them.
        left_str = " ".join([str(w) for w in context_left])
        right_str = " ".join([str(w) for w in context_right])

        # Use add_special_tokens=False because we manually construct the sequence in the Dataset/Model
        left_ids = self.bpe_tokenizer.encode(left_str).ids if left_str else []
        right_ids = self.bpe_tokenizer.encode(right_str).ids if right_str else []

        # Encode Target (Char)
        target_ids = self.char_tokenizer.encode(target_token, add_special_tokens=False)

        return {
            "context_left_ids": left_ids,
            "target_char_ids": target_ids,
            "context_right_ids": right_ids,
        }

    def encode_target_text(self, text: str) -> List[int]:
        """
        Encodes the output label (normalized text) using BPE.
        Adds SOS and EOS tokens.
        """
        if self.bpe_tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        # We format the string to ensure BPE handles it as a sentence/word
        encoded = self.bpe_tokenizer.encode(str(text))

        # Add SOS and EOS manually if tokenizer doesn't add them by default in this mode
        sos_id = self.bpe_tokenizer.token_to_id(SOS_TOKEN)
        eos_id = self.bpe_tokenizer.token_to_id(EOS_TOKEN)

        return [sos_id] + encoded.ids + [eos_id]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes BPE IDs back to string (for model predictions).
        """
        if self.bpe_tokenizer is None:
            raise RuntimeError("Tokenizer not trained or loaded.")

        return self.bpe_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, bpe_path: str, char_path: str):
        """
        Saves both tokenizers.
        """
        ensure_dir(bpe_path)
        ensure_dir(char_path)

        # Save BPE (tokenizers library handles json/folder structure)
        # We save the model and vocab
        self.bpe_tokenizer.save(bpe_path)

        # Save Char
        self.char_tokenizer.save(char_path)

    def load(self, bpe_path: str, char_path: str):
        """
        Loads both tokenizers.
        """
        # Load BPE
        self.bpe_tokenizer = Tokenizer.from_file(bpe_path)

        # Load Char
        self.char_tokenizer.load(char_path)

    @property
    def bpe_vocab_size(self) -> int:
        return self.bpe_tokenizer.get_vocab_size() if self.bpe_tokenizer else 0

    @property
    def char_vocab_size(self) -> int:
        return self.char_tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.bpe_tokenizer.token_to_id(PAD_TOKEN)
