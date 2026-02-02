import os
import json
import pandas as pd
import sentencepiece as spm
import torch
from library.config import Config


class HybridTokenizer:
    """
    Implements a Heterogeneous Granularity Tokenizer.
    - Encoder: Character-level tokenizer with context anchoring.
    - Decoder: BPE (Byte Pair Encoding) tokenizer trained on target text.
    """

    def __init__(self):
        # Paths
        self.bpe_model_prefix = Config.BPE_MODEL_PREFIX
        self.bpe_model_path = f"{self.bpe_model_prefix}.model"
        self.bpe_vocab_path = f"{self.bpe_model_prefix}.vocab"
        self.char_vocab_path = os.path.join(Config.TOKENIZER_DIR, "char_vocab.json")

        # Encoder Special Tokens (Character Level)
        # 0 is reserved for PAD
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.SEP_TOKEN = "<SEP>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"

        self.enc_special_tokens = [
            self.PAD_TOKEN,  # ID 0
            self.UNK_TOKEN,  # ID 1
            self.SEP_TOKEN,  # ID 2
            self.START_TOKEN,  # ID 3
            self.END_TOKEN,  # ID 4
        ]

        self.char2id = {}
        self.id2char = {}

        # Decoder (BPE) Processor
        self.sp = spm.SentencePieceProcessor()

        # Decoder Special Token IDs (defined by SentencePiece training)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def train(self, load_cached_data: bool = True):
        """
        Trains the tokenizers or loads them from cache.

        Args:
            load_cached_data (bool): If True, attempts to load existing models.
        """
        # Ensure directory exists
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)

        # Check if artifacts exist
        artifacts_exist = os.path.exists(self.bpe_model_path) and os.path.exists(
            self.char_vocab_path
        )

        if load_cached_data and artifacts_exist:
            # Load from cache
            self._load_char_vocab()
            self.sp.Load(self.bpe_model_path)
            return

        # Train from scratch
        print("Training tokenizers from scratch...")

        # Load training data
        df = pd.read_csv(Config.TRAIN_DATA)
        # Ensure string types
        df["before"] = df["before"].fillna("").astype(str)
        df["after"] = df["after"].fillna("").astype(str)

        # 1. Train BPE for Decoder (Target Text)
        self._train_bpe(df["after"].tolist())

        # 2. Build Character Vocab for Encoder (Source Text)
        # We scan 'before' text. Contexts also come from 'before' distribution.
        self._build_char_vocab(df["before"].tolist())

        print("Tokenizer training complete.")

    def _train_bpe(self, texts):
        """
        Trains SentencePiece model on target texts.
        """
        # Save texts to a temporary file for SentencePiece
        temp_file = os.path.join(Config.TOKENIZER_DIR, "temp_bpe_train.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        # Train SentencePiece
        # user_defined_symbols are not strictly needed for decoder output usually,
        # but we ensure basic special tokens are handled.
        # pad_id=0, bos_id=1, eos_id=2, unk_id=3
        cmd = (
            f"--input={temp_file} "
            f"--model_prefix={self.bpe_model_prefix} "
            f"--vocab_size={Config.BPE_VOCAB_SIZE} "
            f"--character_coverage=1.0 "
            f"--model_type=bpe "
            f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
        )

        spm.SentencePieceTrainer.train(cmd)

        # Load the trained model
        self.sp.Load(self.bpe_model_path)

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def _build_char_vocab(self, texts):
        """
        Builds character-level vocabulary from source texts.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Assign IDs
        # Start with special tokens
        self.char2id = {token: idx for idx, token in enumerate(self.enc_special_tokens)}
        start_idx = len(self.enc_special_tokens)

        for idx, char in enumerate(sorted_chars):
            self.char2id[char] = start_idx + idx

        # Create reverse mapping
        self.id2char = {v: k for k, v in self.char2id.items()}

        # Save to JSON
        with open(self.char_vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def _load_char_vocab(self):
        """
        Loads character vocabulary from JSON.
        """
        with open(self.char_vocab_path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)
        self.id2char = {v: k for k, v in self.char2id.items()}

    def encode_src(self, prev_context: str, target_token: str, next_context: str):
        """
        Encodes source input into character IDs with context anchoring.
        Format: [Prev_Chars] <SEP> <START> [Target_Chars] <END> <SEP> [Next_Chars]

        Args:
            prev_context: Text preceding the target.
            target_token: The token to normalize.
            next_context: Text following the target.

        Returns:
            torch.Tensor: Tensor of character IDs (padded).
        """

        # Helper to map chars to IDs
        def text_to_ids(text):
            return [self.char2id.get(c, self.char2id[self.UNK_TOKEN]) for c in text]

        prev_ids = text_to_ids(prev_context)
        target_ids = text_to_ids(target_token)
        next_ids = text_to_ids(next_context)

        sep_id = self.char2id[self.SEP_TOKEN]
        start_id = self.char2id[self.START_TOKEN]
        end_id = self.char2id[self.END_TOKEN]

        # Construct sequence
        # [Prev] <SEP> <START> [Target] <END> <SEP> [Next]
        full_seq = (
            prev_ids + [sep_id, start_id] + target_ids + [end_id, sep_id] + next_ids
        )

        # Truncate if necessary (keep the center/target)
        # If too long, we trim from the ends (start of prev, end of next)
        max_len = Config.MAX_ENC_LEN
        if len(full_seq) > max_len:
            # Simple truncation: cut from end
            full_seq = full_seq[:max_len]

        # Padding
        pad_len = max_len - len(full_seq)
        if pad_len > 0:
            full_seq = full_seq + [self.char2id[self.PAD_TOKEN]] * pad_len

        return torch.tensor(full_seq, dtype=torch.long)

    def encode_tgt(self, target_text: str):
        """
        Encodes target text into BPE IDs.
        Adds <SOS> and <EOS>.

        Returns:
            torch.Tensor: Tensor of BPE IDs (padded).
        """
        ids = self.sp.EncodeAsIds(target_text)

        # Add SOS and EOS
        ids = [self.bos_id] + ids + [self.eos_id]

        # Truncate
        max_len = Config.MAX_DEC_LEN
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]

        # Pad
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        """
        Decodes a list/tensor of BPE IDs back to string.
        Removes special tokens.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Filter out special tokens for display/metric calculation
        # Keep only valid content tokens
        valid_ids = [i for i in ids if i not in [self.pad_id, self.bos_id, self.eos_id]]

        return self.sp.DecodeIds(valid_ids)

    def get_vocab_sizes(self):
        """
        Returns (encoder_vocab_size, decoder_vocab_size)
        """
        return len(self.char2id), len(self.sp)
