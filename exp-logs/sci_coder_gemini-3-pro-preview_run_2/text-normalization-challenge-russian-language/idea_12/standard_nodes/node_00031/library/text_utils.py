import os
import json
import pandas as pd
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from collections import Counter
from library.config import TOKENIZER_DIR, CACHE_DIR, ModelConfig

# =============================================================================
# CONSTANTS
# =============================================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"

# Special tokens list for Character Tokenizer
# Indices: PAD=0, UNK=1, SOS=2, EOS=3, SEP=4
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, SEP_TOKEN]

# =============================================================================
# CHARACTER TOKENIZER
# =============================================================================


class CharTokenizer:
    """
    Character-level tokenizer that handles special tokens and structural separators.
    """

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = 0

    def fit(self, texts, max_vocab_size=1000):
        """
        Builds vocabulary from a list of strings.
        """
        counter = Counter()
        for text in texts:
            # Count characters
            counter.update(str(text))

        # Initialize with special tokens
        self.char2id = {token: i for i, token in enumerate(SPECIAL_TOKENS)}

        # Add most common characters
        current_id = len(SPECIAL_TOKENS)
        # Reserve slots for special tokens
        limit = max_vocab_size - len(SPECIAL_TOKENS)

        for char, _ in counter.most_common(limit):
            if char not in self.char2id:
                self.char2id[char] = current_id
                current_id += 1

        self.id2char = {i: char for char, i in self.char2id.items()}
        self.vocab_size = len(self.char2id)

    def save(self, path):
        """Saves vocabulary to JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char2id": self.char2id}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Loads vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char2id = data["char2id"]
            # Keys in json are always strings, convert to int for id2char
            self.id2char = {int(v): k for k, v in self.char2id.items()}
            self.vocab_size = len(self.char2id)

    def encode(self, text, max_len=None):
        """
        Encodes text to IDs. Handles <SEP> as a special atomic token.
        """
        ids = []
        # Split by the separator token to preserve it as a single ID
        parts = text.split(SEP_TOKEN)

        sep_id = self.char2id[SEP_TOKEN]
        unk_id = self.char2id[UNK_TOKEN]

        for i, part in enumerate(parts):
            # Encode characters in the part
            for char in part:
                ids.append(self.char2id.get(char, unk_id))

            # Re-insert SEP token between parts
            if i < len(parts) - 1:
                ids.append(sep_id)

        if max_len:
            ids = ids[:max_len]
            # Pad if necessary (optional here, usually done in collate)
            if len(ids) < max_len:
                pad_id = self.char2id[PAD_TOKEN]
                ids += [pad_id] * (max_len - len(ids))

        return ids

    def decode(self, ids):
        """Decodes IDs back to string."""
        chars = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if i == self.char2id[PAD_TOKEN]:
                continue
            chars.append(self.id2char.get(i, UNK_TOKEN))
        return "".join(chars)


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================


def build_char_vocab(df, vocab_size=300, load_cached_data=True):
    """
    Builds or loads the character tokenizer.
    """
    vocab_path = os.path.join(TOKENIZER_DIR, "char_vocab.json")
    tokenizer = CharTokenizer()

    if load_cached_data and os.path.exists(vocab_path):
        print(f"Loading character vocabulary from {vocab_path}")
        tokenizer.load(vocab_path)
    else:
        print("Building character vocabulary...")
        # Use 'before' column for vocabulary
        texts = df["before"].fillna("").astype(str).tolist()
        tokenizer.fit(texts, max_vocab_size=vocab_size)
        tokenizer.save(vocab_path)
        print(
            f"Character vocabulary saved to {vocab_path} (Size: {tokenizer.vocab_size})"
        )

    return tokenizer


def train_bpe_tokenizer(df, vocab_size=4000, load_cached_data=True):
    """
    Trains or loads a SentencePiece BPE tokenizer for the target text.
    """
    model_prefix = os.path.join(TOKENIZER_DIR, "bpe_ru_target")
    model_path = model_prefix + ".model"

    if load_cached_data and os.path.exists(model_path):
        print(f"Loading BPE tokenizer from {model_path}")
    else:
        print("Training BPE tokenizer...")
        text_path = os.path.join(TOKENIZER_DIR, "train_text_for_bpe.txt")

        # Extract non-empty target texts
        texts = df["after"].fillna("").astype(str)
        texts = texts[texts.str.len() > 0]

        with open(text_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n")

        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="[PAD]",
            unk_piece="[UNK]",
            bos_piece="[BOS]",
            eos_piece="[EOS]",
        )
        print(f"BPE tokenizer trained and saved to {model_prefix}.model")

        if os.path.exists(text_path):
            os.remove(text_path)

    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def format_context_window(df, context_window=2):
    """
    Constructs input strings with context: Prev_2 Prev_1 <SEP> Target <SEP> Next_1 Next_2
    Returns a numpy array of strings.
    """
    # Create a copy to avoid SettingWithCopy warnings on the original DF
    df_proc = df.copy()

    # Ensure string type
    df_proc["token_str"] = df_proc["before"].fillna("").astype(str)

    # Initialize series for concatenation
    # Using pandas Series string concatenation is efficient enough and readable
    full_series = pd.Series("", index=df_proc.index)

    # --- Left Context ---
    for i in range(context_window, 0, -1):
        # Shift i positions down
        shifted_str = df_proc["token_str"].shift(i).fillna("")
        shifted_sent = df_proc["sentence_id"].shift(i)

        # Mask boundaries (if sentence_id changed, context is invalid)
        mask = shifted_sent == df_proc["sentence_id"]
        valid_str = shifted_str.where(mask, "")

        # Append to series: "word "
        full_series = full_series + valid_str + " "

    # --- Center (Target) ---
    # Add separators: "<SEP> target <SEP>"
    full_series = full_series + SEP_TOKEN + " " + df_proc["token_str"] + " " + SEP_TOKEN

    # --- Right Context ---
    for i in range(1, context_window + 1):
        # Shift i positions up
        shifted_str = df_proc["token_str"].shift(-i).fillna("")
        shifted_sent = df_proc["sentence_id"].shift(-i)

        mask = shifted_sent == df_proc["sentence_id"]
        valid_str = shifted_str.where(mask, "")

        # Append to series: " word"
        full_series = full_series + " " + valid_str

    return full_series.values


def preprocess_dataset(df, split_name, context_window=2, load_cached_data=True):
    """
    Applies context formatting and caches the result to Parquet.
    Returns a DataFrame with 'input_text' and 'target_text' (if available).
    """
    cache_path = os.path.join(CACHE_DIR, f"processed_{split_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed {split_name} data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing {split_name} data (Context Window: {context_window})...")

    # Format Input
    input_texts = format_context_window(df, context_window=context_window)

    # Create Result DataFrame
    result_df = pd.DataFrame(
        {
            "id": (
                df["id"]
                if "id" in df.columns
                else df["sentence_id"].astype(str) + "_" + df["token_id"].astype(str)
            ),
            "input_text": input_texts,
        }
    )

    # Format Target (if exists)
    if "after" in df.columns:
        result_df["target_text"] = df["after"].fillna("").astype(str)

    # Save
    print(f"Saving processed data to {cache_path}")
    result_df.to_parquet(cache_path, index=False)

    return result_df


# =============================================================================
# PYTORCH DATASET
# =============================================================================


class TextDataset(Dataset):
    """
    PyTorch Dataset for the Hybrid Cascade Model.
    """

    def __init__(
        self,
        df,
        char_tokenizer,
        bpe_tokenizer,
        max_enc_len=128,
        max_dec_len=128,
        mode="train",
    ):
        self.df = df
        self.char_tokenizer = char_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.mode = mode

        # Pre-check columns
        self.has_target = "target_text" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Encoder Input (Characters)
        input_text = row["input_text"]
        enc_ids = self.char_tokenizer.encode(input_text, max_len=self.max_enc_len)

        item = {
            "encoder_input": torch.tensor(enc_ids, dtype=torch.long),
            "id": row["id"],
        }

        # Decoder Target (BPE) - Only for training/validation
        if self.has_target:
            target_text = row["target_text"]
            # Add EOS/BOS handled by SentencePiece or manually?
            # SP add_bos=True, add_eos=True
            dec_ids = self.bpe_tokenizer.EncodeAsIds(target_text)

            # Add SOS/EOS explicitly if not in SP defaults or to match Transformer expectations
            # Config says: bos_id=2, eos_id=3
            dec_ids = [2] + dec_ids + [3]

            # Truncate/Pad
            if len(dec_ids) > self.max_dec_len:
                dec_ids = dec_ids[: self.max_dec_len]
                dec_ids[-1] = 3  # Ensure EOS is present

            # Padding is handled in collate_fn usually, but we can return list or tensor
            item["decoder_target"] = torch.tensor(dec_ids, dtype=torch.long)

        return item
