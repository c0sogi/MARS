import os
import json
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
from library.config import Config
from library.utils import setup_logger


class CharTokenizer:
    """
    Character-level tokenizer for the source input.
    Handles special tokens: [PAD], [UNK], [SEP].
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SEP_TOKEN = "[SEP]"

    PAD_ID = 0
    UNK_ID = 1
    SEP_ID = 2

    def __init__(self):
        self.char2id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.SEP_TOKEN: self.SEP_ID,
        }
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = 3

    def fit(self, texts: List[str]):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        for char in sorted_chars:
            if char not in self.char2id:
                self.char2id[char] = self.vocab_size
                self.id2char[self.vocab_size] = char
                self.vocab_size += 1

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """
        Encodes a string into a list of character IDs.
        """
        text = str(text)
        ids = [self.char2id.get(c, self.UNK_ID) for c in text]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)
        self.id2char = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id)


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for the Char-to-Subword Transformer.
    Input: [Prev] <SEP> [Target] <SEP> [Next] (Character Tokenized)
    Output: Normalized Text (BPE Tokenized)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        bpe_model_path: str,
        context_source_path: Optional[str] = None,
        char_vocab_path: Optional[str] = None,
        max_input_len: int = Config.MAX_INPUT_LEN,
        mode: str = "train",
    ):
        """
        Args:
            data: DataFrame containing samples.
            bpe_model_path: Path to the trained SentencePiece model.
            context_source_path: Path to the full CSV (e.g., metadata/train.csv) to recover context.
                                 If None, assumes 'data' is the full sequence and generates context directly.
            char_vocab_path: Path to save/load character vocabulary.
            max_input_len: Maximum length for source sequence.
            mode: 'train' (returns src, tgt) or 'inference' (returns src).
        """
        self.logger = setup_logger("NormalizationDataset")
        self.data = data.copy()
        self.max_input_len = max_input_len
        self.mode = mode

        # Load BPE Tokenizer
        self.sp = spm.SentencePieceProcessor()
        if not os.path.exists(bpe_model_path):
            # Fallback if specific model file not found, try adding .model extension
            if os.path.exists(bpe_model_path + ".model"):
                bpe_model_path = bpe_model_path + ".model"
            else:
                raise FileNotFoundError(f"BPE model not found at {bpe_model_path}")
        self.sp.load(bpe_model_path)
        self.pad_idx_tgt = self.sp.pad_id()  # 0
        self.bos_idx = self.sp.bos_id()  # 2
        self.eos_idx = self.sp.eos_id()  # 3

        # Prepare Context (Prev/Next)
        self._prepare_context(context_source_path)

        # Initialize Char Tokenizer
        self.char_tokenizer = CharTokenizer()
        self._setup_char_vocab(char_vocab_path, context_source_path)

        self.pad_idx_src = self.char_tokenizer.PAD_ID

        self.logger.info(
            f"Dataset initialized. Mode: {mode}, Samples: {len(self.data)}"
        )

    def _prepare_context(self, context_source_path: Optional[str]):
        """
        Generates 'prev' and 'next' columns.
        If context_source_path is provided, loads full data to generate context and merges it.
        Otherwise, generates context directly from self.data (assuming it's a full sequence).
        """
        PAD_START = "^"
        PAD_END = "$"

        if context_source_path and os.path.exists(context_source_path):
            self.logger.info(f"Recovering context from {context_source_path}...")
            # Load full source to get correct context
            df_full = pd.read_csv(context_source_path)
            df_full["before"] = df_full["before"].fillna("").astype(str)

            # Generate context on full data
            # Shift within groups is slow, so we use global shift with mask
            df_full["prev"] = df_full["before"].shift(1).fillna(PAD_START)
            df_full["next"] = df_full["before"].shift(-1).fillna(PAD_END)

            # Handle boundaries
            sent_ids = df_full["sentence_id"].values
            # Start of sentence: current != prev
            is_start = np.concatenate(([True], sent_ids[1:] != sent_ids[:-1]))
            df_full.loc[is_start, "prev"] = PAD_START

            # End of sentence: current != next
            is_end = np.concatenate((sent_ids[:-1] != sent_ids[1:], [True]))
            df_full.loc[is_end, "next"] = PAD_END

            # Merge context into the dataset
            # We need sentence_id and token_id to match
            cols_to_merge = ["sentence_id", "token_id", "prev", "next"]

            # Ensure types match for merge
            self.data["sentence_id"] = self.data["sentence_id"].astype(str)
            self.data["token_id"] = self.data["token_id"].astype(int)
            df_full["sentence_id"] = df_full["sentence_id"].astype(str)
            df_full["token_id"] = df_full["token_id"].astype(int)

            self.data = pd.merge(
                self.data,
                df_full[cols_to_merge],
                on=["sentence_id", "token_id"],
                how="left",
            )

            # Fill NaNs if any merge failed (shouldn't happen if source is correct)
            self.data["prev"] = self.data["prev"].fillna(PAD_START)
            self.data["next"] = self.data["next"].fillna(PAD_END)

        else:
            self.logger.info("Generating context from input dataframe directly...")
            # Assume self.data is sorted and complete (e.g. test set)
            df = self.data.copy()
            df["before"] = df["before"].fillna("").astype(str)

            df["prev"] = df["before"].shift(1).fillna(PAD_START)
            df["next"] = df["before"].shift(-1).fillna(PAD_END)

            if "sentence_id" in df.columns:
                sent_ids = df["sentence_id"].values
                is_start = np.concatenate(([True], sent_ids[1:] != sent_ids[:-1]))
                df.loc[is_start, "prev"] = PAD_START
                is_end = np.concatenate((sent_ids[:-1] != sent_ids[1:], [True]))
                df.loc[is_end, "next"] = PAD_END

            self.data = df

    def _setup_char_vocab(
        self, vocab_path: Optional[str], context_source_path: Optional[str]
    ):
        """
        Loads or builds the character vocabulary.
        """
        # Try to load first
        if vocab_path and os.path.exists(vocab_path):
            self.logger.info(f"Loading char vocab from {vocab_path}")
            self.char_tokenizer.load(vocab_path)
            return

        # If not found, build it
        self.logger.info("Building char vocab...")
        texts_to_scan = []

        # Use context source for comprehensive vocab if available
        if context_source_path and os.path.exists(context_source_path):
            df_source = pd.read_csv(context_source_path)
            texts_to_scan.extend(df_source["before"].dropna().astype(str).tolist())
        else:
            # Fallback to current data
            texts_to_scan.extend(self.data["before"].dropna().astype(str).tolist())
            texts_to_scan.extend(self.data["prev"].dropna().astype(str).tolist())
            texts_to_scan.extend(self.data["next"].dropna().astype(str).tolist())

        self.char_tokenizer.fit(texts_to_scan)
        self.logger.info(f"Vocab built. Size: {self.char_tokenizer.vocab_size}")

        if vocab_path:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            self.char_tokenizer.save(vocab_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        prev_tok = str(row["prev"])
        curr_tok = str(row["before"])
        next_tok = str(row["next"])

        # Construct Source String: [Prev] <SEP> [Current] <SEP> [Next]
        # We process them as a single sequence of chars with separators
        # Or we can tokenize separately and concat IDs.
        # Tokenizing separately allows us to insert the special SEP ID cleanly.

        prev_ids = self.char_tokenizer.encode(prev_tok)
        curr_ids = self.char_tokenizer.encode(curr_tok)
        next_ids = self.char_tokenizer.encode(next_tok)
        sep_id = [self.char_tokenizer.SEP_ID]

        src_ids = prev_ids + sep_id + curr_ids + sep_id + next_ids

        # Truncate if necessary (keeping the center is usually best, but simple truncation is okay)
        if len(src_ids) > self.max_input_len:
            src_ids = src_ids[: self.max_input_len]

        src_tensor = torch.tensor(src_ids, dtype=torch.long)

        if self.mode == "train":
            target_text = str(row["after"])
            # Encode target with BPE
            # add_bos=True, add_eos=True
            tgt_ids = self.sp.encode(target_text, add_bos=True, add_eos=True)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
            return src_tensor, tgt_tensor
        else:
            return src_tensor


def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding.
    """
    # Check if we have targets (tuples) or just source (tensors)
    if isinstance(batch[0], tuple):
        src_batch, tgt_batch = zip(*batch)
        has_target = True
    else:
        src_batch = batch
        has_target = False

    # Pad Source
    # padding_value must match char_tokenizer.PAD_ID which is 0
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )

    if has_target:
        # Pad Target
        # padding_value must match sp.pad_id() which is 0
        tgt_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, batch_first=True, padding_value=0
        )
        return src_padded, tgt_padded
    else:
        return src_padded
