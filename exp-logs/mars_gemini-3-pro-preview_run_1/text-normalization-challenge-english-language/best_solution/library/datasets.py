import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from library.config import Config


class TaggerDataset(Dataset):
    """
    Dataset for the Morphologically-Aware Bi-LSTM Tagger.
    Operates on sentence-level data.
    """

    def __init__(self, df, vocab_words, vocab_chars, vocab_classes, split="train"):
        """
        Args:
            df (pd.DataFrame): Grouped dataframe containing 'before', 'class' (optional), etc.
            vocab_words (Vocabulary): Vocabulary for word-level embeddings.
            vocab_chars (Vocabulary): Vocabulary for character-level embeddings.
            vocab_classes (Vocabulary): Vocabulary for target classes.
            split (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.split = split

        # Pre-fetch indices for special tokens to speed up __getitem__
        self.unk_word_idx = self.vocab_words.stoi.get(Config.UNK_TOKEN, 0)
        self.unk_char_idx = self.vocab_chars.stoi.get(Config.UNK_TOKEN, 0)
        self.pad_char_idx = self.vocab_chars.stoi.get(Config.PAD_TOKEN, 0)

        # Determine if we have labels
        self.has_labels = (split != "test") and ("class" in df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Extract raw tokens
        # The grouped dataframe contains lists of strings
        tokens = row["before"]

        # Truncate sequence length
        if len(tokens) > Config.MAX_SEQ_LEN:
            tokens = tokens[: Config.MAX_SEQ_LEN]

        # 2. Word IDs
        word_ids = self.vocab_words.lookup_indices(tokens, unk_token=Config.UNK_TOKEN)

        # 3. Character IDs (List of Lists)
        # For each token, get char indices, truncate to MAX_CHAR_LEN
        char_ids_list = []
        for token in tokens:
            # Convert token to chars
            chars = list(str(token))
            # Lookup
            c_ids = self.vocab_chars.lookup_indices(chars, unk_token=Config.UNK_TOKEN)
            # Truncate
            if len(c_ids) > Config.MAX_CHAR_LEN:
                c_ids = c_ids[: Config.MAX_CHAR_LEN]
            char_ids_list.append(c_ids)

        # 4. Class IDs (Target)
        class_ids = []
        if self.has_labels:
            raw_classes = row["class"]
            if len(raw_classes) > Config.MAX_SEQ_LEN:
                raw_classes = raw_classes[: Config.MAX_SEQ_LEN]
            class_ids = self.vocab_classes.lookup_indices(raw_classes, unk_token=None)
            # Note: Classes shouldn't be UNK usually, but if so, lookup_indices handles None safely if coded so.
            # However, Vocabulary.lookup_indices returns None for missing if unk_token is None.
            # We map None to a safe index (e.g. PLAIN or 0) or handle it.
            # Assuming vocab_classes covers all train classes. For safety, map None to 0.
            class_ids = [c if c is not None else 0 for c in class_ids]
        else:
            # Dummy classes for test set
            class_ids = [0] * len(tokens)

        # 5. Metadata for submission reconstruction
        # id is a list of strings like "12_0", "12_1"
        row_ids = row["id"]
        if len(row_ids) > Config.MAX_SEQ_LEN:
            row_ids = row_ids[: Config.MAX_SEQ_LEN]

        return {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "char_ids": char_ids_list,  # List of lists, will be padded in collate
            "class_ids": torch.tensor(class_ids, dtype=torch.long),
            "sentence_len": len(tokens),
            "row_ids": row_ids,
        }


def tagger_collate_fn(batch):
    """
    Collate function for TaggerDataset.
    Pads word_ids and class_ids (1D sequences).
    Pads char_ids (2D sequences -> 3D tensor).
    """
    # Extract fields
    batch_word_ids = [item["word_ids"] for item in batch]
    batch_class_ids = [item["class_ids"] for item in batch]
    batch_char_ids_list = [
        item["char_ids"] for item in batch
    ]  # List(Batch) of List(Seq) of List(Char)
    batch_row_ids = [item["row_ids"] for item in batch]

    # Pad 1D sequences (Batch, Seq)
    # Use 0 (assuming PAD_TOKEN is 0) for padding
    padded_word_ids = pad_sequence(batch_word_ids, batch_first=True, padding_value=0)
    padded_class_ids = pad_sequence(
        batch_class_ids, batch_first=True, padding_value=0
    )  # -100 if using ignore_index in loss

    # Pad 2D sequences (Batch, Seq, Char)
    # Dimensions
    batch_size = len(batch)
    max_seq_len = padded_word_ids.size(1)
    max_char_len = Config.MAX_CHAR_LEN

    # Initialize tensor with padding index (0)
    padded_char_ids = torch.zeros(
        (batch_size, max_seq_len, max_char_len), dtype=torch.long
    )

    for i, sent_chars in enumerate(batch_char_ids_list):
        # sent_chars is a list of lists of char indices
        for j, token_chars in enumerate(sent_chars):
            if j >= max_seq_len:
                break

            length = min(len(token_chars), max_char_len)
            if length > 0:
                padded_char_ids[i, j, :length] = torch.tensor(
                    token_chars[:length], dtype=torch.long
                )

    return {
        "word_ids": padded_word_ids,
        "char_ids": padded_char_ids,
        "class_ids": padded_class_ids,
        "row_ids": batch_row_ids,  # List of lists
    }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Character-Level Seq2Seq Fallback Model.
    Operates on token-level data.
    Filters data to only include tokens that change (before != after) for training.
    """

    def __init__(self, df, vocab_chars, split="train"):
        """
        Args:
            df (pd.DataFrame): Raw dataframe (not grouped).
            vocab_chars (Vocabulary): Character vocabulary.
            split (str): 'train', 'val', or 'test'.
        """
        self.vocab_chars = vocab_chars
        self.split = split

        # Filter data for training/validation
        if split in ["train", "val"]:
            # Ensure we have required columns
            if "after" in df.columns:
                # Filter: Keep only where before != after
                # We assume df has string types already handled by loader
                mask = df["before"] != df["after"]
                self.df = df[mask].reset_index(drop=True)
            else:
                # If 'after' is missing in val (unlikely given metadata), keep all
                self.df = df.reset_index(drop=True)
        else:
            # For test or inference usage, we might pass specific tokens
            # But typically this dataset is for training.
            # If used for inference, we don't filter.
            self.df = df.reset_index(drop=True)

        # Special tokens
        self.sos_idx = self.vocab_chars.stoi[Config.SOS_TOKEN]
        self.eos_idx = self.vocab_chars.stoi[Config.EOS_TOKEN]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src_text = row["before"]

        # Source IDs
        # No truncation for source usually, or loose truncation
        src_ids = self.vocab_chars.lookup_indices(
            list(src_text), unk_token=Config.UNK_TOKEN
        )

        # Target IDs
        if self.split != "test" and "after" in row:
            tgt_text = row["after"]
            raw_tgt_ids = self.vocab_chars.lookup_indices(
                list(tgt_text), unk_token=Config.UNK_TOKEN
            )
            # Add SOS and EOS
            tgt_ids = [self.sos_idx] + raw_tgt_ids + [self.eos_idx]

            # Truncate Target
            if len(tgt_ids) > Config.SEQ2SEQ_MAX_LEN:
                tgt_ids = tgt_ids[: Config.SEQ2SEQ_MAX_LEN - 1] + [self.eos_idx]
        else:
            tgt_ids = []  # No target for test

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "raw_before": src_text,
        }


def seq2seq_collate_fn(batch):
    """
    Collate function for Seq2SeqDataset.
    Pads src_ids and tgt_ids.
    """
    batch_src_ids = [item["src_ids"] for item in batch]
    batch_tgt_ids = [item["tgt_ids"] for item in batch]
    batch_raw = [item["raw_before"] for item in batch]

    # Pad sequences
    # Padding value 0 (PAD_TOKEN)
    padded_src = pad_sequence(batch_src_ids, batch_first=True, padding_value=0)

    if len(batch_tgt_ids[0]) > 0:
        padded_tgt = pad_sequence(batch_tgt_ids, batch_first=True, padding_value=0)
    else:
        padded_tgt = None

    return {"src_ids": padded_src, "tgt_ids": padded_tgt, "raw_before": batch_raw}
