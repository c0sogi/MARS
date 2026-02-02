import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from library.config import ProjectConfig, DataConfig, TrainingConfig, set_seed

# Set seed for reproducibility
set_seed(TrainingConfig.SEED)


class Vocabulary:
    """
    Handles mapping between tokens (strings) and indices (integers).
    Supports saving/loading from Parquet.
    """

    def __init__(self, specials=None):
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

        for i, token in enumerate(self.specials):
            self.stoi[token] = i
            self.itos[i] = token

    def __len__(self):
        return len(self.stoi)

    def add_token(self, token):
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

    def lookup_indices(self, tokens, unk_token=None):
        indices = []
        unk_idx = self.stoi.get(unk_token) if unk_token else None

        for token in tokens:
            idx = self.stoi.get(token, unk_idx)
            if idx is not None:
                indices.append(idx)
        return indices

    def lookup_token(self, idx):
        return self.itos.get(idx, None)

    def save(self, path):
        # Save as parquet: token, index
        data = [{"token": token, "index": idx} for token, idx in self.stoi.items()]
        df = pd.DataFrame(data)
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        vocab = cls()
        # Reset init state to avoid duplication if specials were default
        vocab.stoi = {}
        vocab.itos = {}
        vocab.specials = []

        for _, row in df.iterrows():
            token = row["token"]
            idx = row["index"]
            vocab.stoi[token] = idx
            vocab.itos[idx] = token
        return vocab


def load_dataset_raw(split="train"):
    """
    Loads the raw dataframe from metadata.
    """
    if split == "train":
        path = ProjectConfig.TRAIN_PATH
    elif split == "val":
        path = ProjectConfig.VAL_PATH
    elif split == "test":
        path = ProjectConfig.TEST_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    # Load with string types to prevent pandas inference issues
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    # Debug mode subsampling
    if ProjectConfig.DEBUG and split == "train":
        df = df.head(ProjectConfig.DEBUG_SIZE)

    return df


def build_vocabularies(load_cached_data=True):
    """
    Builds or loads Word, Character, and Class vocabularies.
    """
    word_vocab_path = ProjectConfig.VOCAB_WORDS_PATH
    char_vocab_path = ProjectConfig.VOCAB_CHARS_PATH
    class_vocab_path = ProjectConfig.VOCAB_CLASSES_PATH

    # Try loading
    if load_cached_data:
        vocab_words = Vocabulary.load(word_vocab_path)
        vocab_chars = Vocabulary.load(char_vocab_path)
        vocab_classes = Vocabulary.load(class_vocab_path)

        if vocab_words and vocab_chars and vocab_classes:
            return vocab_words, vocab_chars, vocab_classes

    # Build from scratch
    df_train = load_dataset_raw("train")

    # 1. Word Vocabulary
    # Specials: PAD, UNK
    vocab_words = Vocabulary(specials=[DataConfig.PAD_TOKEN, DataConfig.UNK_TOKEN])
    word_counts = Counter(df_train["before"].tolist())

    # Sort by frequency then alpha
    sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

    count = 0
    for word, freq in sorted_words:
        if freq >= DataConfig.MIN_WORD_FREQ:
            vocab_words.add_token(word)
            count += 1
            if count >= DataConfig.MAX_WORD_VOCAB_SIZE:
                break

    # 2. Character Vocabulary
    # Specials: PAD, UNK, SOS, EOS (SOS/EOS needed for Seq2Seq)
    vocab_chars = Vocabulary(
        specials=[
            DataConfig.PAD_TOKEN,
            DataConfig.UNK_TOKEN,
            DataConfig.SOS_TOKEN,
            DataConfig.EOS_TOKEN,
        ]
    )

    # Collect all chars from 'before' and 'after'
    unique_chars = set()
    for text in df_train["before"].unique():
        unique_chars.update(str(text))
    for text in df_train["after"].unique():
        unique_chars.update(str(text))

    for char in sorted(list(unique_chars)):
        vocab_chars.add_token(char)

    # 3. Class Vocabulary
    # No specials needed usually, but PAD might be useful if batching labels (though we usually don't pad labels here)
    vocab_classes = Vocabulary()
    unique_classes = sorted(df_train["class"].unique().tolist())
    for cls_name in unique_classes:
        vocab_classes.add_token(cls_name)

    # Save
    vocab_words.save(word_vocab_path)
    vocab_chars.save(char_vocab_path)
    vocab_classes.save(class_vocab_path)

    return vocab_words, vocab_chars, vocab_classes


def build_knowledge_base(load_cached_data=True):
    """
    Builds a deterministic map: (before_token, class) -> after_token.
    """
    kb_path = ProjectConfig.KNOWLEDGE_BASE_PATH

    if load_cached_data and os.path.exists(kb_path):
        df_kb = pd.read_parquet(kb_path)
        # Convert to dictionary for O(1) lookup
        # Key: "token|class", Value: "normalized"
        kb_dict = {}
        for _, row in df_kb.iterrows():
            key = (row["before"], row["class"])
            kb_dict[key] = row["after"]
        return kb_dict

    df_train = load_dataset_raw("train")

    # Group by before+class and find the most frequent normalization
    # In this dataset, ambiguity for (token, class) is rare, but we take mode to be safe.
    # To save time/memory, we can just drop duplicates keeping the last or first.
    # Let's count to be precise.

    # Create a composite key to group
    # We only care about pairs that actually exist
    kb_df = df_train[["before", "class", "after"]].copy()

    # Drop exact duplicates first
    kb_df = kb_df.drop_duplicates()

    # If there are conflicts (same before+class, different after), we resolve them.
    # Since we dropped duplicates, we can check if (before, class) is unique.
    # For this implementation, we simply take the last occurrence as the "truth"
    # assuming the dataset is generally consistent.
    kb_df = kb_df.drop_duplicates(subset=["before", "class"], keep="last")

    # Save
    kb_df.to_parquet(kb_path, index=False)

    # Return dict
    kb_dict = {}
    for _, row in kb_df.iterrows():
        key = (row["before"], row["class"])
        kb_dict[key] = row["after"]

    return kb_dict


class TaggerDataset(Dataset):
    """
    Dataset for the Multi-Granularity Tagger.
    Returns:
        - word_id: int
        - char_ids: List[int]
        - class_id: int (or -1 if test)
        - raw_text: str
        - id: str (sentence_token id)
    """

    def __init__(self, df, vocab_words, vocab_chars, vocab_classes, is_test=False):
        self.df = df
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.is_test = is_test

        # Pre-fetch columns to avoid overhead in __getitem__
        self.before_tokens = self.df["before"].astype(str).tolist()
        self.ids = self.df["id"].tolist()

        if not self.is_test:
            self.classes = self.df["class"].tolist()
        else:
            self.classes = None

        # Cache special indices
        self.word_unk_idx = self.vocab_words.stoi[DataConfig.UNK_TOKEN]
        self.char_unk_idx = self.vocab_chars.stoi[DataConfig.UNK_TOKEN]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_text = self.before_tokens[idx]
        row_id = self.ids[idx]

        # Word ID
        word_id = self.vocab_words.stoi.get(raw_text, self.word_unk_idx)

        # Char IDs
        # Truncate to MAX_TOKEN_LEN
        char_ids = self.vocab_chars.lookup_indices(
            list(raw_text), unk_token=DataConfig.UNK_TOKEN
        )
        if len(char_ids) > DataConfig.MAX_TOKEN_LEN:
            char_ids = char_ids[: DataConfig.MAX_TOKEN_LEN]

        # Class ID
        class_id = -1
        if not self.is_test:
            cls_name = self.classes[idx]
            class_id = self.vocab_classes.stoi.get(cls_name, -1)

        return {
            "word_id": torch.tensor(word_id, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "raw_text": raw_text,
            "id": row_id,
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for the Seq2Seq Fallback Model.
    Filters data to only include changed tokens (before != after).
    Returns:
        - src_char_ids: List[int] (with SOS/EOS)
        - tgt_char_ids: List[int] (with SOS/EOS)
        - class_id: int
    """

    def __init__(self, df, vocab_chars, vocab_classes):
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes

        # Filter for changed tokens only
        # We assume df has 'before', 'after', 'class'
        mask = df["before"] != df["after"]
        self.df = df[mask].reset_index(drop=True)

        self.before_tokens = self.df["before"].astype(str).tolist()
        self.after_tokens = self.df["after"].astype(str).tolist()
        self.classes = self.df["class"].tolist()

        self.sos_idx = self.vocab_chars.stoi[DataConfig.SOS_TOKEN]
        self.eos_idx = self.vocab_chars.stoi[DataConfig.EOS_TOKEN]
        self.unk_idx = self.vocab_chars.stoi[DataConfig.UNK_TOKEN]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.before_tokens[idx]
        tgt_text = self.after_tokens[idx]
        cls_name = self.classes[idx]

        # Convert to indices
        src_indices = self.vocab_chars.lookup_indices(
            list(src_text), unk_token=DataConfig.UNK_TOKEN
        )
        tgt_indices = self.vocab_chars.lookup_indices(
            list(tgt_text), unk_token=DataConfig.UNK_TOKEN
        )

        # Add SOS/EOS
        # Source: [chars] (Encoder usually doesn't need SOS/EOS strictly, but good practice to handle length)
        # Target: [SOS, chars, EOS]

        # For this specific architecture (Char Encoder -> Char Decoder):
        # Encoder input: raw chars
        # Decoder input: SOS + target chars
        # Decoder target: target chars + EOS

        # We will return full sequences and handle shifting in the model or collate
        src_seq = src_indices  # Encoder input
        tgt_seq = [self.sos_idx] + tgt_indices + [self.eos_idx]

        class_id = self.vocab_classes.stoi.get(cls_name, 0)  # Default to 0 if unknown

        return {
            "src_char_ids": torch.tensor(src_seq, dtype=torch.long),
            "tgt_char_ids": torch.tensor(tgt_seq, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }


def collate_fn_tagger(batch):
    """
    Collate function for TaggerDataset.
    Pads char_ids. Stacks word_ids and class_ids.
    """
    word_ids = torch.stack([item["word_id"] for item in batch])
    class_ids = torch.stack([item["class_id"] for item in batch])
    ids = [item["id"] for item in batch]
    raw_texts = [item["raw_text"] for item in batch]

    # Pad char sequences
    # char_ids is a list of 1D tensors
    char_ids_list = [item["char_ids"] for item in batch]
    # Pad with PAD_TOKEN index (usually 0, but check vocab)
    # We assume PAD is at index 0 based on Vocabulary init order
    char_ids_padded = pad_sequence(char_ids_list, batch_first=True, padding_value=0)

    return {
        "word_ids": word_ids,
        "char_ids": char_ids_padded,
        "class_ids": class_ids,
        "ids": ids,
        "raw_texts": raw_texts,
    }


def collate_fn_seq2seq(batch):
    """
    Collate function for Seq2SeqDataset.
    Pads src and tgt sequences.
    """
    class_ids = torch.stack([item["class_id"] for item in batch])

    src_list = [item["src_char_ids"] for item in batch]
    tgt_list = [item["tgt_char_ids"] for item in batch]

    # Pad with 0 (PAD token)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0)

    return {
        "src_char_ids": src_padded,
        "tgt_char_ids": tgt_padded,
        "class_ids": class_ids,
    }
