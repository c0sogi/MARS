import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Tuple, Dict, Optional
from library.config import Config
from library.utils import set_seed


# --------------------------------------------------------------------------
# Vocabulary Management
# --------------------------------------------------------------------------
class Vocabulary:
    def __init__(self, name: str, specials: List[str] = None):
        self.name = name
        self.stoi = {}
        self.itos = {}
        self.specials = specials if specials else []

    def build(self, tokens: List[str], min_freq: int = 1, max_size: int = None):
        """Builds vocabulary from a list of tokens."""
        counter = Counter(tokens)

        # Start with special tokens
        self.itos = {i: s for i, s in enumerate(self.specials)}
        self.stoi = {s: i for i, s in self.itos.items()}
        idx = len(self.specials)

        # Sort by frequency then alphabetically
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for token, freq in sorted_tokens:
            if freq < min_freq:
                break
            if max_size and len(self.stoi) >= max_size:
                break

            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1

    def __len__(self):
        return len(self.stoi)

    def lookup_indices(self, tokens: List[str], unk_token: str = None) -> List[int]:
        unk_idx = self.stoi.get(unk_token) if unk_token else None
        return [self.stoi.get(t, unk_idx) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        return self.itos.get(idx, "")

    def save(self, path: str):
        """Saves vocabulary to a parquet file."""
        data = [{"id": i, "token": t} for i, t in self.itos.items()]
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def load(self, path: str):
        """Loads vocabulary from a parquet file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        df = pd.read_parquet(path)
        self.stoi = {row["token"]: row["id"] for _, row in df.iterrows()}
        self.itos = {row["id"]: row["token"] for _, row in df.iterrows()}


# --------------------------------------------------------------------------
# Dataset Classes
# --------------------------------------------------------------------------
class TaggerDataset(Dataset):
    """
    Dataset for Bi-LSTM-CRF Tagger.
    Returns:
        - token_ids: (seq_len)
        - char_ids: (seq_len, char_len)
        - label_ids: (seq_len)
        - mask: (seq_len)
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        vocab_tokens: Vocabulary,
        vocab_chars: Vocabulary,
        vocab_classes: Vocabulary,
        is_test: bool = False,
    ):
        self.data = data_df
        self.vocab_tokens = vocab_tokens
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes
        self.is_test = is_test

        self.pad_token_id = vocab_tokens.stoi[Config.PAD_TOKEN]
        self.unk_token_id = vocab_tokens.stoi[Config.UNK_TOKEN]
        self.pad_char_id = vocab_chars.stoi[Config.PAD_TOKEN]
        self.unk_char_id = vocab_chars.stoi[Config.UNK_TOKEN]

        if not self.is_test:
            self.pad_class_id = vocab_classes.stoi[Config.PAD_TOKEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Raw tokens
        raw_tokens = row["before"]  # This is a list of strings
        if not isinstance(raw_tokens, (list, np.ndarray)):
            # Fallback if loading failed to interpret list
            raw_tokens = [str(raw_tokens)]

        seq_len = min(len(raw_tokens), Config.MAX_SEQ_LEN)

        # 1. Token IDs
        token_ids = [
            self.vocab_tokens.stoi.get(t, self.unk_token_id)
            for t in raw_tokens[:seq_len]
        ]
        token_ids += [self.pad_token_id] * (Config.MAX_SEQ_LEN - seq_len)

        # 2. Character IDs (3D tensor preparation)
        # We construct a list of lists, then pad
        char_ids = []
        for t in raw_tokens[:seq_len]:
            chars = list(str(t))
            c_ids = [
                self.vocab_chars.stoi.get(c, self.unk_char_id)
                for c in chars[: Config.MAX_CHAR_LEN]
            ]
            c_ids += [self.pad_char_id] * (Config.MAX_CHAR_LEN - len(c_ids))
            char_ids.append(c_ids)

        # Pad the sequence dimension for chars
        for _ in range(Config.MAX_SEQ_LEN - seq_len):
            char_ids.append([self.pad_char_id] * Config.MAX_CHAR_LEN)

        # 3. Labels
        label_ids = []
        if not self.is_test:
            raw_classes = row["class"]
            label_ids = [
                self.vocab_classes.stoi.get(c, self.pad_class_id)
                for c in raw_classes[:seq_len]
            ]
            label_ids += [self.pad_class_id] * (Config.MAX_SEQ_LEN - seq_len)
        else:
            # Dummy labels for test
            label_ids = [0] * Config.MAX_SEQ_LEN

        # 4. Mask
        mask = [1] * seq_len + [0] * (Config.MAX_SEQ_LEN - seq_len)

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),  # Bool for masking
            "original_len": torch.tensor(seq_len, dtype=torch.long),
        }


class Seq2SeqDataset(Dataset):
    """
    Dataset for Transformer Fallback (Seq2Seq).
    Input: Raw Token Chars + Class ID
    Output: Normalized Token Chars
    """

    def __init__(
        self, data_df: pd.DataFrame, vocab_chars: Vocabulary, vocab_classes: Vocabulary
    ):
        self.data = data_df
        self.vocab_chars = vocab_chars
        self.vocab_classes = vocab_classes

        self.pad_char_id = vocab_chars.stoi[Config.PAD_TOKEN]
        self.unk_char_id = vocab_chars.stoi[Config.UNK_TOKEN]
        self.sos_char_id = vocab_chars.stoi[Config.SOS_TOKEN]
        self.eos_char_id = vocab_chars.stoi[Config.EOS_TOKEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        raw_text = str(row["before"])
        target_text = str(row["after"])
        class_name = str(row["class"])

        # Source: Chars
        src_indices = [self.vocab_chars.stoi.get(c, self.unk_char_id) for c in raw_text]
        # Truncate or Pad Source
        if len(src_indices) > Config.MAX_CHAR_LEN:
            src_indices = src_indices[: Config.MAX_CHAR_LEN]
        else:
            # No padding needed here if we use collate_fn or if we just pad to max
            # For simplicity with batching, let's pad to MAX_CHAR_LEN
            src_indices += [self.pad_char_id] * (Config.MAX_CHAR_LEN - len(src_indices))

        # Target: <SOS> + Chars + <EOS>
        tgt_indices = (
            [self.sos_char_id]
            + [self.vocab_chars.stoi.get(c, self.unk_char_id) for c in target_text]
            + [self.eos_char_id]
        )

        # Pad Target to SEQ2SEQ_MAX_OUTPUT_LEN
        if len(tgt_indices) > Config.SEQ2SEQ_MAX_OUTPUT_LEN:
            tgt_indices = tgt_indices[: Config.SEQ2SEQ_MAX_OUTPUT_LEN]
            tgt_indices[-1] = self.eos_char_id  # Ensure EOS is present
        else:
            tgt_indices += [self.pad_char_id] * (
                Config.SEQ2SEQ_MAX_OUTPUT_LEN - len(tgt_indices)
            )

        class_id = self.vocab_classes.stoi.get(
            class_name, 0
        )  # Default to 0 if unknown (shouldn't happen)

        return {
            "src_ids": torch.tensor(src_indices, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_indices, dtype=torch.long),
            "class_id": torch.tensor(class_id, dtype=torch.long),
        }


# --------------------------------------------------------------------------
# Data Processing & Caching
# --------------------------------------------------------------------------
def group_sentences(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """
    Groups a token-per-row DataFrame into a sentence-per-row DataFrame.
    """
    # Ensure string types
    df["before"] = df["before"].astype(str)
    if not is_test:
        df["class"] = df["class"].astype(str)
        df["after"] = df["after"].astype(str)

    # Group by sentence_id
    # We use list aggregation
    if is_test:
        grouped = df.groupby("sentence_id")["before"].apply(list).reset_index()
        # Also need IDs to reconstruct submission
        grouped_ids = df.groupby("sentence_id")["id"].apply(list).reset_index()
        grouped = grouped.merge(grouped_ids, on="sentence_id")
    else:
        grouped = (
            df.groupby("sentence_id")
            .agg({"before": list, "after": list, "class": list})
            .reset_index()
        )

    return grouped


def build_knowledge_base(
    train_df: pd.DataFrame, save_path: str, load_cached: bool = True
) -> Dict[Tuple[str, str], str]:
    """
    Builds a deterministic dictionary (raw, class) -> normalized.
    Saves/Loads from Parquet.
    """
    if load_cached and os.path.exists(save_path):
        print(f"Loading Knowledge Base from {save_path}...")
        kb_df = pd.read_parquet(save_path)
        # Convert to dict
        kb = {}
        for _, row in kb_df.iterrows():
            kb[(row["before"], row["class"])] = row["after"]
        return kb

    print("Building Knowledge Base from training data...")
    # Filter for valid pairs
    # We prioritize the most frequent mapping if conflicts exist (though unlikely for same class)
    # Actually, for this task, (token, class) -> after is usually 1:1.
    # If duplicates exist, we take the most common one.

    # Count occurrences
    counts = (
        train_df.groupby(["before", "class", "after"]).size().reset_index(name="count")
    )
    # Sort by count desc
    counts = counts.sort_values("count", ascending=False)
    # Drop duplicates keeping first (most frequent)
    unique_mappings = counts.drop_duplicates(subset=["before", "class"])

    # Save to parquet
    unique_mappings[["before", "class", "after"]].to_parquet(save_path, index=False)

    kb = {}
    for _, row in unique_mappings.iterrows():
        kb[(row["before"], row["class"])] = row["after"]

    print(f"Knowledge Base built with {len(kb)} entries.")
    return kb


def get_data(load_cached: bool = True):
    """
    Main function to load, process, and cache all data and vocabularies.
    Returns:
        vocab_tokens, vocab_chars, vocab_classes, train_grouped, val_grouped, test_grouped, seq2seq_train_df
    """
    set_seed()
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Paths for cached grouped dataframes
    train_grouped_path = os.path.join(Config.WORK_DIR, "train_grouped.parquet")
    val_grouped_path = os.path.join(Config.WORK_DIR, "val_grouped.parquet")
    test_grouped_path = os.path.join(Config.WORK_DIR, "test_grouped.parquet")

    # 1. Load or Build Vocabularies
    vocab_tokens = Vocabulary("tokens", specials=[Config.PAD_TOKEN, Config.UNK_TOKEN])
    vocab_chars = Vocabulary(
        "chars",
        specials=[
            Config.PAD_TOKEN,
            Config.UNK_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
        ],
    )
    vocab_classes = Vocabulary(
        "classes", specials=[Config.PAD_TOKEN]
    )  # Classes usually don't need UNK if we cover all

    vocabs_exist = (
        os.path.exists(Config.VOCAB_TOKENS_PATH)
        and os.path.exists(Config.VOCAB_CHARS_PATH)
        and os.path.exists(Config.VOCAB_CLASSES_PATH)
    )

    if load_cached and vocabs_exist:
        print("Loading vocabularies from cache...")
        vocab_tokens.load(Config.VOCAB_TOKENS_PATH)
        vocab_chars.load(Config.VOCAB_CHARS_PATH)
        vocab_classes.load(Config.VOCAB_CLASSES_PATH)
    else:
        print("Building vocabularies...")
        # Load raw train data for vocab building
        df_train = pd.read_csv(Config.TRAIN_FILE, keep_default_na=False)
        if Config.DEBUG:
            df_train = df_train.iloc[: Config.DEBUG_SIZE]

        # Build Tokens Vocab
        vocab_tokens.build(
            df_train["before"].astype(str).tolist(),
            min_freq=Config.MIN_FREQ,
            max_size=Config.MAX_VOCAB_SIZE,
        )

        # Build Classes Vocab
        vocab_classes.build(df_train["class"].astype(str).tolist())

        # Build Chars Vocab (from both before and after)
        all_text = (
            df_train["before"].astype(str).tolist()
            + df_train["after"].astype(str).tolist()
        )
        all_chars = [c for text in all_text for c in text]
        vocab_chars.build(all_chars)

        # Save
        vocab_tokens.save(Config.VOCAB_TOKENS_PATH)
        vocab_chars.save(Config.VOCAB_CHARS_PATH)
        vocab_classes.save(Config.VOCAB_CLASSES_PATH)

    print(
        f"Vocab Sizes: Tokens={len(vocab_tokens)}, Chars={len(vocab_chars)}, Classes={len(vocab_classes)}"
    )

    # 2. Load or Process Grouped Data (for Tagger)
    if (
        load_cached
        and os.path.exists(train_grouped_path)
        and os.path.exists(val_grouped_path)
        and os.path.exists(test_grouped_path)
    ):
        print("Loading grouped datasets from cache...")
        train_grouped = pd.read_parquet(train_grouped_path)
        val_grouped = pd.read_parquet(val_grouped_path)
        test_grouped = pd.read_parquet(test_grouped_path)
    else:
        print("Processing and grouping datasets...")
        # Load Raw
        df_train = pd.read_csv(Config.TRAIN_FILE, keep_default_na=False)
        df_val = pd.read_csv(Config.VAL_FILE, keep_default_na=False)
        df_test = pd.read_csv(Config.TEST_FILE, keep_default_na=False)

        if Config.DEBUG:
            print("Debug mode: Subsampling data...")
            # Subsample by sentence_id to keep integrity
            train_sents = df_train["sentence_id"].unique()[:2000]
            df_train = df_train[df_train["sentence_id"].isin(train_sents)]

            val_sents = df_val["sentence_id"].unique()[:500]
            df_val = df_val[df_val["sentence_id"].isin(val_sents)]

            test_sents = df_test["sentence_id"].unique()[:100]
            df_test = df_test[df_test["sentence_id"].isin(test_sents)]

        # Group
        print("Grouping Train...")
        train_grouped = group_sentences(df_train, is_test=False)
        print("Grouping Val...")
        val_grouped = group_sentences(df_val, is_test=False)
        print("Grouping Test...")
        test_grouped = group_sentences(df_test, is_test=True)

        # Save
        print("Saving grouped datasets...")
        train_grouped.to_parquet(train_grouped_path, index=False)
        val_grouped.to_parquet(val_grouped_path, index=False)
        test_grouped.to_parquet(test_grouped_path, index=False)

    # 3. Prepare Seq2Seq Data (Filtered Train)
    # We don't necessarily need to cache this separately if it's fast, but let's be safe.
    # It comes from df_train where before != after
    # We need to reload raw df_train if we didn't keep it, or process from grouped?
    # Better to process from raw to avoid exploding the lists again.
    # We'll just load raw train again efficiently or assume it's fast enough.
    # Since we are inside get_data, let's just do it.

    # Note: For Seq2Seq, we only train on items that CHANGE.
    # However, to save memory, we can just return the path or let the caller handle it.
    # But the requirement asks for DataLoaders.

    # Let's return the raw DF subset for Seq2Seq
    # We'll load the raw train file one more time to filter for Seq2Seq
    # This is safer than unrolling the grouped dataframe.
    df_train_raw = pd.read_csv(Config.TRAIN_FILE, keep_default_na=False)
    if Config.DEBUG:
        df_train_raw = df_train_raw.iloc[: Config.DEBUG_SIZE]

    seq2seq_train_df = df_train_raw[
        df_train_raw["before"] != df_train_raw["after"]
    ].copy()

    # Build Knowledge Base
    build_knowledge_base(
        df_train_raw, Config.KNOWLEDGE_BASE_PATH, load_cached=load_cached
    )

    return (
        vocab_tokens,
        vocab_chars,
        vocab_classes,
        train_grouped,
        val_grouped,
        test_grouped,
        seq2seq_train_df,
    )


# --------------------------------------------------------------------------
# Loader Generators
# --------------------------------------------------------------------------
def get_tagger_loaders(batch_size=Config.BATCH_SIZE, load_cached=True):
    """
    Returns DataLoaders for the Tagger model (Train, Val, Test).
    """
    vt, vc, vcl, train_g, val_g, test_g, _ = get_data(load_cached=load_cached)

    train_ds = TaggerDataset(train_g, vt, vc, vcl, is_test=False)
    val_ds = TaggerDataset(val_g, vt, vc, vcl, is_test=False)
    test_ds = TaggerDataset(test_g, vt, vc, vcl, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vt, vc, vcl


def get_seq2seq_loaders(batch_size=Config.BATCH_SIZE, load_cached=True):
    """
    Returns DataLoaders for the Seq2Seq model (Train only, usually).
    Validation for Seq2Seq can be a subset of the train split or the same val set filtered.
    """
    vt, vc, vcl, _, val_g, _, train_s2s_df = get_data(load_cached=load_cached)

    # For validation of Seq2Seq, we need to extract changed tokens from val_grouped
    # Or easier: read raw val file and filter.
    df_val_raw = pd.read_csv(Config.VAL_FILE, keep_default_na=False)
    if Config.DEBUG:
        df_val_raw = df_val_raw.iloc[: Config.DEBUG_SIZE]
    val_s2s_df = df_val_raw[df_val_raw["before"] != df_val_raw["after"]].copy()

    train_ds = Seq2SeqDataset(train_s2s_df, vc, vcl)
    val_ds = Seq2SeqDataset(val_s2s_df, vc, vcl)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, vc, vcl
