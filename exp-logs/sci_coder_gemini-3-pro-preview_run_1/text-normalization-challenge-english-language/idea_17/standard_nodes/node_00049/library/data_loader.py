import os
import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from library.config import (
    CACHE_DIR,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    PLAIN_SUBSAMPLE_RATIO,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    TAGGER_BATCH_SIZE,
    SEQ2SEQ_BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.utils import extract_regex_features, set_seed
from library.vocabulary import Vocabulary

# Set seed for reproducibility
set_seed(SEED)


def build_knowledge_base(load_cached_data: bool = True) -> pd.DataFrame:
    """
    Constructs a deterministic mapping from (before, class) -> after using the training set.
    Resolves conflicts by taking the most frequent normalization.
    """
    cache_path = os.path.join(CACHE_DIR, "knowledge_base.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading Knowledge Base from {cache_path}...")
        return pd.read_parquet(cache_path)

    print("Building Knowledge Base from source...")
    df = pd.read_csv(TRAIN_DATA_PATH, dtype=str, keep_default_na=False)

    # Group by input token and class, then find the most frequent target
    # We use a trick: groupby -> value_counts -> head(1) to get mode
    kb_df = (
        df.groupby(["before", "class"])["after"]
        .agg(
            lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else x.iloc[0]
        )
        .reset_index()
    )

    print(f"Knowledge Base constructed with {len(kb_df)} entries.")
    kb_df.to_parquet(cache_path, index=False)
    return kb_df


def _preprocess_tagger_data(
    df: pd.DataFrame,
    vocab_words: Vocabulary,
    vocab_chars: Vocabulary,
    vocab_classes: Vocabulary,
    is_test: bool = False,
) -> List[Dict]:
    """
    Helper to process dataframe into list of sentence dictionaries.
    """
    # Group by sentence_id
    # We need to preserve order, so we assume the CSV is sorted or we sort it.
    # The metadata generation script ensures sorting by sentence_id, token_id.

    print("Grouping tokens into sentences...")
    # Using a more memory efficient approach than simple groupby for large dfs
    sentences = []

    # We'll iterate through the dataframe.
    # Since it's sorted by sentence_id, we can just chunk it.
    # However, pandas groupby is optimized enough for 7M rows usually.

    # Extract columns needed
    cols = ["sentence_id", "before"]
    if not is_test:
        cols.extend(["class"])

    # Pre-calculate regex features to speed up loop
    print("Extracting regex features...")
    # This can be slow, so we use a list comprehension which is often faster than .apply
    raw_tokens = df["before"].astype(str).tolist()
    regex_feats_list = [
        extract_regex_features(t) for t in tqdm(raw_tokens, desc="Regex")
    ]

    df["regex_features"] = regex_feats_list

    # Group
    groups = df.groupby("sentence_id")

    print("Converting to tensors...")
    for _, group in tqdm(groups, desc="Sentences"):
        tokens = group["before"].astype(str).tolist()
        regex = group["regex_features"].tolist()

        # Word IDs
        word_ids = vocab_words.lookup_indices(tokens)

        # Char IDs (List of Lists)
        char_ids_list = []
        for t in tokens:
            # For tagger, we just map chars to IDs. No SOS/EOS needed for CNN.
            c_ids = vocab_chars.lookup_indices(list(t))
            if len(c_ids) == 0:
                c_ids = [0]  # Handle empty string if any
            char_ids_list.append(c_ids)

        sample = {
            "word_ids": word_ids,
            "char_ids": char_ids_list,
            "regex_features": regex,
        }

        if not is_test:
            classes = group["class"].astype(str).tolist()
            class_ids = vocab_classes.lookup_indices(classes)
            sample["class_ids"] = class_ids

            # Store sentence ID for tracking if needed, though not strictly required for training
            sample["sentence_id"] = group["sentence_id"].iloc[0]
        else:
            sample["sentence_id"] = group["sentence_id"].iloc[0]
            # Store token IDs for submission mapping
            sample["token_ids"] = group["token_id"].tolist()

        sentences.append(sample)

    return sentences


def get_tagger_data(
    split: str,
    vocab_words: Vocabulary,
    vocab_chars: Vocabulary,
    vocab_classes: Vocabulary,
    load_cached_data: bool = True,
) -> List[Dict]:
    """
    Loads, filters (Signal-Dense), and preprocesses data for the Tagger.
    """
    cache_file = f"tagger_{split}_data.pt"
    cache_path = os.path.join(CACHE_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading preprocessed tagger data ({split}) from {cache_path}...")
        return torch.load(cache_path)

    print(f"Processing tagger data for split: {split}...")

    if split == "train":
        df = pd.read_csv(TRAIN_DATA_PATH, dtype=str, keep_default_na=False)

        # --- Signal-Dense Filtering ---
        print("Applying Signal-Dense Filtering...")
        # Identify interesting classes
        boring_classes = {"PLAIN", "PUNCT"}

        # We need to know which sentences have interesting tokens.
        # Create a boolean mask for interesting tokens
        df["is_interesting"] = ~df["class"].isin(boring_classes)

        # Group by sentence to find interesting sentences
        sent_interesting = df.groupby("sentence_id")["is_interesting"].any()

        interesting_sent_ids = sent_interesting[sent_interesting].index
        boring_sent_ids = sent_interesting[~sent_interesting].index

        print(f"Total Sentences: {len(sent_interesting)}")
        print(f"Interesting Sentences (Keep 100%): {len(interesting_sent_ids)}")
        print(f"Boring Sentences (Total): {len(boring_sent_ids)}")

        # Subsample boring sentences
        # Ensure reproducibility
        rng = np.random.RandomState(SEED)
        keep_boring_count = int(len(boring_sent_ids) * PLAIN_SUBSAMPLE_RATIO)
        keep_boring_ids = rng.choice(
            boring_sent_ids, size=keep_boring_count, replace=False
        )
        print(f"Boring Sentences (Kept): {len(keep_boring_ids)}")

        valid_sent_ids = set(interesting_sent_ids).union(set(keep_boring_ids))

        # Filter dataframe
        df = df[df["sentence_id"].isin(valid_sent_ids)].copy()
        print(f"Filtered Train DataFrame size: {len(df)} rows")

        data = _preprocess_tagger_data(
            df, vocab_words, vocab_chars, vocab_classes, is_test=False
        )

    elif split == "val":
        df = pd.read_csv(VAL_DATA_PATH, dtype=str, keep_default_na=False)
        # No filtering for validation, we want true metrics
        data = _preprocess_tagger_data(
            df, vocab_words, vocab_chars, vocab_classes, is_test=False
        )

    elif split == "test":
        df = pd.read_csv(TEST_DATA_PATH, dtype=str, keep_default_na=False)
        data = _preprocess_tagger_data(
            df, vocab_words, vocab_chars, vocab_classes, is_test=True
        )

    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"Saving processed data to {cache_path}...")
    torch.save(data, cache_path)
    return data


class TaggerDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def tagger_collate_fn(batch: List[Dict]):
    """
    Custom collate to handle:
    1. Padding word_ids (batch, max_seq_len)
    2. Padding regex_features (batch, max_seq_len, num_feats)
    3. Padding char_ids (batch, max_seq_len, max_char_len) - This is the tricky 3D one.
    4. Padding class_ids (batch, max_seq_len)
    """
    # Extract sequences
    word_ids_seqs = [torch.tensor(item["word_ids"], dtype=torch.long) for item in batch]
    regex_seqs = [
        torch.tensor(item["regex_features"], dtype=torch.float) for item in batch
    ]

    # Pad 1D sequences (Words, Regex, Classes)
    padded_word_ids = pad_sequence(word_ids_seqs, batch_first=True, padding_value=0)
    padded_regex = pad_sequence(regex_seqs, batch_first=True, padding_value=0)

    # Handle Labels if present
    padded_class_ids = None
    if "class_ids" in batch[0]:
        class_ids_seqs = [
            torch.tensor(item["class_ids"], dtype=torch.long) for item in batch
        ]
        # We use -100 or 0 for padding? usually CrossEntropyLoss ignores index -100 by default.
        # But our vocab has 0 as padding/unknown. Let's use 0 and handle masking if needed,
        # or -100. Let's use -100 for safety in loss calculation.
        padded_class_ids = pad_sequence(
            class_ids_seqs, batch_first=True, padding_value=-100
        )

    # Handle 3D Char IDs
    # Structure: List[List[List[int]]] -> (Batch, Seq, Char)
    # We need to find max_seq_len and max_char_len in this batch
    max_seq_len = padded_word_ids.size(1)
    max_char_len = 0
    for item in batch:
        for token_chars in item["char_ids"]:
            max_char_len = max(max_char_len, len(token_chars))

    # Initialize 3D tensor
    batch_size = len(batch)
    padded_char_ids = torch.zeros(
        (batch_size, max_seq_len, max_char_len), dtype=torch.long
    )

    for i, item in enumerate(batch):
        char_seq = item["char_ids"]  # List of lists
        for j, token_chars in enumerate(char_seq):
            length = len(token_chars)
            if length > 0:
                padded_char_ids[i, j, :length] = torch.tensor(
                    token_chars, dtype=torch.long
                )

    batch_out = {
        "word_ids": padded_word_ids,
        "char_ids": padded_char_ids,
        "regex_features": padded_regex,
    }

    if padded_class_ids is not None:
        batch_out["class_ids"] = padded_class_ids

    # Pass through metadata for inference reconstruction
    if "token_ids" in batch[0]:
        batch_out["token_ids"] = [item["token_ids"] for item in batch]
        batch_out["sentence_id"] = [item["sentence_id"] for item in batch]

    return batch_out


# -----------------------------------------------------------------------------
# SEQ2SEQ DATA LOADING
# -----------------------------------------------------------------------------


def get_seq2seq_data(
    split: str,
    vocab_chars: Vocabulary,
    vocab_classes: Vocabulary,
    load_cached_data: bool = True,
) -> List[Dict]:
    """
    Loads and preprocesses data for the Seq2Seq Fallback model.
    Filters only for tokens where before != after.
    """
    cache_file = f"seq2seq_{split}_data.pt"
    cache_path = os.path.join(CACHE_DIR, cache_file)

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading preprocessed seq2seq data ({split}) from {cache_path}...")
        return torch.load(cache_path)

    print(f"Processing seq2seq data for split: {split}...")

    if split == "train":
        df = pd.read_csv(TRAIN_DATA_PATH, dtype=str, keep_default_na=False)
    elif split == "val":
        df = pd.read_csv(VAL_DATA_PATH, dtype=str, keep_default_na=False)
    else:
        # We don't really have a 'test' set for seq2seq training loop,
        # inference is done dynamically on OOV tokens.
        return []

    # Filter for changed tokens only
    print("Filtering for changed tokens...")
    df_changed = df[df["before"] != df["after"]].copy()
    print(f"Original size: {len(df)}, Changed size: {len(df_changed)}")

    processed_data = []

    # Pre-lookup to speed up
    sos_id = vocab_chars.token2id[SOS_TOKEN]
    eos_id = vocab_chars.token2id[EOS_TOKEN]

    print("Converting to tensors...")
    # Iterate rows
    # Using itertuples is faster than iterrows
    for row in tqdm(
        df_changed.itertuples(index=False), total=len(df_changed), desc="Seq2Seq"
    ):
        # row: sentence_id, token_id, class, before, after, id
        # Note: pandas namedtuples rely on column names.

        src_text = str(row.before)
        trg_text = str(row.after)
        cls_text = str(
            getattr(row, "_2")
        )  # 'class' is a reserved keyword, pandas renames it to _2 usually or similar idx
        # To be safe, let's use dict access or explicit column mapping

    # Safer iteration
    before_list = df_changed["before"].astype(str).tolist()
    after_list = df_changed["after"].astype(str).tolist()
    class_list = df_changed["class"].astype(str).tolist()

    class_ids_all = vocab_classes.lookup_indices(class_list)

    for src, trg, cls_id in tqdm(
        zip(before_list, after_list, class_ids_all),
        total=len(before_list),
        desc="Seq2Seq",
    ):
        src_ids = vocab_chars.lookup_indices(list(src))

        # Target needs SOS and EOS
        trg_ids = [sos_id] + vocab_chars.lookup_indices(list(trg)) + [eos_id]

        processed_data.append(
            {"src_char_ids": src_ids, "trg_char_ids": trg_ids, "class_id": cls_id}
        )

    print(f"Saving processed data to {cache_path}...")
    torch.save(processed_data, cache_path)
    return processed_data


class Seq2SeqDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def seq2seq_collate_fn(batch: List[Dict]):
    src_seqs = [torch.tensor(item["src_char_ids"], dtype=torch.long) for item in batch]
    trg_seqs = [torch.tensor(item["trg_char_ids"], dtype=torch.long) for item in batch]
    class_ids = [item["class_id"] for item in batch]

    padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    padded_trg = pad_sequence(trg_seqs, batch_first=True, padding_value=0)
    class_ids_tensor = torch.tensor(class_ids, dtype=torch.long)

    return {
        "src_char_ids": padded_src,
        "trg_char_ids": padded_trg,
        "class_ids": class_ids_tensor,
    }


# -----------------------------------------------------------------------------
# MAIN LOADER FUNCTION
# -----------------------------------------------------------------------------


def get_dataloaders(
    vocab_words: Vocabulary,
    vocab_chars: Vocabulary,
    vocab_classes: Vocabulary,
    load_cached_data: bool = True,
):
    """
    Returns a dictionary of dataloaders for Tagger (train/val) and Seq2Seq (train/val).
    """

    # --- Tagger Data ---
    tagger_train_data = get_tagger_data(
        "train", vocab_words, vocab_chars, vocab_classes, load_cached_data
    )
    tagger_val_data = get_tagger_data(
        "val", vocab_words, vocab_chars, vocab_classes, load_cached_data
    )

    tagger_train_ds = TaggerDataset(tagger_train_data)
    tagger_val_ds = TaggerDataset(tagger_val_data)

    tagger_train_loader = DataLoader(
        tagger_train_ds,
        batch_size=TAGGER_BATCH_SIZE,
        shuffle=True,
        collate_fn=tagger_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    tagger_val_loader = DataLoader(
        tagger_val_ds,
        batch_size=TAGGER_BATCH_SIZE,
        shuffle=False,
        collate_fn=tagger_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # --- Seq2Seq Data ---
    seq2seq_train_data = get_seq2seq_data(
        "train", vocab_chars, vocab_classes, load_cached_data
    )
    seq2seq_val_data = get_seq2seq_data(
        "val", vocab_chars, vocab_classes, load_cached_data
    )

    seq2seq_train_ds = Seq2SeqDataset(seq2seq_train_data)
    seq2seq_val_ds = Seq2SeqDataset(seq2seq_val_data)

    seq2seq_train_loader = DataLoader(
        seq2seq_train_ds,
        batch_size=SEQ2SEQ_BATCH_SIZE,
        shuffle=True,
        collate_fn=seq2seq_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    seq2seq_val_loader = DataLoader(
        seq2seq_val_ds,
        batch_size=SEQ2SEQ_BATCH_SIZE,
        shuffle=False,
        collate_fn=seq2seq_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    return {
        "tagger_train": tagger_train_loader,
        "tagger_val": tagger_val_loader,
        "seq2seq_train": seq2seq_train_loader,
        "seq2seq_val": seq2seq_val_loader,
    }


def get_test_dataloader(
    vocab_words: Vocabulary,
    vocab_chars: Vocabulary,
    vocab_classes: Vocabulary,
    load_cached_data: bool = True,
):
    test_data = get_tagger_data(
        "test", vocab_words, vocab_chars, vocab_classes, load_cached_data
    )
    test_ds = TaggerDataset(test_data)

    # Larger batch size for inference
    test_loader = DataLoader(
        test_ds,
        batch_size=TAGGER_BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=tagger_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    return test_loader
