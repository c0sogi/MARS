import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from library.config import Config
from library.utils import load_raw_data, save_to_cache, load_from_cache, set_seed
from library.tokenization import CharTokenizer, BPETokenizer


class SemioticDataset(Dataset):
    def __init__(self, df, char_tokenizer, bpe_tokenizer, max_src_len, max_tgt_len):
        self.df = df.reset_index(drop=True)
        self.char_tokenizer = char_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Pre-convert columns to lists for faster access in __getitem__
        self.before = self.df["before"].astype(str).tolist()
        self.after = self.df["after"].astype(str).tolist()
        self.prev_1 = self.df["prev_1"].fillna("").astype(str).tolist()
        self.prev_2 = self.df["prev_2"].fillna("").astype(str).tolist()
        self.next_1 = self.df["next_1"].fillna("").astype(str).tolist()
        self.next_2 = self.df["next_2"].fillna("").astype(str).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve context and target
        p2 = self.prev_2[idx]
        p1 = self.prev_1[idx]
        curr = self.before[idx]
        n1 = self.next_1[idx]
        n2 = self.next_2[idx]
        target_text = self.after[idx]

        # Construct Source Sequence: Prev_2 Prev_1 <SEP> Target <SEP> Next_1 Next_2
        # We use the tokenizer's encode method which handles chars.
        # We insert the SEP token ID manually.
        sep_id = self.char_tokenizer.sep_token_id

        # Helper to encode string to ids without specials
        def enc(text):
            return self.char_tokenizer.encode(text, add_special_tokens=False)

        # We use a space (or UNK if space not in vocab) to separate context words
        # Since CharTokenizer might not have space, we can check.
        # If space is not in vocab, it maps to UNK. That serves as a delimiter.
        space_ids = enc(" ")

        src_ids = []
        # Context Left
        if p2:
            src_ids.extend(enc(p2))
            src_ids.extend(space_ids)
        if p1:
            src_ids.extend(enc(p1))

        # Separator
        src_ids.append(sep_id)

        # Target Chars
        src_ids.extend(enc(curr))

        # Separator
        src_ids.append(sep_id)

        # Context Right
        if n1:
            src_ids.extend(enc(n1))
            src_ids.extend(space_ids)
        if n2:
            src_ids.extend(enc(n2))

        # Truncate source if necessary (keeping the center is ideal, but simple truncation is standard)
        # We prioritize the center (target) so we truncate ends if needed, but for now simple slice
        if len(src_ids) > self.max_src_len:
            src_ids = src_ids[: self.max_src_len]

        # Encode Target (BPE)
        tgt_ids = self.bpe_tokenizer.encode(target_text, add_special_tokens=True)
        if len(tgt_ids) > self.max_tgt_len:
            tgt_ids = tgt_ids[: self.max_tgt_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    # Pad sequences
    # For source, we pad with PAD token (0 usually, but check tokenizer)
    # CharTokenizer: PAD is 0 (first in specials)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)

    # For target, BPE tokenizer PAD ID
    # SentencePiece pad_id is usually 0, but we check BPETokenizer wrapper
    # In library.tokenization, BPETokenizer exposes pad_token_id
    # We need to instantiate one to check, but here we assume standard SP behavior or passed instance
    # The collate_fn usually needs access to pad_id.
    # We'll assume 0 for now as per standard SP training in library.
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded


def _add_context_columns(df):
    """
    Adds prev_1, prev_2, next_1, next_2 columns to the dataframe
    respecting sentence boundaries.
    """
    # Ensure sorted
    df = df.sort_values(by=["sentence_id", "token_id"]).copy()

    # Vectorized shifts
    df["prev_1"] = df["before"].shift(1)
    df["prev_2"] = df["before"].shift(2)
    df["next_1"] = df["before"].shift(-1)
    df["next_2"] = df["before"].shift(-2)

    # Check sentence boundaries
    # If sentence_id changes, the context is invalid (belongs to another sentence)
    sid = df["sentence_id"]

    # Masks
    mask_p1 = sid == sid.shift(1)
    mask_p2 = sid == sid.shift(2)
    mask_n1 = sid == sid.shift(-1)
    mask_n2 = sid == sid.shift(-2)

    # Apply masks
    df.loc[~mask_p1, "prev_1"] = ""
    df.loc[~mask_p2, "prev_2"] = ""
    df.loc[~mask_n1, "next_1"] = ""
    df.loc[~mask_n2, "next_2"] = ""

    return df


def build_hfbb_stats(load_cached_data=True):
    """
    Constructs and caches the Hierarchical Frequency Back-off statistics.
    """
    # Define paths
    paths = {
        "unigram": Config.HFBB_UNIGRAM_PATH,
        "bigram_prev": Config.HFBB_BIGRAM_PREV_PATH,
        "bigram_next": Config.HFBB_BIGRAM_NEXT_PATH,
        "trigram": Config.HFBB_TRIGRAM_PATH,
    }

    # Check cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in paths.values())
        if all_exist:
            print("Loading HFBB stats from cache...")
            return {k: load_from_cache(p) for k, p in paths.items()}

    print("Building HFBB stats from scratch...")

    # Load full training data
    df = load_raw_data("train")

    # Add context for n-grams
    df = _add_context_columns(df)

    # 1. Unigram with Confidence
    # Group by before, count after
    print("Computing Unigram stats...")
    uni_counts = df.groupby(["before", "after"]).size().reset_index(name="count")
    # Calculate total counts per 'before'
    uni_totals = uni_counts.groupby("before")["count"].transform("sum")
    uni_counts["confidence"] = uni_counts["count"] / uni_totals
    # Get the mode (max count) for each 'before'
    # Sort by count desc, then drop duplicates to keep top
    uni_best = uni_counts.sort_values("count", ascending=False).drop_duplicates(
        ["before"]
    )
    uni_best = uni_best[["before", "after", "confidence"]]

    # 2. Bigram Prev: (prev_1, before) -> after
    print("Computing Bigram (Prev) stats...")
    # Filter valid context
    mask_bp = df["prev_1"] != ""
    bi_prev_df = (
        df[mask_bp]
        .groupby(["prev_1", "before", "after"])
        .size()
        .reset_index(name="count")
    )
    bi_prev_best = bi_prev_df.sort_values("count", ascending=False).drop_duplicates(
        ["prev_1", "before"]
    )
    bi_prev_best = bi_prev_best[["prev_1", "before", "after"]]

    # 3. Bigram Next: (before, next_1) -> after
    print("Computing Bigram (Next) stats...")
    mask_bn = df["next_1"] != ""
    bi_next_df = (
        df[mask_bn]
        .groupby(["before", "next_1", "after"])
        .size()
        .reset_index(name="count")
    )
    bi_next_best = bi_next_df.sort_values("count", ascending=False).drop_duplicates(
        ["before", "next_1"]
    )
    bi_next_best = bi_next_best[["before", "next_1", "after"]]

    # 4. Trigram: (prev_1, before, next_1) -> after
    print("Computing Trigram stats...")
    mask_tri = (df["prev_1"] != "") & (df["next_1"] != "")
    tri_df = (
        df[mask_tri]
        .groupby(["prev_1", "before", "next_1", "after"])
        .size()
        .reset_index(name="count")
    )
    tri_best = tri_df.sort_values("count", ascending=False).drop_duplicates(
        ["prev_1", "before", "next_1"]
    )
    tri_best = tri_best[["prev_1", "before", "next_1", "after"]]

    # Save to cache
    print("Saving HFBB stats...")
    save_to_cache(uni_best, paths["unigram"])
    save_to_cache(bi_prev_best, paths["bigram_prev"])
    save_to_cache(bi_next_best, paths["bigram_next"])
    save_to_cache(tri_best, paths["trigram"])

    return {
        "unigram": uni_best,
        "bigram_prev": bi_prev_best,
        "bigram_next": bi_next_best,
        "trigram": tri_best,
    }


def _process_dataset(split, char_tokenizer, bpe_tokenizer, load_cached_data=True):
    """
    Internal function to process a specific split (train/val) into a SemioticDataset.
    """
    is_train = split == "train"
    cache_path = (
        Config.TRANSFORMER_TRAIN_PATH if is_train else Config.TRANSFORMER_VAL_PATH
    )

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed {split} data from cache...")
        df_processed = load_from_cache(cache_path)
    else:
        print(f"Processing {split} data from scratch...")
        df = load_raw_data(split)

        # Add context
        df = _add_context_columns(df)

        # Filter for semiotic tokens
        # We keep tokens that match the regex
        print(f"Filtering semiotic tokens (Regex: {Config.SEMIOTIC_REGEX})...")
        mask_semiotic = (
            df["before"].astype(str).str.contains(Config.SEMIOTIC_REGEX, regex=True)
        )
        df_semiotic = df[mask_semiotic].copy()

        if is_train:
            # Upsampling Logic
            print("Applying Class-Balanced Upsampling...")
            # Calculate class counts
            class_counts = df_semiotic["class"].value_counts()
            if not class_counts.empty:
                max_count = class_counts.max()
                print(f"Dominant class count: {max_count}")

                dfs_to_concat = []
                for cls, count in class_counts.items():
                    cls_df = df_semiotic[df_semiotic["class"] == cls]
                    # We upsample to max_count
                    if count < max_count:
                        # Sample with replacement
                        resampled = cls_df.sample(
                            n=max_count, replace=True, random_state=Config.SEED
                        )
                        dfs_to_concat.append(resampled)
                    else:
                        dfs_to_concat.append(cls_df)

                df_semiotic = pd.concat(dfs_to_concat, ignore_index=True)
                # Shuffle
                df_semiotic = df_semiotic.sample(
                    frac=1, random_state=Config.SEED
                ).reset_index(drop=True)
                print(f"Upsampled train size: {len(df_semiotic)}")

        # Save processed dataframe to cache
        save_to_cache(df_semiotic, cache_path)
        df_processed = df_semiotic

    # Create Dataset object
    dataset = SemioticDataset(
        df_processed,
        char_tokenizer,
        bpe_tokenizer,
        Config.MAX_SRC_LEN,
        Config.MAX_TGT_LEN,
    )
    return dataset


def get_dataloaders(char_tokenizer, bpe_tokenizer, load_cached_data=True):
    """
    Returns train and validation DataLoaders for the Transformer model.
    """
    # Train Dataset
    train_dataset = _process_dataset(
        "train", char_tokenizer, bpe_tokenizer, load_cached_data
    )

    # Val Dataset
    val_dataset = _process_dataset(
        "val", char_tokenizer, bpe_tokenizer, load_cached_data
    )

    # Debug Subset
    if Config.DEBUG_SUBSET_SIZE > 0:
        print(f"DEBUG: Subsetting train data to {Config.DEBUG_SUBSET_SIZE}")
        indices = torch.randperm(len(train_dataset))[: Config.DEBUG_SUBSET_SIZE]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader
