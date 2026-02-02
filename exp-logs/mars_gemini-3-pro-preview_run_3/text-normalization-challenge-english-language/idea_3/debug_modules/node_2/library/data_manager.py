import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import CharTokenizer, build_or_load_tokenizer, encode_context_window


def load_parquet_data(split="train"):
    """
    Loads the raw metadata parquet files.
    Args:
        split: 'train', 'val', or 'test'
    Returns:
        pd.DataFrame
    """
    if split == "train":
        path = Config.TRAIN_DATA_PATH
    elif split == "val":
        path = Config.VAL_DATA_PATH
    elif split == "test":
        path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_parquet(path)

    # Debug mode subsampling
    if Config.DEBUG:
        print(f"[DEBUG] Subsampling {split} data to {Config.DEBUG_SIZE} rows.")
        if len(df) > Config.DEBUG_SIZE:
            # Try to keep sentences intact if possible, but simple head is safer for stability
            df = df.head(Config.DEBUG_SIZE).copy()

    return df


def _add_context_columns(df):
    """
    Adds 'prev_before' and 'next_before' columns to the dataframe
    based on sentence_id and token_id order.
    """
    # Ensure sorted order
    df = df.sort_values(["sentence_id", "token_id"]).reset_index(drop=True)

    # Vectorized shift
    # We group by sentence_id to ensure we don't shift across sentence boundaries
    # However, groupby().shift() can be slow.
    # Faster approach: shift global and mask where sentence_id changes.

    df["prev_before"] = df["before"].shift(1).fillna("")
    df["next_before"] = df["before"].shift(-1).fillna("")

    # Check boundaries
    # If sentence_id[i] != sentence_id[i-1], then prev is invalid
    sentence_ids = df["sentence_id"].values

    # Mask start of sentences
    # valid_prev: sentence_id[i] == sentence_id[i-1]
    valid_prev = np.concatenate(([False], sentence_ids[1:] == sentence_ids[:-1]))
    df.loc[~valid_prev, "prev_before"] = ""

    # Mask end of sentences
    # valid_next: sentence_id[i] == sentence_id[i+1]
    valid_next = np.concatenate((sentence_ids[:-1] == sentence_ids[1:], [False]))
    df.loc[~valid_next, "next_before"] = ""

    return df


def build_symbolic_stats(df_train, load_cached_data=True):
    """
    Constructs N-gram frequency maps for symbolic lookup.
    Returns dictionaries: trigram, bigram_left, bigram_right, unigram.
    """
    stats_dir = Config.STATS_CACHE_DIR
    os.makedirs(stats_dir, exist_ok=True)

    files = {
        "trigram": os.path.join(stats_dir, "stats_trigram.parquet"),
        "bigram_left": os.path.join(stats_dir, "stats_bigram_left.parquet"),
        "bigram_right": os.path.join(stats_dir, "stats_bigram_right.parquet"),
        "unigram": os.path.join(stats_dir, "stats_unigram.parquet"),
    }

    # Helper to load dict from parquet
    def load_dict(path, key_cols):
        if not os.path.exists(path):
            return None
        print(f"Loading stats from {path}...")
        d = pd.read_parquet(path)
        # Convert to dict
        if len(key_cols) == 1:
            return dict(zip(d[key_cols[0]], d["after"]))
        else:
            return dict(zip(zip(*[d[c] for c in key_cols]), d["after"]))

    # Check cache
    if load_cached_data:
        results = {}
        all_exist = True
        key_defs = {
            "trigram": ["prev", "curr", "next"],
            "bigram_left": ["prev", "curr"],
            "bigram_right": ["curr", "next"],
            "unigram": ["curr"],
        }

        for name, path in files.items():
            loaded = load_dict(path, key_defs[name])
            if loaded is None:
                all_exist = False
                break
            results[name] = loaded

        if all_exist:
            print("All symbolic stats loaded from cache.")
            return results

    print("Building symbolic stats from scratch...")

    # Ensure context
    if "prev_before" not in df_train.columns:
        df_train = _add_context_columns(df_train)

    # Prepare data for aggregation
    # We want most frequent 'after' for each N-gram

    def get_best_mapping(df_grouped, group_cols):
        # Count occurrences: group by (N-gram keys + target)
        counts = (
            df_grouped.groupby(group_cols + ["after"]).size().reset_index(name="count")
        )
        # Sort by count desc
        counts = counts.sort_values("count", ascending=False)
        # Drop duplicates on keys, keeping first (most frequent)
        best = counts.drop_duplicates(subset=group_cols)
        return best[group_cols + ["after"]]

    # 1. Unigram: curr -> after
    print("Computing Unigrams...")
    uni_df = get_best_mapping(df_train, ["before"])
    uni_df.columns = ["curr", "after"]
    uni_df.to_parquet(files["unigram"], index=False)
    unigram_dict = dict(zip(uni_df["curr"], uni_df["after"]))

    # 2. Bigram Left: (prev, curr) -> after
    print("Computing Left Bigrams...")
    bi_l_df = get_best_mapping(df_train, ["prev_before", "before"])
    bi_l_df.columns = ["prev", "curr", "after"]
    bi_l_df.to_parquet(files["bigram_left"], index=False)
    bigram_left_dict = dict(
        zip(zip(bi_l_df["prev"], bi_l_df["curr"]), bi_l_df["after"])
    )

    # 3. Bigram Right: (curr, next) -> after
    print("Computing Right Bigrams...")
    bi_r_df = get_best_mapping(df_train, ["before", "next_before"])
    bi_r_df.columns = ["curr", "next", "after"]
    bi_r_df.to_parquet(files["bigram_right"], index=False)
    bigram_right_dict = dict(
        zip(zip(bi_r_df["curr"], bi_r_df["next"]), bi_r_df["after"])
    )

    # 4. Trigram: (prev, curr, next) -> after
    print("Computing Trigrams...")
    tri_df = get_best_mapping(df_train, ["prev_before", "before", "next_before"])
    tri_df.columns = ["prev", "curr", "next", "after"]
    tri_df.to_parquet(files["trigram"], index=False)
    trigram_dict = dict(
        zip(zip(tri_df["prev"], tri_df["curr"], tri_df["next"]), tri_df["after"])
    )

    return {
        "trigram": trigram_dict,
        "bigram_left": bigram_left_dict,
        "bigram_right": bigram_right_dict,
        "unigram": unigram_dict,
    }


def prepare_neural_dataframe(df, split="train", load_cached_data=True):
    """
    Prepares the dataframe for the neural model:
    1. Adds context (prev, next).
    2. Filters for 'hard' cases (if training).
    3. Caches result.
    """
    cache_path = os.path.join(Config.PROCESSED_DATA_DIR, f"{split}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed {split} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data for neural model...")

    # 1. Add Context
    df = _add_context_columns(df)

    # 2. Filter (Only for Train/Val)
    # Test set must keep all rows requested, or we filter externally.
    # Here we assume we process what is given, but for Train we strictly filter.
    if split in ["train", "val"]:
        # Logic: Keep if class NOT in SAFE_CLASSES OR has digits/symbols
        # Note: SAFE_CLASSES = {"PLAIN", "PUNCT"}

        # Vectorized check for digits or symbols
        # This can be slow on strings.
        # Heuristic: if class is PLAIN or PUNCT, we might still want it if it looks weird?
        # The prompt says: "Keep only samples where the token class is NOT PLAIN or PUNCT,
        # OR where the token contains digits or symbols."

        # Condition A: Class is dangerous
        cond_class = ~df["class"].isin(Config.SAFE_CLASSES)

        # Condition B: Content is dangerous (digits or non-alnum)
        # We can use regex or string methods. Regex is cleaner in pandas.
        # r'[0-9\W_]' matches digits or non-alphanumeric (including underscore)
        # But we need to exclude spaces from "non-alphanumeric" if words have spaces?
        # Usually tokens don't have spaces unless multi-word.
        # Let's use a lambda for correctness as per prompt "digits or symbols".

        def is_complex(text):
            text = str(text)
            for c in text:
                if c.isdigit() or (not c.isalnum() and not c.isspace()):
                    return True
            return False

        # Apply is_complex only to rows that failed cond_class to save time?
        # No, it's an OR condition.

        # To speed up, we can just use the class filter primarily,
        # and then add PLAIN/PUNCT that look complex.

        # For efficiency in this constrained environment, let's trust the class labels
        # provided in training data mostly, but add the regex check.
        # However, applying a lambda over 7M rows is slow.
        # Use regex: contains digit OR (not alnum and not space)
        # Regex: `\d|[^\w\s]`

        cond_complex = df["before"].astype(str).str.contains(r"\d|[^\w\s]", regex=True)

        mask = cond_class | cond_complex

        original_len = len(df)
        df = df[mask].copy()
        print(
            f"Filtered {split} set: {original_len} -> {len(df)} rows ({(len(df)/original_len)*100:.2f}%)"
        )

    # 3. Save
    print(f"Saving processed data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def get_tokenizer(df_train=None, load_cached_data=True):
    """
    Returns the CharTokenizer. Fits on df_train if not cached.
    """
    vocab_path = os.path.join(Config.WORKING_DIR, "tokenizer.json")

    if load_cached_data and os.path.exists(vocab_path):
        return build_or_load_tokenizer(None, vocab_path, load_cached_data=True)

    if df_train is None:
        raise ValueError(
            "Tokenizer not found in cache and no training data provided to fit."
        )

    print("Fitting tokenizer on training data...")
    # We fit on 'before' and 'after' to cover all characters
    texts = pd.concat(
        [df_train["before"].astype(str), df_train["after"].astype(str)]
    ).unique()

    return build_or_load_tokenizer(texts, vocab_path, load_cached_data=False)


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    Input: [prev] <SEP> [curr] <SEP> [next]
    Target: [after]
    """

    def __init__(self, df, tokenizer, max_len=128, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Pre-extract columns to numpy for faster access
        self.prevs = df["prev_before"].astype(str).values
        self.currs = df["before"].astype(str).values
        self.nexts = df["next_before"].astype(str).values

        if self.mode == "train":
            self.targets = df["after"].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get strings
        p = self.prevs[idx]
        c = self.currs[idx]
        n = self.nexts[idx]

        # Encode Source
        src_ids = encode_context_window(self.tokenizer, p, c, n)

        # Truncate/Pad Source
        # We assume encode_context_window returns raw IDs.
        # We need to handle length.
        if len(src_ids) > self.max_len:
            # Center truncation around the current token?
            # Simple approach: truncate end
            src_ids = src_ids[: self.max_len]

        # Convert to tensor
        src_tensor = torch.tensor(src_ids, dtype=torch.long)

        if self.mode != "train":
            return src_tensor

        # Encode Target
        target_str = self.targets[idx]
        # Add SOS and EOS
        tgt_ids = self.tokenizer.encode(target_str, add_special_tokens=True)

        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[: self.max_len]

        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return src_tensor, tgt_tensor


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences with padding.
    """
    pad_id = 0  # Assuming <pad> is at index 0, but better to check tokenizer.
    # We don't have tokenizer instance here easily unless passed.
    # Config.TOKEN_PAD is defined, but we need the ID.
    # We can assume the standard CharTokenizer logic: special tokens are first.
    # <pad> is usually 0.

    # Check if batch contains targets
    has_targets = isinstance(batch[0], tuple)

    if has_targets:
        srcs, tgts = zip(*batch)
    else:
        srcs = batch
        tgts = None

    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence

    # We need to access the pad_id. Since we can't easily pass tokenizer to collate_fn
    # without partials, and CharTokenizer implementation puts PAD at index 0:
    # self.char2id[Config.TOKEN_PAD] = 0
    pad_id = 0

    src_padded = pad_sequence(srcs, batch_first=True, padding_value=pad_id)

    if tgts is not None:
        tgt_padded = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
        return src_padded, tgt_padded

    return src_padded
