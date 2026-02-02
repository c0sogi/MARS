import os
import json
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Union
from library.config import Config
from library.utils import seed_everything, is_digit_token

# --- Constants ---
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SEP_TOKEN = "<sep>"


class CharTokenizer:
    """
    Character-level tokenizer for the neural normalizer.
    Handles mapping between characters and integer IDs.
    """

    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, SEP_TOKEN]

    def fit_on_texts(self, texts: List[str]):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for determinism
        sorted_chars = sorted(list(unique_chars))

        # Assign IDs
        self.char2id = {token: idx for idx, token in enumerate(self.specials)}
        start_idx = len(self.specials)
        for idx, char in enumerate(sorted_chars):
            self.char2id[char] = start_idx + idx

        self.id2char = {v: k for k, v in self.char2id.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Converts a string to a list of IDs.
        """
        text = str(text)
        ids = [self.char2id.get(c, self.char2id[UNK_TOKEN]) for c in text]
        if add_special_tokens:
            ids = [self.char2id[SOS_TOKEN]] + ids + [self.char2id[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        Converts a list of IDs back to a string.
        """
        chars = []
        for idx in ids:
            token = self.id2char.get(idx, UNK_TOKEN)
            if remove_special_tokens and token in self.specials:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str):
        """Saves the vocabulary to a JSON file."""
        data = {
            "char2id": self.char2id,
            "id2char": {k: v for k, v in self.id2char.items()},  # keys are ints
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.char2id = data["char2id"]
        # JSON keys are always strings, convert back to int
        self.id2char = {int(k): v for k, v in data["id2char"].items()}

    def __len__(self):
        return len(self.char2id)

    @property
    def pad_token_id(self) -> int:
        return self.char2id[PAD_TOKEN]

    @property
    def sos_token_id(self) -> int:
        return self.char2id[SOS_TOKEN]

    @property
    def eos_token_id(self) -> int:
        return self.char2id[EOS_TOKEN]


class ContextAnchoredDataset(Dataset):
    """
    PyTorch Dataset for the Context-Anchored Neural Model.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: CharTokenizer,
        config: Config,
        is_test: bool = False,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_seq_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row["input_text"]

        # Encoder input: No special tokens needed for encoder usually, but standard is often just raw
        # We will not add SOS/EOS to encoder input, just padding in collate
        src_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        if self.is_test:
            return {
                "id": row["id"],  # submission id
                "src": torch.tensor(src_ids, dtype=torch.long),
                "src_len": len(src_ids),
            }
        else:
            target_text = row["target_text"]
            # Decoder target: Needs SOS and EOS
            tgt_ids = self.tokenizer.encode(target_text, add_special_tokens=True)

            return {
                "src": torch.tensor(src_ids, dtype=torch.long),
                "tgt": torch.tensor(tgt_ids, dtype=torch.long),
                "src_len": len(src_ids),
                "tgt_len": len(tgt_ids),
            }


def collate_fn(batch, pad_idx):
    """
    Custom collate function to handle variable length sequences.
    """
    # Check if this is a test batch
    is_test = "tgt" not in batch[0]

    # Process Source
    src_lens = [item["src_len"] for item in batch]
    max_src_len = max(src_lens)

    padded_src = torch.full((len(batch), max_src_len), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["src_len"]
        padded_src[i, :length] = item["src"]

    batch_out = {"src": padded_src, "src_pad_mask": (padded_src == pad_idx)}

    if is_test:
        batch_out["id"] = [item["id"] for item in batch]
    else:
        # Process Target
        tgt_lens = [item["tgt_len"] for item in batch]
        max_tgt_len = max(tgt_lens)

        padded_tgt = torch.full((len(batch), max_tgt_len), pad_idx, dtype=torch.long)
        for i, item in enumerate(batch):
            length = item["tgt_len"]
            padded_tgt[i, :length] = item["tgt"]

        batch_out["tgt"] = padded_tgt
        # For loss calculation, we often ignore pad index, but mask is useful too
        batch_out["tgt_pad_mask"] = padded_tgt == pad_idx

    return batch_out


def construct_window_input(tokens: List[str], target_idx: int, window_size: int) -> str:
    """
    Constructs the input string: [Prev2] [Prev1] <sep> [Target] <sep> [Next1] [Next2]
    """
    seq_len = len(tokens)

    # Extract Context
    prev_tokens = []
    for i in range(window_size, 0, -1):
        curr = target_idx - i
        if curr >= 0:
            prev_tokens.append(str(tokens[curr]))
        # We do not explicitly add padding tokens to the string,
        # the model learns from the absence/presence of context words.
        # Alternatively, we could add a special placeholder, but empty list is fine for string join.

    next_tokens = []
    for i in range(1, window_size + 1):
        curr = target_idx + i
        if curr < seq_len:
            next_tokens.append(str(tokens[curr]))

    target_token = str(tokens[target_idx])

    # Construct string
    # Format: "prev_words <sep> target <sep> next_words"
    # Using space to separate words in context
    prev_str = " ".join(prev_tokens)
    next_str = " ".join(next_tokens)

    return f"{prev_str} {SEP_TOKEN} {target_token} {SEP_TOKEN} {next_str}".strip()


def process_split(
    df: pd.DataFrame,
    split_name: str,
    config: Config,
    tokenizer: Optional[CharTokenizer] = None,
) -> pd.DataFrame:
    """
    Core logic to process raw dataframe into neural training sequences.
    """
    print(f"Processing {split_name} split...")

    # 1. Group by sentence
    # This creates a Series where each item is a list of values for that sentence
    grouped_before = df.groupby("sentence_id")["before"].apply(list)
    grouped_token_ids = df.groupby("sentence_id")["token_id"].apply(list)

    has_targets = "after" in df.columns
    if has_targets:
        grouped_after = df.groupby("sentence_id")["after"].apply(list)
        grouped_class = df.groupby("sentence_id")["class"].apply(list)
    else:
        # For test set, we don't have after/class
        grouped_after = None
        grouped_class = None

    # 2. Iterate and Filter
    processed_samples = []

    # Get sentence IDs
    sentence_ids = grouped_before.index.tolist()

    # Stats
    total_tokens = 0
    selected_tokens = 0

    for sid in sentence_ids:
        tokens_in = grouped_before[sid]
        t_ids = grouped_token_ids[sid]

        if has_targets:
            tokens_out = grouped_after[sid]
            classes = grouped_class[sid]
        else:
            tokens_out = None
            classes = None

        seq_len = len(tokens_in)
        total_tokens += seq_len

        for idx in range(seq_len):
            token_text = str(tokens_in[idx])

            should_include = False

            if split_name == "test":
                # For test, we process tokens that look like they need neural normalization
                # i.e., they contain digits. The symbolic router handles the rest.
                # However, to be safe and allow the router to fallback to neural for anything,
                # we could process everything. But to save time/space, let's stick to the heuristic:
                # "If no Trigram match... check if token contains digits".
                # So we definitely need digit-containing tokens.
                if is_digit_token(token_text):
                    should_include = True
            else:
                # Train/Val Logic
                token_class = classes[idx]

                # Rule 1: Always include semiotic classes (non-PLAIN/PUNCT)
                if token_class not in ["PLAIN", "PUNCT"]:
                    should_include = True

                # Rule 2: Include PLAIN/PUNCT if they contain digits (e.g. "3D")
                elif is_digit_token(token_text):
                    should_include = True

                # Rule 3: Randomly sample PLAIN context (Anchoring)
                # Only for training, maybe less for validation to keep it focused?
                # The idea says "Context-Anchored Neural Dataset" for training.
                elif np.random.random() < config.plain_subset_ratio:
                    should_include = True

            if should_include:
                input_str = construct_window_input(
                    tokens_in, idx, config.context_window
                )

                sample = {
                    "input_text": input_str,
                }

                if has_targets:
                    sample["target_text"] = str(tokens_out[idx])
                    sample["class"] = classes[idx]
                else:
                    # For test, we need the ID to map back
                    sample["id"] = f"{sid}_{t_ids[idx]}"

                processed_samples.append(sample)
                selected_tokens += 1

    print(f"  Total tokens: {total_tokens}")
    print(f"  Selected tokens: {selected_tokens} ({selected_tokens/total_tokens:.4f})")

    return pd.DataFrame(processed_samples)


def load_and_process_data(
    config: Config, split: str = "train", load_cached: bool = True
) -> pd.DataFrame:
    """
    Main entry point for data loading. Handles caching.
    """
    # Determine paths
    if split == "train":
        meta_path = os.path.join(config.metadata_dir, "train.csv")
        cache_path = config.train_seq_path
    elif split == "val":
        meta_path = os.path.join(config.metadata_dir, "val.csv")
        cache_path = config.val_seq_path
    elif split == "test":
        meta_path = os.path.join(config.metadata_dir, "test.csv")
        cache_path = config.test_seq_path
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Try Load Cache
    if load_cached and os.path.exists(cache_path):
        print(f"Loading cached {split} sequences from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            print(f"  Loaded {len(df)} samples.")
            return df
        except Exception as e:
            print(f"  Failed to load cache: {e}. Reprocessing...")

    # 2. Process from Scratch
    print(f"Loading raw metadata from {meta_path}...")
    # Using specific dtypes to save memory
    dtype_dict = {"sentence_id": "int32", "token_id": "int32", "before": "object"}
    if split != "test":
        dtype_dict.update({"class": "category", "after": "object"})

    df_raw = pd.read_csv(meta_path, dtype=dtype_dict)

    # Handle NaNs
    df_raw["before"] = df_raw["before"].fillna("")
    if split != "test":
        df_raw["after"] = df_raw["after"].fillna("")
        df_raw["class"] = df_raw["class"].fillna("PLAIN")  # Default to PLAIN if missing

    # Process
    df_processed = process_split(df_raw, split, config)

    # 3. Save Cache
    print(f"Saving processed {split} data to {cache_path}...")
    df_processed.to_parquet(cache_path, index=False)

    return df_processed


def get_tokenizer(
    config: Config, train_df: Optional[pd.DataFrame] = None
) -> CharTokenizer:
    """
    Loads or creates the tokenizer.
    If tokenizer exists at config.tokenizer_path, loads it.
    Else, fits on train_df and saves.
    """
    tokenizer = CharTokenizer()

    if os.path.exists(config.tokenizer_path):
        print(f"Loading tokenizer from {config.tokenizer_path}...")
        tokenizer.load(config.tokenizer_path)
    else:
        print("Tokenizer not found. Fitting on training data...")
        if train_df is None:
            raise ValueError(
                "Tokenizer not found and no training data provided to fit."
            )

        # Collect all text for vocabulary
        # We need characters from both input (with context) and target
        texts = train_df["input_text"].tolist() + train_df["target_text"].tolist()
        tokenizer.fit_on_texts(texts)

        print(f"Saving tokenizer to {config.tokenizer_path}...")
        tokenizer.save(config.tokenizer_path)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    return tokenizer


def get_dataloader(
    config: Config, split: str = "train", load_cached: bool = True, shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    High-level function to get a DataLoader for a specific split.
    """
    # 1. Load Data
    df = load_and_process_data(config, split=split, load_cached=load_cached)

    # 2. Get Tokenizer
    # For train split, we might need to fit it. For others, we expect it to exist.
    if split == "train":
        tokenizer = get_tokenizer(config, train_df=df)
    else:
        tokenizer = get_tokenizer(config)  # Will load from disk

    # 3. Create Dataset
    is_test = split == "test"
    dataset = ContextAnchoredDataset(df, tokenizer, config, is_test=is_test)

    # 4. Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader
