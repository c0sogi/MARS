import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import string
from library.config import Config


class Tokenizer:
    def __init__(self):
        self.char_map = Config.get_char_map()
        self.idx_map = {v: k for k, v in self.char_map.items()}
        self.class_map = Config.get_class_map()

    def tokenize(self, text):
        """Converts string to list of character IDs."""
        if not isinstance(text, str):
            text = str(text)
        return [self.char_map.get(c, Config.UNK_IDX) for c in text]

    def detokenize(self, indices):
        """Converts list of IDs to string."""
        result = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx in [Config.PAD_IDX, Config.SOS_IDX, Config.EOS_IDX, Config.SEP_IDX]:
                continue
            char = self.idx_map.get(idx, Config.UNK_TOKEN)
            result.append(char)
        return "".join(result)

    def get_factored_features(self, text):
        """
        Returns tuple of lists: (char_ids, case_ids, type_ids).
        """
        if not isinstance(text, str):
            text = str(text)

        char_ids = []
        case_ids = []
        type_ids = []

        for c in text:
            # Char ID
            char_ids.append(self.char_map.get(c, Config.UNK_IDX))

            # Case ID
            if c.isupper():
                case_ids.append(2)  # UPPER
            elif c.islower():
                case_ids.append(3)  # LOWER
            else:
                case_ids.append(1)  # NONE

            # Type ID
            if c.isdigit():
                type_ids.append(2)  # DIGIT
            elif c.isalpha():
                type_ids.append(3)  # LETTER
            elif c in string.punctuation or c in [
                "$",
                "£",
                "€",
                "¥",
            ]:  # Basic symbols check
                type_ids.append(4)  # SYMBOL
            else:
                type_ids.append(1)  # OTHER

        return char_ids, case_ids, type_ids


class TextNormalizerDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, mode="train"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode  # 'train', 'val', 'test'

        # Pre-compute class indices if available
        if "class" in self.df.columns:
            self.class_indices = (
                self.df["class"]
                .map(self.tokenizer.class_map)
                .fillna(0)
                .astype(int)
                .values
            )
        else:
            self.class_indices = np.zeros(len(self.df), dtype=int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Get Context
        prev_text = row.get("prev", "")
        curr_text = row.get("before", "")
        next_text = row.get("next", "")

        # Handle NaN
        if pd.isna(prev_text):
            prev_text = ""
        if pd.isna(curr_text):
            curr_text = ""
        if pd.isna(next_text):
            next_text = ""

        # 2. Construct Source Sequence with Factors
        # Format: prev <SEP> curr <SEP> next

        p_c, p_case, p_type = self.tokenizer.get_factored_features(prev_text)
        c_c, c_case, c_type = self.tokenizer.get_factored_features(curr_text)
        n_c, n_case, n_type = self.tokenizer.get_factored_features(next_text)

        # Separator Features
        # Char: SEP_IDX
        # Case: NONE (1)
        # Type: OTHER (1)
        sep_char = [Config.SEP_IDX]
        sep_case = [1]
        sep_type = [1]

        src_char = p_c + sep_char + c_c + sep_char + n_c
        src_case = p_case + sep_case + c_case + sep_case + n_case
        src_type = p_type + sep_type + c_type + sep_type + n_type

        # Truncate/Pad Source
        if len(src_char) > self.max_len:
            # Truncate end
            src_char = src_char[: self.max_len]
            src_case = src_case[: self.max_len]
            src_type = src_type[: self.max_len]

        pad_len = self.max_len - len(src_char)
        if pad_len > 0:
            src_char += [Config.PAD_IDX] * pad_len
            src_case += [0] * pad_len  # PAD case
            src_type += [0] * pad_len  # PAD type

        # 3. Construct Target Sequence (if available)
        tgt_ids = []
        if self.mode != "test":
            after_text = row.get("after", "")
            if pd.isna(after_text):
                after_text = ""

            raw_tgt = self.tokenizer.tokenize(after_text)
            tgt_ids = [Config.SOS_IDX] + raw_tgt + [Config.EOS_IDX]

            if len(tgt_ids) > self.max_len:
                tgt_ids = tgt_ids[: self.max_len]  # Truncate
                tgt_ids[-1] = Config.EOS_IDX  # Ensure EOS

            pad_len_tgt = self.max_len - len(tgt_ids)
            if pad_len_tgt > 0:
                tgt_ids += [Config.PAD_IDX] * pad_len_tgt

        # 4. Class Label
        class_idx = self.class_indices[idx]

        return {
            "src_char": torch.tensor(src_char, dtype=torch.long),
            "src_case": torch.tensor(src_case, dtype=torch.long),
            "src_type": torch.tensor(src_type, dtype=torch.long),
            "tgt": (
                torch.tensor(tgt_ids, dtype=torch.long)
                if tgt_ids
                else torch.tensor([], dtype=torch.long)
            ),
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "id": row.get("id", ""),
        }


def process_context(df):
    """
    Adds 'prev' and 'next' columns based on sentence_id and token_id.
    Assumes df is sorted by sentence_id, token_id.
    """
    # Ensure sorting
    df = df.sort_values(["sentence_id", "token_id"])

    # Shift
    df["prev"] = df["before"].shift(1).fillna("")
    df["next"] = df["before"].shift(-1).fillna("")

    # Mask boundaries
    # If sentence_id changes, prev should be empty for the new sentence start
    sent_diff = df["sentence_id"].diff()
    mask_start = sent_diff != 0
    df.loc[mask_start, "prev"] = ""

    # If sentence_id changes for next (look ahead), next should be empty for old sentence end
    sent_diff_next = df["sentence_id"].diff(-1)
    mask_end = sent_diff_next != 0
    df.loc[mask_end, "next"] = ""

    return df


def load_data(split="train", load_cached_data=True):
    """
    Loads data, processes context, and applies soft filtering if split='train'.
    """
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    cache_path = os.path.join(Config.PROCESSED_DATA_DIR, f"{split}_processed.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        return pd.read_parquet(cache_path)

    print(f"Processing {split} data from scratch...")

    # Load Metadata
    if split == "train":
        path = Config.TRAIN_META_PATH
    elif split == "val":
        path = Config.VAL_META_PATH
    else:
        path = Config.TEST_META_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_parquet(path)

    # Ensure string types
    df["before"] = df["before"].astype(str)
    if "after" in df.columns:
        df["after"] = df["after"].astype(str)

    # Process Context
    # This operation is heavy, so we cache it.
    df = process_context(df)

    # Soft Filtering (Only for Train)
    if split == "train":
        print("Applying Soft Filtering...")
        # Hard samples: Not PLAIN or PUNCT
        hard_mask = ~df["class"].isin(["PLAIN", "PUNCT"])
        df_hard = df[hard_mask]

        # Easy samples: PLAIN or PUNCT
        easy_mask = df["class"].isin(["PLAIN", "PUNCT"])
        df_easy = df[easy_mask]

        # Sample easy
        if Config.SOFT_FILTER_RATIO < 1.0:
            df_easy = df_easy.sample(
                frac=Config.SOFT_FILTER_RATIO, random_state=Config.SEED
            )

        # Combine
        df = pd.concat([df_hard, df_easy], axis=0)
        # Shuffle
        df = df.sample(frac=1.0, random_state=Config.SEED).reset_index(drop=True)

    # Save to cache
    print(f"Saving processed {split} data to {cache_path}...")
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloader(
    split="train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
):
    df = load_data(split, load_cached_data)
    tokenizer = Tokenizer()

    mode = "test" if split == "test" else "train"

    dataset = TextNormalizerDataset(df, tokenizer, max_len=Config.MAX_LEN, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return loader
