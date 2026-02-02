import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config


class CharTokenizer:
    """
    Character-level tokenizer that handles special tokens and class labels.
    """

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        # Ensure PAD is index 0
        self.specials = [
            Config.PAD_TOKEN,
            Config.SOS_TOKEN,
            Config.EOS_TOKEN,
            Config.SEP_TOKEN,
            Config.UNK_TOKEN,
        ]
        self.classes = Config.CLASS_TOKENS

        # Initialize vocabulary with specials and classes
        self._add_tokens(self.specials + self.classes)

    def _add_tokens(self, tokens):
        for t in tokens:
            if t not in self.char_to_id:
                idx = len(self.char_to_id)
                self.char_to_id[t] = idx
                self.id_to_char[idx] = t

    def fit_on_texts(self, texts):
        """
        Builds vocabulary from a list of strings.
        """
        unique_chars = set()
        for text in texts:
            unique_chars.update(str(text))

        # Sort for deterministic behavior
        sorted_chars = sorted(list(unique_chars))
        self._add_tokens(sorted_chars)

    def encode(self, text):
        """
        Converts a string to a list of token IDs.
        """
        unk_id = self.char_to_id[Config.UNK_TOKEN]
        return [self.char_to_id.get(c, unk_id) for c in str(text)]

    def get_id(self, token):
        """
        Returns ID for a specific token (e.g., special token or class token).
        """
        return self.char_to_id.get(token, self.char_to_id[Config.UNK_TOKEN])

    def decode(self, ids):
        """
        Converts a list of IDs back to a string.
        """
        return "".join([self.id_to_char.get(i, "") for i in ids])

    def save(self, path):
        """
        Saves the vocabulary to a JSON file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "char_to_id": self.char_to_id,
                    "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
                },
                f,
                indent=2,
            )

    def load(self, path):
        """
        Loads the vocabulary from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found at {path}")

        with open(path, "r") as f:
            data = json.load(f)
            self.char_to_id = data["char_to_id"]
            self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}

    def __len__(self):
        return len(self.char_to_id)


class NormalizationDataset(Dataset):
    """
    PyTorch Dataset for Text Normalization.
    """

    def __init__(self, data_path, tokenizer, split="train", max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.split = split

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        # Load data
        self.df = pd.read_parquet(data_path)

        # Ensure correct types
        text_cols = ["before", "prev", "next"]
        if "after" in self.df.columns:
            text_cols.append("after")
        if "class" in self.df.columns:
            text_cols.append("class")

        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Input Context
        prev_token = row["prev"]
        curr_token = row["before"]
        next_token = row["next"]

        sep_id = self.tokenizer.get_id(Config.SEP_TOKEN)

        # Construct Source: [prev] <SEP> [curr] <SEP> [next]
        src_ids = (
            self.tokenizer.encode(prev_token)
            + [sep_id]
            + self.tokenizer.encode(curr_token)
            + [sep_id]
            + self.tokenizer.encode(next_token)
        )

        # Truncate source if too long
        if len(src_ids) > self.max_len:
            src_ids = src_ids[: self.max_len]

        item = {"src": torch.tensor(src_ids, dtype=torch.long), "id": row["id"]}

        # Targets (Train/Val only)
        if self.split in ["train", "val"]:
            target_text = row["after"]
            class_label = row["class"]

            # Format class token: e.g. "DATE" -> "<DATE>"
            class_token = f"<{class_label}>"
            class_id = self.tokenizer.get_id(class_token)

            sos_id = self.tokenizer.get_id(Config.SOS_TOKEN)
            eos_id = self.tokenizer.get_id(Config.EOS_TOKEN)

            # Content: <CLASS> <SEP> text
            content_ids = [class_id, sep_id] + self.tokenizer.encode(target_text)

            # Decoder Input: <SOS> <CLASS> <SEP> text
            tgt_in = [sos_id] + content_ids

            # Target Output: <CLASS> <SEP> text <EOS>
            tgt_out = content_ids + [eos_id]

            # Truncate targets if necessary
            if len(tgt_in) > self.max_len:
                tgt_in = tgt_in[: self.max_len]
            if len(tgt_out) > self.max_len:
                tgt_out = tgt_out[: self.max_len]

            item["tgt_in"] = torch.tensor(tgt_in, dtype=torch.long)
            item["tgt_out"] = torch.tensor(tgt_out, dtype=torch.long)

        else:
            # For inference, we might need the raw 'before' token for heuristic fallbacks
            item["before"] = curr_token

        return item


def collate_fn(batch):
    """
    Custom collate function to pad sequences in the batch.
    """
    src_list = [item["src"] for item in batch]
    # Pad with 0 (assuming PAD_TOKEN index is 0)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=0
    )

    batch_out = {"src": src_padded, "id": [item["id"] for item in batch]}

    if "tgt_in" in batch[0]:
        tgt_in_list = [item["tgt_in"] for item in batch]
        tgt_out_list = [item["tgt_out"] for item in batch]

        tgt_in_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_in_list, batch_first=True, padding_value=0
        )
        tgt_out_padded = torch.nn.utils.rnn.pad_sequence(
            tgt_out_list, batch_first=True, padding_value=0
        )

        batch_out["tgt_in"] = tgt_in_padded
        batch_out["tgt_out"] = tgt_out_padded

    if "before" in batch[0]:
        batch_out["before"] = [item["before"] for item in batch]

    return batch_out


def _add_context(df):
    """
    Adds 'prev' and 'next' columns to the dataframe based on sentence boundaries.
    """
    # Create shifted columns
    df["prev"] = df["before"].shift(1)
    df["next"] = df["before"].shift(-1)

    # Identify sentence boundaries
    sent_ids = df["sentence_id"]
    is_start = sent_ids != sent_ids.shift(1)
    is_end = sent_ids != sent_ids.shift(-1)

    # Handle boundaries with SOS/EOS
    # Fill NaNs first to avoid type issues
    df["prev"] = df["prev"].fillna(Config.SOS_TOKEN)
    df["next"] = df["next"].fillna(Config.EOS_TOKEN)

    df.loc[is_start, "prev"] = Config.SOS_TOKEN
    df.loc[is_end, "next"] = Config.EOS_TOKEN

    return df


def prepare_neural_data(load_cached_data=True):
    """
    Orchestrates the data preparation pipeline:
    1. Builds/Loads Tokenizer.
    2. Processes Train/Val/Test data (add context, filter hard samples).
    3. Caches processed data to parquet.

    Returns:
        tokenizer: The loaded/built CharTokenizer instance.
    """
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # --- 1. Tokenizer ---
    tokenizer = CharTokenizer()
    if load_cached_data and os.path.exists(Config.TOKENIZER_PATH):
        print(f"Loading tokenizer from {Config.TOKENIZER_PATH}")
        tokenizer.load(Config.TOKENIZER_PATH)
    else:
        print("Building tokenizer from training data...")
        if not os.path.exists(Config.TRAIN_META):
            raise FileNotFoundError(f"Training metadata missing: {Config.TRAIN_META}")

        # Load full training data to build comprehensive vocab
        df_train = pd.read_parquet(Config.TRAIN_META)

        # Collect all text
        texts = pd.concat(
            [df_train["before"].astype(str), df_train["after"].astype(str)]
        ).tolist()

        tokenizer.fit_on_texts(texts)
        tokenizer.save(Config.TOKENIZER_PATH)
        print(f"Tokenizer saved. Vocab size: {len(tokenizer)}")
        del df_train, texts

    # --- 2. Process Datasets ---
    files_map = {
        "train": (Config.TRAIN_META, Config.PROCESSED_TRAIN),
        "val": (Config.VAL_META, Config.PROCESSED_VAL),
        "test": (Config.TEST_META, Config.PROCESSED_TEST),
    }

    for split, (input_path, output_path) in files_map.items():
        if load_cached_data and os.path.exists(output_path):
            print(f"Skipping {split} processing (cached at {output_path}).")
            continue

        print(f"Processing {split} data...")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file missing: {input_path}")

        df = pd.read_parquet(input_path)

        # Add context (prev/next)
        df = _add_context(df)

        if split in ["train", "val"]:
            # Filter Logic: Retain only "Hard" samples
            # Exclude PLAIN, PUNCT, and purely Alphabetic tokens
            mask_plain = df["class"] == "PLAIN"
            mask_punct = df["class"] == "PUNCT"
            mask_alpha = df["before"].astype(str).str.isalpha()

            # Keep if NOT (PLAIN or PUNCT or Alpha)
            df_filtered = df[~(mask_plain | mask_punct | mask_alpha)].copy()

            if Config.DEBUG:
                print(f"DEBUG mode: Subsampling {split} set to {Config.DEBUG_SIZE}")
                df_filtered = df_filtered.head(Config.DEBUG_SIZE)

            print(f"Saving filtered {split} data ({len(df_filtered)} samples)...")
            df_filtered.to_parquet(output_path, index=False)

        else:
            # Test set: Keep all samples but with context added
            print(f"Saving processed {split} data ({len(df)} samples)...")
            df.to_parquet(output_path, index=False)

    return tokenizer
