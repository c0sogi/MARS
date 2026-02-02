import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import set_seed


class TweetDataset(Dataset):
    """
    Dataset class for Sentiment Extraction.
    Handles tokenization and target generation (mapping character spans to token indices).
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["text"].values
        self.sentiments = df["sentiment"].values
        self.text_ids = df["textID"].values

        if not self.is_test:
            self.selected_texts = df["selected_text"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Retrieve and Preprocess Text
        text = str(self.texts[idx])
        # Normalize whitespace (critical for alignment consistency)
        text = " ".join(text.split())

        # 2. Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(encoding["token_type_ids"], dtype=torch.long)
        offsets = torch.tensor(encoding["offset_mapping"], dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "offsets": offsets,
            "text": text,
            "sentiment": self.sentiments[idx],
            "textID": self.text_ids[idx],
        }

        # 3. Generate Targets (Train/Val only)
        if not self.is_test:
            selected_text = str(self.selected_texts[idx])
            selected_text = " ".join(selected_text.split())

            # Find character start/end
            start_char = text.find(selected_text)

            # Default to (0, 0) if not found (should be filtered out by prepare_data)
            start_position = 0
            end_position = 0

            if start_char != -1:
                end_char = start_char + len(selected_text)

                # Create character mask
                char_mask = np.zeros(len(text))
                char_mask[start_char:end_char] = 1

                # Project to token mask
                target_idx = []
                for i, (off_start, off_end) in enumerate(encoding["offset_mapping"]):
                    # Skip padding/special tokens that have no width or are masked
                    if encoding["attention_mask"][i] == 0 or off_start == off_end:
                        continue

                    # Check overlap between token span and character mask
                    # We consider a token part of the target if it overlaps at all
                    if np.sum(char_mask[off_start:off_end]) > 0:
                        target_idx.append(i)

                if len(target_idx) > 0:
                    start_position = target_idx[0]
                    end_position = target_idx[-1]

            item["start_positions"] = torch.tensor(start_position, dtype=torch.long)
            item["end_positions"] = torch.tensor(end_position, dtype=torch.long)
            item["selected_text"] = selected_text

        return item


def get_tokenizer(config):
    """
    Initializes and returns the tokenizer based on config.
    """
    return AutoTokenizer.from_pretrained(config.model_name)


def prepare_data(config, load_cached_data=True):
    """
    Prepares the training data pipeline.

    1. Checks for cached parquet file.
    2. If not cached:
       - Loads train and val metadata.
       - Merges them for full CV.
       - Filters out NaNs.
       - Filters out 'neutral' rows (if config.train_on_neutral is False).
       - Filters out rows where selected_text cannot be aligned to text.
       - Creates Stratified K-Folds.
       - Saves to cache.

    Args:
        config: Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with 'fold' column.
    """
    cache_path = os.path.join(config.output_dir, "train_folds.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached training data from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Processing training data from scratch...")

    # 2. Load Metadata
    # We combine provided train and val splits to create our own CV folds
    df_train = pd.read_csv(config.train_path)
    df_val = pd.read_csv(config.val_path)
    df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

    # 3. Basic Cleaning
    df = df.dropna(subset=["text", "selected_text", "sentiment"])

    # Create temporary clean columns for filtering logic
    # (Dataset class repeats this, but we need it here to filter impossible targets)
    df["text_clean"] = df["text"].apply(lambda x: " ".join(str(x).split()))
    df["sel_clean"] = df["selected_text"].apply(lambda x: " ".join(str(x).split()))

    # 4. Filter Neutrals
    if not config.train_on_neutral:
        initial_count = len(df)
        df = df[df["sentiment"] != "neutral"].copy()
        print(
            f"Filtered out {initial_count - len(df)} neutral rows (Config.train_on_neutral=False)."
        )

    # 5. Filter Impossible Targets (Alignment Check)
    if config.filter_impossible_targets:
        initial_count = len(df)

        def check_alignment(row):
            return row["text_clean"].find(row["sel_clean"]) != -1

        df = df[df.apply(check_alignment, axis=1)].copy()
        print(f"Filtered out {initial_count - len(df)} rows with impossible alignment.")

    # 6. Stratified K-Fold
    # Reset index after filtering to ensure continuous indices for folding
    df = df.reset_index(drop=True)

    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(df, df["sentiment"])):
        df.loc[val_idx, "fold"] = fold

    # 7. Save Cache
    # Drop temp columns
    df = df.drop(columns=["text_clean", "sel_clean"])

    os.makedirs(config.output_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Data processed and saved to {cache_path}")

    return df


def prepare_test_data(config):
    """
    Loads the test dataset from metadata.
    """
    if not os.path.exists(config.test_path):
        raise FileNotFoundError(f"Test file not found at {config.test_path}")
    return pd.read_csv(config.test_path)
