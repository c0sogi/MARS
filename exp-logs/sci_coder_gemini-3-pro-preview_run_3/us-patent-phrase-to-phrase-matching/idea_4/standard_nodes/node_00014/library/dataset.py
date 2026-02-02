import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


class PearsonDataset(Dataset):
    """
    Dataset class for the Phrase Matching task.
    Constructs input sequences in the format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    Provides both continuous scores (for regression) and discrete labels (for classification).
    """

    def __init__(self, df, tokenizer, max_len=Config.max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-compute input texts to ensure deterministic behavior and efficiency.
        # We manually construct the string with separators.
        # Note: We assume the tokenizer is capable of parsing the special token string
        # or that the user accepts the string representation.
        # For DebertaV3, we construct the text and rely on the tokenizer to add CLS/SEP
        # at the boundaries, while we insert the internal SEPs.
        sep = tokenizer.sep_token
        self.texts = (
            df["context"].astype(str)
            + sep
            + df["anchor"].astype(str)
            + sep
            + df["target"].astype(str)
        ).tolist()

        if not self.is_test:
            self.scores = df["score"].values
            # Map scores to discrete class indices:
            # 0.00 -> 0
            # 0.25 -> 1
            # 0.50 -> 2
            # 0.75 -> 3
            # 1.00 -> 4
            self.labels = (self.scores * 4).round().astype(int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize the constructed string
        # add_special_tokens=True adds [CLS] at start and [SEP] at end
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,  # Dynamic padding is handled in DataCollator
            truncation=True,
            return_attention_mask=True,
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if not self.is_test:
            item["score"] = float(self.scores[idx])
            item["label_idx"] = int(self.labels[idx])
        else:
            # Dummy targets for inference
            item["score"] = 0.0
            item["label_idx"] = 0

        return item


class DataCollator:
    """
    Collator to handle dynamic padding of batches.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # Determine max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Use tokenizer's pad_token_id (usually 0 for DebertaV3)
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            # Attention mask is 0 for padded tokens
            padded_attention_mask.append(mask + [0] * pad_len)

        output = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

        # Stack targets if present
        if "score" in batch[0]:
            output["score"] = torch.tensor(
                [item["score"] for item in batch], dtype=torch.float
            )
            output["label_idx"] = torch.tensor(
                [item["label_idx"] for item in batch], dtype=torch.long
            )

        return output


def prepare_data(load_cached_data=True):
    """
    Loads Train, Validation, and Test dataframes.
    Implements caching using Parquet to speed up loading on subsequent runs.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache first.

    Returns:
        tuple: (df_train, df_val, df_test)
    """
    Config.create_output_dir()
    cache_dir = Config.output_dir

    train_cache = os.path.join(cache_dir, "train_cache.parquet")
    val_cache = os.path.join(cache_dir, "val_cache.parquet")
    test_cache = os.path.join(cache_dir, "test_cache.parquet")

    # Check if cache exists
    cache_exists = (
        os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    )

    if load_cached_data and cache_exists:
        print(f"Loading data from cache at {cache_dir}...")
        df_train = pd.read_parquet(train_cache)
        df_val = pd.read_parquet(val_cache)
        df_test = pd.read_parquet(test_cache)
    else:
        print("Loading data from metadata CSVs...")
        # Load from the metadata paths defined in Config
        df_train = pd.read_csv(Config.train_path)
        df_val = pd.read_csv(Config.val_path)
        df_test = pd.read_csv(Config.test_path)

        # Save to cache
        print(f"Saving data to cache at {cache_dir}...")
        df_train.to_parquet(train_cache, index=False)
        df_val.to_parquet(val_cache, index=False)
        df_test.to_parquet(test_cache, index=False)

    # Handle Debug Mode
    if Config.debug:
        print("Debug mode enabled: Sampling small subset of data.")
        df_train = df_train.sample(
            n=min(100, len(df_train)), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(50, len(df_val)), random_state=Config.seed
        ).reset_index(drop=True)
        # We usually keep test set intact or sample it similarly for pipeline testing
        df_test = df_test.sample(
            n=min(50, len(df_test)), random_state=Config.seed
        ).reset_index(drop=True)

    return df_train, df_val, df_test
