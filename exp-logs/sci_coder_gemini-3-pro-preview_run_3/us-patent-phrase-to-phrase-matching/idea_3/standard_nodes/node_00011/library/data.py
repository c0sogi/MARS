import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase
from library.config import Config


class PhraseDataset(Dataset):
    """
    Dataset class for Phrase Matching.
    Constructs input in the format: [CLS] context [SEP] anchor [SEP] target [SEP]
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-construct the input text to speed up __getitem__
        # Format: context + [SEP] + anchor + [SEP] + target
        # The tokenizer's add_special_tokens=True will add [CLS] at start and [SEP] at end.
        sep = tokenizer.sep_token
        self.texts = (
            self.df["context"].astype(str)
            + sep
            + self.df["anchor"].astype(str)
            + sep
            + self.df["target"].astype(str)
        ).tolist()

        if not self.is_test:
            self.labels = self.df["score"].values.astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize without padding (padding handled by collator)
        # We rely on add_special_tokens=True to add the CLS and final SEP
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if not self.is_test:
            item["labels"] = self.labels[idx]

        return item


class DynamicPaddingCollator:
    """
    Data Collator that dynamically pads the batch to the longest sequence length.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # batch is a list of dicts

        # Find max length in this batch
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for x in batch:
            # Pad input_ids
            ids = x["input_ids"]
            pad_len = max_len - len(ids)
            # Use tokenizer.pad_token_id
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)

            # Pad attention_mask (pad with 0)
            mask = x["attention_mask"]
            attention_masks.append(mask + [0] * pad_len)

            if "labels" in x:
                labels.append(x["labels"])

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }

        if labels:
            out["labels"] = torch.tensor(labels, dtype=torch.float)

        return out


def _load_and_cache_data(load_cached_data=True, debug=False):
    """
    Internal helper to load dataframes, handling caching logic.
    """
    # Define cache paths
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_cache.parquet")
    val_cache = os.path.join(cache_dir, "val_cache.parquet")
    test_cache = os.path.join(cache_dir, "test_cache.parquet")

    # Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading data from cache...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)

            if debug:
                df_train = df_train.sample(
                    n=min(100, len(df_train)), random_state=Config.seed
                ).reset_index(drop=True)
                df_val = df_val.sample(
                    n=min(50, len(df_val)), random_state=Config.seed
                ).reset_index(drop=True)
                df_test = df_test.sample(
                    n=min(50, len(df_test)), random_state=Config.seed
                ).reset_index(drop=True)

            return df_train, df_val, df_test

    # Process from scratch
    print("Loading data from metadata CSVs...")
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    # Save to cache (before debug slicing to ensure cache is full)
    print(f"Saving cache to {cache_dir}...")
    df_train.to_parquet(train_cache, index=False)
    df_val.to_parquet(val_cache, index=False)
    df_test.to_parquet(test_cache, index=False)

    if debug:
        print("Debug mode: subsampling data.")
        df_train = df_train.sample(
            n=min(100, len(df_train)), random_state=Config.seed
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(50, len(df_val)), random_state=Config.seed
        ).reset_index(drop=True)
        df_test = df_test.sample(
            n=min(50, len(df_test)), random_state=Config.seed
        ).reset_index(drop=True)

    return df_train, df_val, df_test


def prepare_loaders(
    tokenizer: PreTrainedTokenizerBase, load_cached_data=True, debug=Config.debug
):
    """
    Prepares DataLoaders for training, validation, and testing.

    Args:
        tokenizer: The tokenizer to use for encoding.
        load_cached_data (bool): Whether to attempt loading cached dataframes.
        debug (bool): Whether to run in debug mode (subsampled data).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    df_train, df_val, df_test = _load_and_cache_data(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Create Datasets
    train_dataset = PhraseDataset(df_train, tokenizer, Config.max_length, is_test=False)
    val_dataset = PhraseDataset(df_val, tokenizer, Config.max_length, is_test=False)
    test_dataset = PhraseDataset(df_test, tokenizer, Config.max_length, is_test=True)

    # 3. Create Collator
    collator = DynamicPaddingCollator(tokenizer)

    # 4. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
