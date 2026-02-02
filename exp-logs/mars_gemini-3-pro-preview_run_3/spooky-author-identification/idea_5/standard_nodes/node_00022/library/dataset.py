import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from library.config import Config
from library.utils import seed_everything


class AuthorDataset(Dataset):
    """
    PyTorch Dataset for Author Identification (Supervised Classification).
    """

    def __init__(self, df, tokenizer, max_length=512, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'text' and optionally 'label' columns.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): If True, does not expect 'label' column.
        """
        self.texts = df[Config.TEXT_COL].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["label"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class MLMDataset(Dataset):
    """
    PyTorch Dataset for Masked Language Modeling (Domain Adaptation).
    """

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # For MLM, we just tokenize here. Masking is done by the DataCollator.
        # We do not pad here if we want dynamic padding in collator,
        # but for simplicity and consistency with AuthorDataset, we pad to max_length.
        # However, DataCollatorForLanguageModeling usually expects unpadded or handles padding.
        # We will return standard tokenized output.

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            # Special tokens mask is useful for the collator to avoid masking CLS/SEP
            "special_tokens_mask": encoding["special_tokens_mask"].flatten(),
        }


def load_data(load_cached_data=True):
    """
    Loads data from metadata CSVs.
    Implements caching using parquet files in the working directory.
    Maps string labels to integers.

    Returns:
        train_df, val_df, test_df (pd.DataFrame)
    """
    seed_everything(Config.SEED)

    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    # Label mapping
    label_map = {label: idx for idx, label in enumerate(Config.CLASS_LABELS)}

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            # print("Loading data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df

    # Process from scratch
    # print("Loading data from metadata and processing...")

    # Load raw CSVs
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    # Map labels
    if Config.TARGET_COL in train_df.columns:
        train_df["label"] = train_df[Config.TARGET_COL].map(label_map)

    if Config.TARGET_COL in val_df.columns:
        val_df["label"] = val_df[Config.TARGET_COL].map(label_map)

    # Save to cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def load_mlm_corpus(load_cached_data=True):
    """
    Loads and concatenates all available text (train + val + test) for MLM.
    Caches the result.
    """
    cache_path = os.path.join(Config.CACHE_DIR, "mlm_corpus.parquet")

    if load_cached_data and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        return df[Config.TEXT_COL].tolist()

    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Concatenate all texts
    all_texts = pd.concat(
        [train_df[Config.TEXT_COL], val_df[Config.TEXT_COL], test_df[Config.TEXT_COL]],
        ignore_index=True,
    )

    # Cache
    pd.DataFrame({Config.TEXT_COL: all_texts}).to_parquet(cache_path, index=False)

    return all_texts.tolist()


def create_dataloader(
    df, tokenizer, batch_size, is_test=False, shuffle=True, drop_last=False
):
    """
    Factory function to create a DataLoader for supervised tasks.
    """
    dataset = AuthorDataset(
        df=df, tokenizer=tokenizer, max_length=Config.MAX_LENGTH, is_test=is_test
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataloader


def create_mlm_dataloader(texts, tokenizer, batch_size):
    """
    Factory function to create a DataLoader for MLM.
    Uses DataCollatorForLanguageModeling for dynamic masking.
    """
    dataset = MLMDataset(texts=texts, tokenizer=tokenizer, max_length=Config.MAX_LENGTH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.MLM_MASK_PROB
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=data_collator,
    )

    return dataloader
