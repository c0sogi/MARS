import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import Config


class AuthorDataset(Dataset):
    """
    Custom Dataset for Author Identification.
    Handles tokenization and input formatting for Transformer models.
    """

    def __init__(self, texts, tokenizer, max_len, labels=None):
        """
        Args:
            texts (list or np.array): List of text sequences.
            tokenizer (transformers.PreTrainedTokenizer): HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            labels (list or np.array, optional): List of integer labels.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])

        # Tokenize
        # encode_plus handles [CLS], [SEP], padding, and truncation
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten tensors (encode_plus returns [1, seq_len], we need [seq_len])
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Include label if available (for training/validation)
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.long)
            item["label"] = label

        return item


def load_text_data(load_cached_data=True, debug=Config.DEBUG):
    """
    Loads text data from metadata CSVs.
    Applies label mapping and handles caching using Parquet files.

    Args:
        load_cached_data (bool): Whether to load from cache if available.
        debug (bool): Whether to use a subset of data.

    Returns:
        tuple: (train_texts, train_labels, val_texts, val_labels, test_texts, test_ids)
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    suffix = "_debug" if debug else ""

    train_cache = os.path.join(cache_dir, f"train_processed{suffix}.parquet")
    val_cache = os.path.join(cache_dir, f"val_processed{suffix}.parquet")
    test_cache = os.path.join(cache_dir, f"test_processed{suffix}.parquet")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        try:
            print(f"Loading processed text data from {cache_dir}...")
            df_train = pd.read_parquet(train_cache)
            df_val = pd.read_parquet(val_cache)
            df_test = pd.read_parquet(test_cache)

            # Extract arrays
            train_texts = df_train["text"].values
            train_labels = df_train["label"].values

            val_texts = df_val["text"].values
            val_labels = df_val["label"].values

            test_texts = df_test["text"].values
            test_ids = df_test["id"].values

            return (
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                test_texts,
                test_ids,
            )
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing...")

    # 2. Load Raw Data
    print("Loading raw metadata for text processing...")
    try:
        df_train = pd.read_csv(Config.TRAIN_DATA_PATH)
        df_val = pd.read_csv(Config.VAL_DATA_PATH)
        df_test = pd.read_csv(Config.TEST_DATA_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required metadata files not found: {e}")

    # Handle Debug Mode
    if debug:
        print(f"DEBUG mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Process Data
    # Map labels to integers
    if "author" in df_train.columns:
        df_train["label"] = df_train["author"].map(Config.LABEL_MAP)
    if "author" in df_val.columns:
        df_val["label"] = df_val["author"].map(Config.LABEL_MAP)

    # Ensure text is string and handle NaNs
    df_train["text"] = df_train["text"].fillna("").astype(str)
    df_val["text"] = df_val["text"].fillna("").astype(str)
    df_test["text"] = df_test["text"].fillna("").astype(str)

    # 4. Save to Cache
    print(f"Saving processed text data to {cache_dir}...")
    # Select relevant columns for caching
    df_train[["text", "label"]].to_parquet(train_cache, index=False)
    df_val[["text", "label"]].to_parquet(val_cache, index=False)
    df_test[["id", "text"]].to_parquet(test_cache, index=False)

    # 5. Return
    return (
        df_train["text"].values,
        df_train["label"].values,
        df_val["text"].values,
        df_val["label"].values,
        df_test["text"].values,
        df_test["id"].values,
    )
