import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.configuration import Config
from library.utilities import decode_text


class InsultDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.MAX_LEN, is_test=False):
        """
        PyTorch Dataset for Insult Detection.

        Args:
            df (pd.DataFrame): DataFrame containing 'Comment' and optionally 'Insult'.
            tokenizer: Transformer tokenizer instance.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.texts = df["Comment"].values.astype(str)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not self.is_test:
            self.labels = df["Insult"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Flatten to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            # Convert label to float for BCEWithLogitsLoss
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            item["target"] = label

        return item


def load_data(split="train", load_cached_data=True):
    """
    Loads data for a specific split, utilizing Parquet caching.

    Args:
        split (str): One of 'train', 'val', 'test', 'full'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cache_path = os.path.join(Config.CACHE_DIR, f"{split}_decoded.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception:
            # If cache load fails, proceed to regenerate
            pass

    # 2. Load raw data based on split
    if split == "train":
        df = pd.read_csv(Config.TRAIN_PATH)
    elif split == "val":
        df = pd.read_csv(Config.VAL_PATH)
    elif split == "test":
        df = pd.read_csv(Config.TEST_PATH)
    elif split == "full":
        # Concatenate Train and Val for full-data training
        df_train = pd.read_csv(Config.TRAIN_PATH)
        df_val = pd.read_csv(Config.VAL_PATH)
        df = pd.concat([df_train, df_val], axis=0, ignore_index=True)
    else:
        raise ValueError(
            f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'full'."
        )

    # 3. Apply deterministic processing (Text Decoding)
    if "Comment" in df.columns:
        df["Comment"] = df["Comment"].apply(decode_text)

    # 4. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def get_dataloader(df, tokenizer, batch_size, is_test=False, shuffle=True):
    """
    Creates a PyTorch DataLoader.

    Args:
        df (pd.DataFrame): Data.
        tokenizer: Transformer tokenizer.
        batch_size (int): Batch size.
        is_test (bool): Whether this is for testing.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader.
    """
    ds = InsultDataset(df, tokenizer, max_len=Config.MAX_LEN, is_test=is_test)

    # Drop last batch during training if shuffling to maintain consistent stats,
    # but not strictly necessary for small batches unless using BatchNorm (Transformers usually use LayerNorm).
    # We drop last if shuffling to ensure batch size consistency for gradient accumulation.
    drop_last = shuffle

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )

    return loader
