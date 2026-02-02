import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import decode_text


class InsultDataset(Dataset):
    """
    Custom Dataset for Insult Detection.

    Handles:
    1. Tokenization of text.
    2. Ground truth labels (for training).
    3. Soft targets (for knowledge distillation).
    """

    def __init__(self, df, tokenizer, max_len, is_test=False, soft_targets=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the 'Comment' column and optionally 'Insult'.
            tokenizer: HuggingFace tokenizer instance.
            max_len (int): Maximum sequence length for tokenization.
            is_test (bool): If True, does not look for 'Insult' column.
            soft_targets (np.ndarray or list, optional): Soft labels for distillation.
        """
        self.texts = df["Comment"].values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.soft_targets = soft_targets

        if not self.is_test:
            # Ensure labels are float for BCEWithLogitsLoss
            self.labels = df["Insult"].values.astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        # Some tokenizers (like RoBERTa) don't use token_type_ids, but we return them for compatibility
        token_type_ids = inputs.get("token_type_ids", [0] * self.max_len)

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }

        # Add Ground Truth Labels
        if not self.is_test:
            out["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        # Add Soft Targets (Knowledge Distillation)
        if self.soft_targets is not None:
            out["soft_targets"] = torch.tensor(
                self.soft_targets[idx], dtype=torch.float
            )

        return out


def load_processed_data(
    config: Config, split: str, load_cached_data: bool = True
) -> pd.DataFrame:
    """
    Loads data for a specific split, applying decoding and caching.

    Args:
        config (Config): Configuration object.
        split (str): One of 'train', 'val', 'test', or 'train_full'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    cache_path = os.path.join(config.cache_dir, f"{split}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Basic validation to ensure cache isn't corrupted (e.g. empty)
            if not df.empty:
                return df
        except Exception:
            # If load fails, proceed to process from scratch
            pass

    # 2. Process from scratch
    if split == "train":
        df = pd.read_csv(config.train_path)
    elif split == "val":
        df = pd.read_csv(config.val_path)
    elif split == "test":
        df = pd.read_csv(config.test_path)
    elif split == "train_full":
        # Concatenate Train and Val for final student training
        df_train = pd.read_csv(config.train_path)
        df_val = pd.read_csv(config.val_path)
        df = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown split: {split}")

    # Apply text decoding
    if "Comment" in df.columns:
        df["Comment"] = df["Comment"].apply(decode_text)

    # Handle debugging subset
    if config.debug and config.subset_rows:
        df = df.head(config.subset_rows)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}. Error: {e}")

    return df


def get_dataloader(
    config: Config,
    split: str,
    tokenizer,
    shuffle: bool = False,
    drop_last: bool = False,
    soft_targets=None,
    load_cached_data: bool = True,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the specified split.

    Args:
        config (Config): Configuration object.
        split (str): Dataset split ('train', 'val', 'test', 'train_full').
        tokenizer: HuggingFace tokenizer.
        shuffle (bool): Whether to shuffle the data.
        drop_last (bool): Whether to drop the last incomplete batch.
        soft_targets (array-like, optional): Soft targets corresponding to the data.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Load Data
    df = load_processed_data(config, split, load_cached_data=load_cached_data)

    # Determine if this is a test set (no labels)
    # Note: 'test' split definitely has no labels.
    # 'train', 'val', 'train_full' have labels.
    is_test = split == "test"

    # Create Dataset
    dataset = InsultDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=config.max_len,
        is_test=is_test,
        soft_targets=soft_targets,
    )

    # Determine batch size
    batch_size = (
        config.train_batch_size
        if (split in ["train", "train_full"] and not is_test)
        else config.valid_batch_size
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return loader
