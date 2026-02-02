import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class EssayDataset(Dataset):
    """
    Dataset class for Essay Scoring.
    Handles tokenization and formatting of input data for the model.
    """

    def __init__(self, df, tokenizer, max_length=1024, include_labels=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'full_text' and optionally 'score'.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            include_labels (bool): Whether to return labels (scores).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels

        # Ensure text is string
        self.texts = df["full_text"].astype(str).values

        if self.include_labels:
            # Ensure score is float for regression
            self.labels = df["score"].astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize without padding (padding is handled in Collate)
        # Truncation is applied to ensure we don't exceed model capacity
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,  # Return python lists for Collate to handle
            add_special_tokens=True,
        )

        item = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if self.include_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class Collate:
    """
    Collator for dynamic padding.
    Pads batches to the length of the longest sequence in the batch,
    rather than the global max_length, to save compute.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract inputs
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        # Determine max length in this batch
        batch_max_len = max(len(ids) for ids in input_ids)

        # Pad inputs
        # We manually pad because tokenizer.pad accepts a dict or list of dicts,
        # but here we have lists of lists.
        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_length = batch_max_len - len(ids)

            # Pad input_ids with pad_token_id
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_input_ids.append(padded_ids)

            # Pad attention_mask with 0
            padded_mask = mask + [0] * padding_length
            padded_attention_mask.append(padded_mask)

        # Convert to tensors
        batch_out = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

        # Handle labels if present
        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            batch_out["labels"] = torch.stack(labels)

        return batch_out


def get_mlm_data(tokenizer, load_cached_data=True):
    """
    Prepares data for Domain Adaptive Pre-training (MLM).
    Combines Train and Test text data.

    Args:
        tokenizer: Tokenizer for the dataset.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        EssayDataset: Dataset containing combined text without labels.
    """
    cache_path = os.path.join(Config.output_dir, "mlm_data.parquet")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached MLM data from {cache_path}")
        df_combined = pd.read_parquet(cache_path)
    else:
        # 2. Compute/Process data
        print("Creating MLM data from scratch...")

        # Load metadata
        if not os.path.exists(Config.train_path) or not os.path.exists(
            Config.test_path
        ):
            raise FileNotFoundError(
                "Metadata files not found. Ensure metadata generation was successful."
            )

        df_train = pd.read_csv(Config.train_path)
        df_test = pd.read_csv(Config.test_path)

        # Concatenate full_text from both sources
        texts = pd.concat(
            [df_train["full_text"], df_test["full_text"]], axis=0
        ).reset_index(drop=True)
        df_combined = pd.DataFrame({"full_text": texts})

        # Save to cache
        os.makedirs(Config.output_dir, exist_ok=True)
        df_combined.to_parquet(cache_path, index=False)
        print(f"Saved MLM data to {cache_path}")

    # Return Dataset
    # For MLM, we don't need labels.
    return EssayDataset(
        df_combined, tokenizer, max_length=Config.max_length, include_labels=False
    )


def get_supervised_data(tokenizer, load_cached_data=True):
    """
    Prepares data for Supervised Fine-Tuning.
    Loads Train and Validation splits.

    Args:
        tokenizer: Tokenizer for the dataset.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    train_cache_path = os.path.join(Config.output_dir, "train_supervised.parquet")
    val_cache_path = os.path.join(Config.output_dir, "val_supervised.parquet")

    # Load Train
    if load_cached_data and os.path.exists(train_cache_path):
        print(f"Loading cached Train data from {train_cache_path}")
        df_train = pd.read_parquet(train_cache_path)
    else:
        print("Loading Train data from metadata...")
        df_train = pd.read_csv(Config.train_path)
        os.makedirs(Config.output_dir, exist_ok=True)
        df_train.to_parquet(train_cache_path, index=False)

    # Load Val
    if load_cached_data and os.path.exists(val_cache_path):
        print(f"Loading cached Val data from {val_cache_path}")
        df_val = pd.read_parquet(val_cache_path)
    else:
        print("Loading Validation data from metadata...")
        df_val = pd.read_csv(Config.val_path)
        os.makedirs(Config.output_dir, exist_ok=True)
        df_val.to_parquet(val_cache_path, index=False)

    train_dataset = EssayDataset(
        df_train, tokenizer, max_length=Config.max_length, include_labels=True
    )
    val_dataset = EssayDataset(
        df_val, tokenizer, max_length=Config.max_length, include_labels=True
    )

    return train_dataset, val_dataset


def get_test_data(tokenizer, load_cached_data=True):
    """
    Prepares data for Inference.

    Args:
        tokenizer: Tokenizer for the dataset.
        load_cached_data (bool): Whether to load from parquet cache.

    Returns:
        EssayDataset: Dataset containing test data.
    """
    test_cache_path = os.path.join(Config.output_dir, "test_supervised.parquet")

    if load_cached_data and os.path.exists(test_cache_path):
        print(f"Loading cached Test data from {test_cache_path}")
        df_test = pd.read_parquet(test_cache_path)
    else:
        print("Loading Test data from metadata...")
        df_test = pd.read_csv(Config.test_path)
        os.makedirs(Config.output_dir, exist_ok=True)
        df_test.to_parquet(test_cache_path, index=False)

    return EssayDataset(
        df_test, tokenizer, max_length=Config.max_length, include_labels=False
    )
