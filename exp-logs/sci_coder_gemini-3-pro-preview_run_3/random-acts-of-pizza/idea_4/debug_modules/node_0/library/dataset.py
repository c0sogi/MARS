import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import load_data, get_cached_data


class PizzaDataset(Dataset):
    """
    Custom PyTorch Dataset for the Pizza Request data.
    """

    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = (
            torch.tensor(labels, dtype=torch.float) if labels is not None else None
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def _tokenize_helper(texts, tokenizer, max_len):
    """
    Helper to tokenize a list/series of texts into numpy arrays.
    """
    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",  # Return numpy arrays directly
    )
    return encoding["input_ids"], encoding["attention_mask"]


def _compute_transformer_data(train_df, val_df, test_df):
    """
    Computes tokenized data for all splits.
    Returns a flat dictionary of numpy arrays suitable for np.savez.
    """
    print(f"Initializing Tokenizer: {Config.BERT_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)

    # Prepare text lists (handle NaNs)
    train_texts = train_df[Config.TEXT_COL].fillna("").astype(str).tolist()
    val_texts = val_df[Config.TEXT_COL].fillna("").astype(str).tolist()
    test_texts = test_df[Config.TEXT_COL].fillna("").astype(str).tolist()

    print("Tokenizing Train set...")
    train_ids, train_mask = _tokenize_helper(train_texts, tokenizer, Config.MAX_SEQ_LEN)

    print("Tokenizing Validation set...")
    val_ids, val_mask = _tokenize_helper(val_texts, tokenizer, Config.MAX_SEQ_LEN)

    print("Tokenizing Test set...")
    test_ids, test_mask = _tokenize_helper(test_texts, tokenizer, Config.MAX_SEQ_LEN)

    # Extract labels
    train_labels = train_df[Config.TARGET_COL].values.astype(np.int32)
    val_labels = val_df[Config.TARGET_COL].values.astype(np.int32)

    # Return flat dictionary for npz storage
    return {
        "train_input_ids": train_ids,
        "train_attention_mask": train_mask,
        "train_labels": train_labels,
        "val_input_ids": val_ids,
        "val_attention_mask": val_mask,
        "val_labels": val_labels,
        "test_input_ids": test_ids,
        "test_attention_mask": test_mask,
    }


def create_dataloaders(load_cached_data=True, debug=False):
    """
    Main function to create DataLoaders for the Deep Learning branch.

    Args:
        load_cached_data (bool): Whether to use cached tokenized data.
        debug (bool): If True, uses a subset of data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load raw dataframes
    train_df = load_data(Config.TRAIN_PATH, debug=debug)
    val_df = load_data(Config.VAL_PATH, debug=debug)
    test_df = load_data(Config.TEST_PATH, debug=debug)

    suffix = "_debug" if debug else ""
    cache_name = f"transformer_data{suffix}"

    # Get tokenized data (either from cache or compute it)
    data = get_cached_data(
        _compute_transformer_data,
        cache_name,
        load_cached_data=load_cached_data,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # Create Datasets
    train_dataset = PizzaDataset(
        input_ids=data["train_input_ids"],
        attention_mask=data["train_attention_mask"],
        labels=data["train_labels"],
    )

    val_dataset = PizzaDataset(
        input_ids=data["val_input_ids"],
        attention_mask=data["val_attention_mask"],
        labels=data["val_labels"],
    )

    test_dataset = PizzaDataset(
        input_ids=data["test_input_ids"],
        attention_mask=data["test_attention_mask"],
        labels=None,  # No labels for test set
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BERT_TRAIN_PARAMS["batch_size"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader
