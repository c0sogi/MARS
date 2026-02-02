import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List

from library.configuration import Config
from library.utilities import set_seed


class InsultDataset(Dataset):
    """
    PyTorch Dataset for Insult Detection.
    Handles text tokenization, SVD feature retrieval, and target management (hard or soft labels).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        svd_features: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = Config.max_len,
        labels: Optional[Union[np.ndarray, List[float]]] = None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the 'Comment' column.
            svd_features (np.ndarray): Pre-computed SVD features aligned with the DataFrame.
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            labels (Optional[np.ndarray]): Targets for the dataset.
                                           Can be hard labels (0/1) or soft labels (probabilities).
                                           If None, attempts to extract 'Insult' from df.
        """
        self.df = df
        self.texts = df["Comment"].fillna("").astype(str).values
        self.svd_features = svd_features
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Handle Labels
        if labels is not None:
            self.labels = np.array(labels, dtype=float)
        elif "Insult" in df.columns:
            self.labels = df["Insult"].values.astype(float)
        else:
            self.labels = None

        # Validation
        assert len(self.texts) == len(self.svd_features), (
            f"Mismatch between text samples ({len(self.texts)}) and "
            f"SVD features ({len(self.svd_features)})"
        )
        if self.labels is not None:
            assert len(self.texts) == len(self.labels), (
                f"Mismatch between text samples ({len(self.texts)}) and "
                f"labels ({len(self.labels)})"
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by return_tensors='pt'
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # SVD Features
        svd_feat = torch.tensor(self.svd_features[idx], dtype=torch.float)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "svd_features": svd_feat,
        }

        # Targets
        if self.labels is not None:
            # Return as float for BCEWithLogitsLoss (works for both binary and soft targets)
            item["target"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def create_loaders(
    df: pd.DataFrame,
    svd_features: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    labels: Optional[np.ndarray] = None,
    shuffle: bool = False,
    num_workers: int = Config.num_workers,
    max_len: int = Config.max_len,
) -> DataLoader:
    """
    Factory function to create a DataLoader for the InsultDataset.

    Args:
        df (pd.DataFrame): Data containing text.
        svd_features (np.ndarray): SVD features.
        tokenizer (PreTrainedTokenizerBase): Tokenizer.
        batch_size (int): Batch size.
        labels (Optional[np.ndarray]): Explicit labels (hard or soft).
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        max_len (int): Max sequence length.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # Ensure reproducibility in dataloader shuffling
    if shuffle:
        set_seed(Config.seed)

    dataset = InsultDataset(
        df=df,
        svd_features=svd_features,
        tokenizer=tokenizer,
        max_len=max_len,
        labels=labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader
