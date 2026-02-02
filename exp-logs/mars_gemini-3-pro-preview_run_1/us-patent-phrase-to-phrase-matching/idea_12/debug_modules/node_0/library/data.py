import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedGroupKFold
from library.config import Config


class PearsonDataset(Dataset):
    """
    Dataset class for Phrase Similarity.
    Implements Native Pair Encoding with Atomic Context Embeddings.
    """

    def __init__(self, df, tokenizer, max_length, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        # Pre-extract columns for efficiency
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context"].values
        self.ids = df["id"].values

        if self.mode != "test":
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Native Pair Encoding with Atomic Context
        # Segment A: [Context_Token] [Anchor]
        # Segment B: [Target]
        # We construct the string and let the tokenizer handle the [CLS]/[SEP] structure.
        # Since 'context' is added as a special token, the tokenizer will map it to a single ID.
        text_a = f"{context} {anchor}"
        text_b = target

        inputs = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs:
            item["token_type_ids"] = inputs["token_type_ids"].squeeze(0)

        if self.mode != "test":
            # Regression target
            label = torch.tensor(self.scores[idx], dtype=torch.float)
            item["labels"] = label
        else:
            # For inference, we need the ID to format the submission
            item["id"] = self.ids[idx]

        return item


def get_tokenizer_and_resize(config: Config):
    """
    Loads the tokenizer and adds unique CPC contexts as special tokens.
    Returns the updated tokenizer.
    """
    # Load all available data to find the complete set of unique contexts
    df_train = pd.read_csv(config.train_path)
    df_val = pd.read_csv(config.val_path)
    df_test = pd.read_csv(config.test_path)

    # Extract unique contexts
    all_contexts = pd.concat(
        [df_train["context"], df_val["context"], df_test["context"]]
    ).unique()

    # Convert to list of strings
    special_tokens = list(all_contexts)

    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Add contexts as new special tokens
    # This ensures "A47" is treated as a single token, not split into "A" and "47"
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    return tokenizer


def _get_folded_data(config: Config, load_cached_data: bool = True):
    """
    Internal helper to load or create the folded dataset using Stratified Group K-Fold.
    """
    cache_path = os.path.join(config.output_dir, "train_folds.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached folds from {cache_path}")
        return pd.read_parquet(cache_path)

    # 2. Compute folds if cache missing or forced reload
    print("Computing Stratified Group K-Fold splits...")
    df = pd.read_csv(config.train_path)

    # Debug subsampling
    if config.debug:
        df = df.head(config.debug_sample_size).copy()

    # Initialize Splitter
    sgkf = StratifiedGroupKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )

    # Create a string representation of score for stratification
    # (StratifiedGroupKFold expects discrete classes)
    df["score_strat"] = df["score"].astype(str)

    df["fold"] = -1

    # Perform Split
    # groups=df['anchor']: Prevents same anchor appearing in train and val (leakage)
    # y=df['score_strat']: Maintains class distribution balance
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, df["score_strat"], groups=df["anchor"])
    ):
        df.loc[val_idx, "fold"] = fold

    # Save to cache
    os.makedirs(config.output_dir, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def make_dataloaders(
    config: Config, tokenizer, fold: int = 0, load_cached_data: bool = True
):
    """
    Creates train and validation DataLoaders for a specific fold.
    """
    # Get dataframe with fold assignments
    df = _get_folded_data(config, load_cached_data)

    # Split data
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    if config.debug:
        print(
            f"[DEBUG] Fold {fold}: Train samples={len(train_df)}, Val samples={len(val_df)}"
        )

    # Create Datasets
    train_dataset = PearsonDataset(train_df, tokenizer, config.max_length, mode="train")
    val_dataset = PearsonDataset(val_df, tokenizer, config.max_length, mode="val")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def make_test_dataloader(config: Config, tokenizer):
    """
    Creates a DataLoader for the test set (inference).
    """
    df_test = pd.read_csv(config.test_path)

    if config.debug:
        df_test = df_test.head(config.debug_sample_size).copy()

    test_dataset = PearsonDataset(df_test, tokenizer, config.max_length, mode="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    return test_loader
