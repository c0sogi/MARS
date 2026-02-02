import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from library.config import Config
from library.cpc_mapping import get_cpc_texts


class PhraseDataset(Dataset):
    """
    Dataset class for Semantic Similarity.
    Constructs the Tri-Segment input: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer, max_length: int, is_test: bool = False
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to avoid dataframe indexing overhead in __getitem__
        self.contexts = df["context_text"].values
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.ids = df["id"].values

        if not self.is_test:
            self.labels = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context_text = self.contexts[idx]
        anchor = self.anchors[idx]
        target = self.targets[idx]

        # Tri-Segment Input Construction
        # Segment A: Context + [SEP] + Anchor
        # Segment B: Target
        # The tokenizer will handle the final structure: [CLS] SegA [SEP] SegB [SEP]
        # Resulting in: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        segment_a = str(context_text) + self.tokenizer.sep_token + str(anchor)
        segment_b = str(target)

        inputs = self.tokenizer(
            segment_a,
            segment_b,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,  # Important for DeBERTa/BERT
        )

        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }

        if "token_type_ids" in inputs:
            item["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if self.is_test:
            item["id"] = self.ids[idx]
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def get_data_splits(cfg: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads training data, maps CPC codes, and performs Stratified Group K-Fold splitting.
    Uses caching to store the processed dataframe with folds.
    """
    os.makedirs(cfg.working_dir, exist_ok=True)
    cache_path = os.path.join(cfg.working_dir, "train_folds.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading train data with folds from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    print("Processing training data and generating folds...")

    # Load metadata
    df = pd.read_csv(cfg.train_path)

    # Debug mode: subset
    if cfg.debug:
        print(f"DEBUG: Subsetting train data to {cfg.debug_sample_size} rows.")
        df = df.head(cfg.debug_sample_size).copy()

    # Map CPC Context
    cpc_texts = get_cpc_texts(cfg, load_cached_data=load_cached_data)
    df["context_text"] = df["context"].map(cpc_texts)

    # Fill missing context texts if any (though unlikely with proper mapping)
    df["context_text"] = df["context_text"].fillna("")

    # Create Folds
    # Group by 'anchor' to prevent leakage
    # Stratify by 'score' (discrete)
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )

    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, df["score"].astype(str), groups=df["anchor"])
    ):
        df.loc[val_idx, "fold"] = fold

    # 3. Save Cache
    print(f"Saving processed train data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def preprocess_test_data(cfg: Config, load_cached_data: bool = True) -> pd.DataFrame:
    """
    Loads test data and maps CPC codes.
    Uses caching.
    """
    os.makedirs(cfg.working_dir, exist_ok=True)
    cache_path = os.path.join(cfg.working_dir, "test_processed.parquet")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading processed test data from cache: {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute
    print("Processing test data...")
    df = pd.read_csv(cfg.test_path)

    # Debug mode
    if cfg.debug:
        print(f"DEBUG: Subsetting test data to {cfg.debug_sample_size} rows.")
        df = df.head(cfg.debug_sample_size).copy()

    # Map CPC Context
    cpc_texts = get_cpc_texts(cfg, load_cached_data=load_cached_data)
    df["context_text"] = df["context"].map(cpc_texts)
    df["context_text"] = df["context_text"].fillna("")

    # 3. Save Cache
    print(f"Saving processed test data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def prepare_loaders(fold: int, train_df: pd.DataFrame, tokenizer, cfg: Config):
    """
    Creates training and validation DataLoaders for a specific fold.
    """
    # Split data
    df_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    df_valid = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Create Datasets
    train_dataset = PhraseDataset(
        df_train, tokenizer=tokenizer, max_length=cfg.max_length, is_test=False
    )
    valid_dataset = PhraseDataset(
        df_valid, tokenizer=tokenizer, max_length=cfg.max_length, is_test=False
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def prepare_test_loader(test_df: pd.DataFrame, tokenizer, cfg: Config):
    """
    Creates a DataLoader for the test set.
    """
    test_dataset = PhraseDataset(
        test_df, tokenizer=tokenizer, max_length=cfg.max_length, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
