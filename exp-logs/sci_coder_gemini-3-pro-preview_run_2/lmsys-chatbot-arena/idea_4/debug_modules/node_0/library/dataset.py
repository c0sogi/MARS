import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import MetaFeatureScaler


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese Chatbot preference task.
    Handles on-the-fly tokenization of (Prompt, Response) pairs and
    retrieval of pre-computed meta-features.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Convert columns to numpy arrays for faster access
        self.ids = df["id"].values
        # Fill NaNs to avoid tokenization errors
        self.prompts = df["prompt"].fillna("").astype(str).values
        self.response_a = df["response_a"].fillna("").astype(str).values
        self.response_b = df["response_b"].fillna("").astype(str).values

        # Meta features: Expecting columns 'meta_0', 'meta_1', 'meta_2'
        # corresponding to normalized lengths of prompt, res_a, res_b
        self.meta_features = df[["meta_0", "meta_1", "meta_2"]].values.astype(
            np.float32
        )

        if not self.is_test:
            # Targets: [winner_model_a, winner_model_b, winner_tie]
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        res_a = self.response_a[idx]
        res_b = self.response_b[idx]

        # Tokenize Pair A: (Prompt, Response A)
        encoded_a = self.tokenizer(
            prompt,
            res_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Pair B: (Prompt, Response B)
        encoded_b = self.tokenizer(
            prompt,
            res_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "ids": self.ids[idx],
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "meta_features": torch.tensor(self.meta_features[idx], dtype=torch.float),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def _process_and_cache_data(load_cached_data):
    """
    Loads metadata, computes meta-features using MetaFeatureScaler,
    and caches the resulting DataFrames to Parquet.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    train_cache_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_cache_path = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_cache_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache_path)
            and os.path.exists(val_cache_path)
            and os.path.exists(test_cache_path)
        ):
            try:
                df_train = pd.read_parquet(train_cache_path)
                df_val = pd.read_parquet(val_cache_path)
                df_test = pd.read_parquet(test_cache_path)
                return df_train, df_val, df_test
            except Exception:
                # If load fails, proceed to re-process
                pass

    # 2. Process from scratch
    df_train = pd.read_csv(Config.TRAIN_META)
    df_val = pd.read_csv(Config.VAL_META)
    df_test = pd.read_csv(Config.TEST_META)

    # Initialize and fit scaler on training data
    scaler = MetaFeatureScaler()

    # fit_transform returns (N, 3) numpy array
    train_feats = scaler.fit_transform(df_train)
    val_feats = scaler.transform(df_val)
    test_feats = scaler.transform(df_test)

    # Append features to DataFrames
    feat_cols = ["meta_0", "meta_1", "meta_2"]
    df_train[feat_cols] = train_feats
    df_val[feat_cols] = val_feats
    df_test[feat_cols] = test_feats

    # Save to cache
    df_train.to_parquet(train_cache_path, index=False)
    df_val.to_parquet(val_cache_path, index=False)
    df_test.to_parquet(test_cache_path, index=False)

    return df_train, df_val, df_test


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        tokenizer: The HuggingFace tokenizer instance.
        load_cached_data (bool): Whether to attempt loading cached processed data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load processed dataframes
    df_train, df_val, df_test = _process_and_cache_data(load_cached_data)

    # Debug mode: Slice data
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SAMPLES)
        df_val = df_val.head(Config.DEBUG_SAMPLES)
        df_test = df_test.head(Config.DEBUG_SAMPLES)

    # Instantiate Datasets
    train_dataset = ChatbotDataset(
        df_train, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )

    val_dataset = ChatbotDataset(
        df_val, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )

    test_dataset = ChatbotDataset(
        df_test, tokenizer, max_length=Config.MAX_LENGTH, is_test=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
