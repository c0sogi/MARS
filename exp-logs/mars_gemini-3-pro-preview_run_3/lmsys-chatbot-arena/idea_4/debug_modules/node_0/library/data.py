import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Chatbot Arena task.
    Constructs Siamese inputs (Prompt + Response A, Prompt + Response B)
    and calculates explicit scalar features.
    """

    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to lists for faster access during training
        # Fill NaNs with empty strings to prevent tokenization errors
        self.prompts = data["prompt"].fillna("").astype(str).tolist()
        self.responses_a = data["response_a"].fillna("").astype(str).tolist()
        self.responses_b = data["response_b"].fillna("").astype(str).tolist()

        if not self.is_test:
            # Targets: winner_model_a, winner_model_b, winner_tie
            self.targets = data[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # 1. Calculate Explicit Scalar Features
        # Log-transformed character lengths (log(len + 1))
        # These provide high-level signal about verbosity which correlates with preference
        feat_prompt = np.log1p(len(prompt))
        feat_resp_a = np.log1p(len(resp_a))
        feat_resp_b = np.log1p(len(resp_b))

        scalars = torch.tensor(
            [feat_prompt, feat_resp_a, feat_resp_b], dtype=torch.float32
        )

        # 2. Tokenization
        # We construct two separate inputs sharing the same encoder:
        # Branch A: [CLS] Prompt [SEP] Response A [SEP]
        # Branch B: [CLS] Prompt [SEP] Response B [SEP]
        # truncation='only_second' ensures the Prompt is preserved if the sequence is too long.

        # Tokenize Branch A
        enc_a = self.tokenizer(
            prompt,
            resp_a,
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B
        enc_b = self.tokenizer(
            prompt,
            resp_b,
            truncation="only_second",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Prepare output dictionary
        # Squeeze(0) is needed because return_tensors='pt' adds a batch dimension (1, seq_len)
        item = {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "scalars": scalars,
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def load_and_process_data(mode, load_cached_data=True):
    """
    Loads data from metadata CSVs or cached Parquet files.
    Implements the required caching mechanism.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {mode} data from cache: {cache_path} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reloading from source.")

    # 2. Load from source metadata
    if mode == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        source_path = Config.VAL_DATA_PATH
    elif mode == "test":
        source_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Loading {mode} data from metadata: {source_path}")
    df = pd.read_csv(source_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Cached {mode} data to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to cache data: {e}")

    return df


def get_dataloaders(tokenizer, batch_size=None, load_cached_data=True):
    """
    Creates training and validation dataloaders.

    Args:
        tokenizer: The HuggingFace tokenizer.
        batch_size (int, optional): Batch size. Defaults to Config.TRAIN_BATCH_SIZE.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = Config.TRAIN_BATCH_SIZE

    # Load Data
    train_df = load_and_process_data("train", load_cached_data)
    val_df = load_and_process_data("val", load_cached_data)

    # Debug Subsampling
    if Config.DEBUG:
        print(f"DEBUG Mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_ds = ChatbotDataset(train_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    val_ds = ChatbotDataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)

    # Create Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(tokenizer, batch_size=None, load_cached_data=True):
    """
    Creates the test dataloader.

    Args:
        tokenizer: The HuggingFace tokenizer.
        batch_size (int, optional): Batch size. Defaults to Config.VALID_BATCH_SIZE.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: The test dataloader.
    """
    if batch_size is None:
        batch_size = Config.VALID_BATCH_SIZE

    test_df = load_and_process_data("test", load_cached_data)

    test_ds = ChatbotDataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
