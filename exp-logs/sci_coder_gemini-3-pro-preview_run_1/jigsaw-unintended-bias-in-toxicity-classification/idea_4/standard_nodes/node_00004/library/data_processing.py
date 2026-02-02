import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification with Auxiliary Identity Targets.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Ensure text is string and handle NaNs
        self.texts = df["comment_text"].fillna("").astype(str).values

        if not self.is_test:
            self.targets = df["target"].values
            # Auxiliary targets: Identity columns
            # We use the fractional values provided in the data for soft targets
            self.identity_targets = df[Config.IDENTITY_COLUMNS].fillna(0.0).values

            # Sample weights for bias mitigation
            if "sample_weight" in df.columns:
                self.weights = df["sample_weight"].values
            else:
                # Fallback if weight not calculated (should not happen with correct pipeline)
                self.weights = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
            item["aux_target"] = torch.tensor(
                self.identity_targets[idx], dtype=torch.float
            )
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)

        return item


def calculate_sample_weights(df):
    """
    Calculates sample weights to penalize bias.
    Assigns higher weights to examples mentioning identities (both Toxic and Non-Toxic).

    Args:
        df: DataFrame containing identity columns.

    Returns:
        numpy array of weights.
    """
    # Initialize weights with background factor
    weights = np.full(len(df), Config.BACKGROUND_WEIGHT_FACTOR, dtype=np.float32)

    # Check for presence of any identity (value >= 0.5 is considered a mention)
    # Fill NaNs with 0.0 to assume no identity mention if data is missing
    identity_data = df[Config.IDENTITY_COLUMNS].fillna(0.0)
    has_identity = (identity_data >= 0.5).any(axis=1)

    # Assign high weight to rows with identity mentions
    weights[has_identity] = Config.IDENTITY_WEIGHT_FACTOR

    return weights


def load_and_process_data(load_cached_data=True):
    """
    Loads data, calculates weights, and handles caching.

    Args:
        load_cached_data: Boolean, whether to attempt loading from cache.

    Returns:
        train_df, val_df, test_df
    """
    # Define cache paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing data.")
        else:
            print("Cache not found. Processing data from scratch...")
    else:
        print("Forcing data re-processing...")

    # 2. Process from scratch
    print("Loading raw metadata...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Calculate weights for Train and Val
    # (Val weights aren't used for optimization, but kept for consistency)
    print("Calculating sample weights...")
    train_df["sample_weight"] = calculate_sample_weights(train_df)
    val_df["sample_weight"] = calculate_sample_weights(val_df)

    # Ensure text columns are clean (handle NaN)
    train_df["comment_text"] = train_df["comment_text"].fillna("")
    val_df["comment_text"] = val_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    # 3. Save to cache
    print(f"Saving processed data to {Config.CACHE_DIR}...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def create_dataloaders(load_cached_data=True, data_limit=None):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data: Whether to use cached dataframes.
        data_limit: Integer, if set, limits the number of rows (for debugging).

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load Data
    train_df, val_df, test_df = load_and_process_data(load_cached_data=load_cached_data)

    # Apply Data Limit if requested (e.g. for debugging)
    if data_limit is not None:
        print(f"Limiting dataset to {data_limit} samples for debugging.")
        train_df = train_df.iloc[:data_limit]
        val_df = val_df.iloc[:data_limit]
        test_df = test_df.iloc[:data_limit]

    # Initialize Tokenizer
    print(f"Initializing tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_dataset = ToxicityDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
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
