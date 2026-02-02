import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Holds pre-tokenized data, targets, and sample weights.
    """

    def __init__(self, input_ids, attention_mask, targets=None, weights=None):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float)
        else:
            self.targets = None

        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float)
        else:
            self.weights = None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        if self.weights is not None:
            item["weight"] = self.weights[idx]

        return item


def calculate_sample_weights(df):
    """
    Calculates sample weights to mitigate bias.
    Assigns higher weights to:
    1. Toxic examples mentioning an identity (BNSP trap)
    2. Non-toxic examples mentioning an identity (BPSN trap)
    """
    # Default weight is 1.0
    weights = np.ones(len(df), dtype=np.float32)

    # Ensure we have the necessary columns
    required_cols = Config.IDENTITY_COLUMNS + [Config.TARGET_COL]
    if not all(col in df.columns for col in required_cols):
        # If identity columns are missing (e.g. test set), return default weights
        return weights

    # Identify toxic examples (using the standard 0.5 threshold for binary classification logic)
    # Note: We use the continuous target for training, but boolean logic for weighting traps.
    is_toxic = df[Config.TARGET_COL].values >= 0.5

    # Identify if ANY identity is mentioned (value >= 0.5)
    # We check if the max value across identity columns for a row is >= 0.5
    identity_matrix = df[Config.IDENTITY_COLUMNS].values
    is_identity_mentioned = (identity_matrix >= 0.5).any(axis=1)

    # Condition 1: Toxic & Identity Mentioned (BNSP trap)
    bnsp_mask = is_toxic & is_identity_mentioned

    # Condition 2: Non-Toxic & Identity Mentioned (BPSN trap)
    bpsn_mask = (~is_toxic) & is_identity_mentioned

    # Apply weights
    bias_mask = bnsp_mask | bpsn_mask
    weights[bias_mask] = Config.BIAS_LOSS_WEIGHT

    return weights


def process_and_cache(df, tokenizer, name, load_cached_data=True):
    """
    Tokenizes data, calculates weights, and handles caching to .npz files.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(Config.CACHE_DIR, f"{name}_features.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {name} features from cache: {cache_path}")
        data = np.load(cache_path)
        return (
            data["input_ids"],
            data["attention_mask"],
            data["targets"] if "targets" in data else None,
            data["weights"] if "weights" in data else None,
        )

    # 2. Process from scratch
    print(f"Processing {name} features...")

    # Tokenization
    # We use batch_encode_plus for speed
    texts = df[Config.TEXT_COL].fillna("").astype(str).tolist()
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="np",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    targets = None
    weights = None

    # Handle Targets and Weights if training/validation data
    if Config.TARGET_COL in df.columns:
        targets = df[Config.TARGET_COL].values.astype(np.float32)

        # Calculate weights only if identity columns exist
        if all(col in df.columns for col in Config.IDENTITY_COLUMNS):
            weights = calculate_sample_weights(df)
        else:
            weights = np.ones(len(df), dtype=np.float32)

    # 3. Save to cache
    save_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
    if targets is not None:
        save_dict["targets"] = targets
    if weights is not None:
        save_dict["weights"] = weights

    np.savez(cache_path, **save_dict)
    print(f"Saved {name} features to cache.")

    return input_ids, attention_mask, targets, weights


def get_data_loaders(load_cached_data=True):
    """
    Main function to load data, process it, and return DataLoaders.
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Raw Dataframes
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Debug Mode: Sample subset
    if Config.DEBUG:
        print(f"Debug mode enabled. Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # We don't necessarily need to sample test for debug, but it speeds things up
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Process Data
    # Note: We use the 'name' argument to create distinct cache files for debug vs full runs if needed,
    # but here we rely on the user to clear cache or use DEBUG flag which might overwrite.
    # To be safe, we append '_debug' to cache name if debug is on.
    suffix = "_debug" if Config.DEBUG else ""

    train_ids, train_mask, train_y, train_w = process_and_cache(
        train_df, tokenizer, f"train{suffix}", load_cached_data
    )
    val_ids, val_mask, val_y, val_w = process_and_cache(
        val_df, tokenizer, f"val{suffix}", load_cached_data
    )
    test_ids, test_mask, _, _ = process_and_cache(
        test_df, tokenizer, f"test{suffix}", load_cached_data
    )

    # Create Datasets
    train_dataset = ToxicityDataset(train_ids, train_mask, train_y, train_w)
    val_dataset = ToxicityDataset(val_ids, val_mask, val_y, val_w)
    test_dataset = ToxicityDataset(test_ids, test_mask)

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
