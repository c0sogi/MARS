import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    SequentialSampler,
)
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger()


class ToxicityDataset(Dataset):
    """
    Dataset class for Toxicity Classification.
    Handles tokenization and retrieval of targets, auxiliary labels, and sample weights.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values

        # Pre-fetch targets and weights if not in test mode
        if not self.is_test:
            self.targets = df["target"].values

            # Auxiliary columns: Identities + Identity Attack
            # We combine them into a single tensor for the auxiliary heads
            aux_cols = Config.IDENTITY_COLS + Config.AUX_COLS
            # Fill NaNs with 0 for auxiliary targets
            self.aux_targets = df[aux_cols].fillna(0.0).values

            # Sample weights for the loss function
            if "weight" in df.columns:
                self.weights = df["weight"].values
            else:
                self.weights = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.texts[index])

        # Tokenization
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)

        if self.is_test:
            return {
                "ids": ids,
                "mask": mask,
                "token_type_ids": token_type_ids,
                # Return dummy values for consistency in collation if needed, or just IDs
                "target": torch.tensor(0.0, dtype=torch.float),
            }

        target = torch.tensor(self.targets[index], dtype=torch.float)
        aux_targets = torch.tensor(self.aux_targets[index], dtype=torch.float)
        weight = torch.tensor(self.weights[index], dtype=torch.float)

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "target": target,
            "aux_targets": aux_targets,
            "weight": weight,
        }


def _calculate_sample_weights(df):
    """
    Calculates sample weights to prioritize 'Bias Traps'.
    Bias Traps defined as: Non-Toxic + Identity Mention OR Toxic + Identity Mention.
    Essentially, any example with an identity mention gets higher weight.
    """
    # Create a mask for rows where ANY identity column is >= 0.5
    # Assuming identity columns are fractional, >= 0.5 indicates presence.
    # Fillna(0) ensures we don't error on missing data, treating it as no identity.
    identity_mask = (df[Config.IDENTITY_COLS].fillna(0.0) >= 0.5).any(axis=1)

    # Assign weights
    # If identity is present -> BIAS_TRAP_WEIGHT (5.0)
    # Else -> NORMAL_WEIGHT (1.0)
    weights = np.where(identity_mask, Config.BIAS_TRAP_WEIGHT, Config.NORMAL_WEIGHT)

    return weights


def load_and_process_data(
    file_path, cache_name, load_cached_data=True, is_train=True, debug=False
):
    """
    Loads data from CSV, calculates weights (if train), and handles caching.
    """
    cache_dir = Config.OUTPUT_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_parquet(cache_file)

        # If debugging, subsample after loading cache
        if debug:
            df = df.sample(n=min(len(df), 2000), random_state=Config.SEED).reset_index(
                drop=True
            )
        return df

    # 2. Load from source
    logger.info(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path)

    # 3. Process Data (Calculate Weights)
    if is_train:
        logger.info("Calculating sample weights for bias mitigation...")
        df["weight"] = _calculate_sample_weights(df)

    # 4. Save to cache (before debug subsampling to preserve full cache)
    # We only cache if we are not in debug mode (or if we want to cache the full set)
    # To be safe and compliant with requirements, we cache the full processed set.
    logger.info(f"Saving processed data to {cache_file}")
    df.to_parquet(cache_file, index=False)

    # 5. Handle Debug Subsampling
    if debug:
        logger.info("Debug mode: Subsampling data...")
        df = df.sample(n=min(len(df), 2000), random_state=Config.SEED).reset_index(
            drop=True
        )

    return df


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates and returns DataLoaders for Train, Validation, and Test sets.
    """
    # Initialize Tokenizer
    logger.info(f"Initializing Tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Data
    train_df = load_and_process_data(
        Config.TRAIN_PATH,
        "train_processed",
        load_cached_data=load_cached_data,
        is_train=True,
        debug=debug,
    )

    val_df = load_and_process_data(
        Config.VAL_PATH,
        "val_processed",
        load_cached_data=load_cached_data,
        is_train=False,  # Val doesn't need training weights for sampling, but metric calc uses identities
        debug=debug,
    )

    test_df = load_and_process_data(
        Config.TEST_PATH,
        "test_processed",
        load_cached_data=load_cached_data,
        is_train=False,
        debug=debug,
    )

    # Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, Config.MAX_LEN)
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.MAX_LEN)
    test_dataset = ToxicityDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Create Samplers
    # Training: WeightedRandomSampler to oversample bias traps
    train_weights = torch.tensor(train_df["weight"].values, dtype=torch.double)
    train_sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader, val_df
