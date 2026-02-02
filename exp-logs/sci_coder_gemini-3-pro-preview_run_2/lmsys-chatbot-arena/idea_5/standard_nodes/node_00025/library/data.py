import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data")


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Siamese DeBERTa model.
    Handles tokenization of (Prompt, Response A) and (Prompt, Response B) pairs
    and extraction of meta-features.
    """

    def __init__(self, df, tokenizer, max_length=512, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to avoid overhead in __getitem__
        self.prompts = self.df["prompt"].fillna("").astype(str).values
        self.responses_a = self.df["response_a"].fillna("").astype(str).values
        self.responses_b = self.df["response_b"].fillna("").astype(str).values

        # Meta features (normalized)
        self.meta_features = self.df[
            ["norm_len_prompt", "norm_len_a", "norm_len_b"]
        ].values.astype(np.float32)

        if not self.is_test:
            self.targets = self.df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        res_a = self.responses_a[idx]
        res_b = self.responses_b[idx]

        # Tokenize (Prompt, Response A)
        encoded_a = self.tokenizer(
            prompt,
            res_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize (Prompt, Response B)
        encoded_b = self.tokenizer(
            prompt,
            res_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "meta_features": torch.tensor(self.meta_features[idx], dtype=torch.float32),
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def compute_meta_features(df):
    """
    Computes character length features for prompt and responses.
    """
    df["len_prompt"] = df["prompt"].fillna("").str.len()
    df["len_a"] = df["response_a"].fillna("").str.len()
    df["len_b"] = df["response_b"].fillna("").str.len()
    return df


def load_data(load_cached_data=True):
    """
    Loads data, performs feature engineering, and handles caching.
    Loads train, val, and test sets separately to avoid data leakage.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df) with normalized meta-features.
    """
    cache_train_path = os.path.join(Config.CACHE_DIR, "train_processed.parquet")
    cache_val_path = os.path.join(Config.CACHE_DIR, "val_processed.parquet")
    cache_test_path = os.path.join(Config.CACHE_DIR, "test_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            logger.info(f"Loading cached data from {Config.CACHE_DIR}")
            try:
                train_df = pd.read_parquet(cache_train_path)
                val_df = pd.read_parquet(cache_val_path)
                test_df = pd.read_parquet(cache_test_path)
                return train_df, val_df, test_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-processing data.")
        else:
            logger.info("Cache not found. Processing data from scratch.")
    else:
        logger.info("Ignoring cache. Processing data from scratch.")

    # 2. Load raw metadata
    logger.info("Loading raw metadata files...")
    train_df = pd.read_csv(Config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(Config.VAL_DATA_PATH)
    test_df = pd.read_csv(Config.TEST_DATA_PATH)

    logger.info(f"Train set shape: {train_df.shape}")
    logger.info(f"Val set shape: {val_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")

    # 3. Feature Engineering
    logger.info("Computing meta-features...")
    train_df = compute_meta_features(train_df)
    val_df = compute_meta_features(val_df)
    test_df = compute_meta_features(test_df)

    # 4. Normalization
    logger.info("Normalizing meta-features...")
    feature_cols = ["len_prompt", "len_a", "len_b"]
    scaler = StandardScaler()

    # Fit on training data ONLY to prevent leakage
    scaled_train = scaler.fit_transform(train_df[feature_cols])
    scaled_val = scaler.transform(val_df[feature_cols])
    scaled_test = scaler.transform(test_df[feature_cols])

    # Assign back to dataframes
    for i, col in enumerate(feature_cols):
        new_col = f"norm_{col}"
        train_df[new_col] = scaled_train[:, i]
        val_df[new_col] = scaled_val[:, i]
        test_df[new_col] = scaled_test[:, i]

    # 5. Save to cache
    logger.info(f"Saving processed data to {Config.CACHE_DIR}...")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    train_df.to_parquet(cache_train_path, index=False)
    val_df.to_parquet(cache_val_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)

    return train_df, val_df, test_df


def get_folds(df, n_folds=Config.N_FOLDS, seed=Config.SEED):
    """
    Adds a 'fold' column to the dataframe using StratifiedKFold.
    """
    df = df.copy()

    # Create a stratification label
    # 0: model_a wins, 1: model_b wins, 2: tie
    def get_label(row):
        if row["winner_model_a"] == 1:
            return 0
        elif row["winner_model_b"] == 1:
            return 1
        else:
            return 2

    stratify_labels = df.apply(get_label, axis=1)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, stratify_labels)):
        df.loc[val_idx, "fold"] = fold

    logger.info(f"Created {n_folds} folds. Distribution per fold:")
    logger.info(df["fold"].value_counts())

    return df


def get_dataloaders(
    train_df,
    val_df,
    tokenizer,
    batch_size=Config.TRAIN_BATCH_SIZE,
    valid_batch_size=Config.VALID_BATCH_SIZE,
):
    """
    Creates DataLoaders for training and validation.

    Args:
        train_df (pd.DataFrame): Training data for the current fold.
        val_df (pd.DataFrame): Validation data for the current fold.
        tokenizer: Transformers tokenizer instance.
        batch_size (int): Training batch size.
        valid_batch_size (int): Validation batch size.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = ChatbotDataset(
        train_df, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )

    val_dataset = ChatbotDataset(
        val_df, tokenizer, max_length=Config.MAX_LENGTH, is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(test_df, tokenizer, batch_size=Config.VALID_BATCH_SIZE):
    """
    Creates DataLoader for inference on test set.
    """
    test_dataset = ChatbotDataset(
        test_df, tokenizer, max_length=Config.MAX_LENGTH, is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
