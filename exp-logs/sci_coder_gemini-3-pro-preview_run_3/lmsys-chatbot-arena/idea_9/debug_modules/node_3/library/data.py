import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data")


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Chatbot Arena.

    Features:
    - Tokenizes inputs for Siamese DeBERTa (Branch A and Branch B).
    - Extracts scalar features (log-transformed token lengths).
    - Returns dictionary compatible with the model.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to avoid overhead in __getitem__
        self.prompts = self.df["prompt"].fillna("").astype(str).values
        self.responses_a = self.df["response_a"].fillna("").astype(str).values
        self.responses_b = self.df["response_b"].fillna("").astype(str).values

        if not self.is_test:
            self.targets = self.df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # Tokenize inputs for Branch A: [CLS] Prompt [SEP] Response A [SEP]
        # We use return_token_type_ids=True usually, but DeBERTa V3 relies on relative positions.
        # However, standard HF tokenizers handle the pair construction.
        inputs_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize inputs for Branch B: [CLS] Prompt [SEP] Response B [SEP]
        inputs_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Compute Scalar Features: Log-transformed lengths of tokenized sequences
        # We need lengths of Prompt, Response A, and Response B.
        # Since we are inside __getitem__, we do a quick encode to get accurate token counts.
        # To optimize, we don't need the full tensor, just the length.
        # Note: This adds some CPU overhead but ensures exact alignment with requirements.

        # Helper to get length without special tokens
        def get_token_len(text):
            return len(self.tokenizer.encode(text, add_special_tokens=False))

        len_p = get_token_len(prompt)
        len_a = get_token_len(resp_a)
        len_b = get_token_len(resp_b)

        # Log transform (log(x + 1))
        features = torch.tensor(
            [np.log1p(len_p), np.log1p(len_a), np.log1p(len_b)], dtype=torch.float32
        )

        # Prepare output dictionary
        item = {
            "input_ids_a": inputs_a["input_ids"].squeeze(0),
            "attention_mask_a": inputs_a["attention_mask"].squeeze(0),
            "input_ids_b": inputs_b["input_ids"].squeeze(0),
            "attention_mask_b": inputs_b["attention_mask"].squeeze(0),
            "features": features,
        }

        # Include targets for training/validation
        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def process_dataframe(df, mode="train"):
    """
    Processes the dataframe.
    For 'train' mode, applies symmetric augmentation.
    """
    # Ensure text columns are strings
    text_cols = ["prompt", "response_a", "response_b"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    if mode == "train":
        logger.info(f"Applying symmetric augmentation to {mode} set...")
        # Create a copy for swapping
        df_swapped = df.copy()

        # Swap Responses
        df_swapped = df_swapped.rename(
            columns={
                "response_a": "response_b",
                "response_b": "response_a",
                "winner_model_a": "winner_model_b",
                "winner_model_b": "winner_model_a",
            }
        )

        # Concatenate original and swapped
        df_final = pd.concat([df, df_swapped], axis=0, ignore_index=True)

        # Shuffle to mix original and swapped samples
        df_final = df_final.sample(frac=1, random_state=Config.seed).reset_index(
            drop=True
        )

        logger.info(f"Original size: {len(df)}, Augmented size: {len(df_final)}")
        return df_final

    return df


def load_data(load_cached_data=True):
    """
    Loads data from cache or raw CSVs.
    Implements caching logic using Parquet.
    """
    cache_dir = Config.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    files = {
        "train": (Config.train_path, os.path.join(cache_dir, "train_data.parquet")),
        "val": (Config.val_path, os.path.join(cache_dir, "val_data.parquet")),
        "test": (Config.test_path, os.path.join(cache_dir, "test_data.parquet")),
    }

    dfs = {}

    for mode, (csv_path, cache_path) in files.items():
        loaded = False
        if load_cached_data and os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached {mode} data from {cache_path}...")
                dfs[mode] = pd.read_parquet(cache_path)
                loaded = True
            except Exception as e:
                logger.warning(f"Failed to load cache for {mode}: {e}")

        if not loaded:
            logger.info(f"Processing {mode} data from {csv_path}...")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Source file not found: {csv_path}")

            raw_df = pd.read_csv(csv_path)
            processed_df = process_dataframe(raw_df, mode=mode)

            # Save to cache
            logger.info(f"Saving {mode} data to cache...")
            processed_df.to_parquet(cache_path, index=False)
            dfs[mode] = processed_df

    return dfs["train"], dfs["val"], dfs["test"]


def get_dataloaders(load_cached_data=True, debug=False, debug_size=100):
    """
    Main entry point to get PyTorch DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        debug (bool): If True, subsets data for quick debugging.
        debug_size (int): Size of subset if debug is True.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load Data
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # 2. Handle Debug Mode
    if debug:
        logger.info(f"Debug mode enabled. Subsetting data to {debug_size} rows.")
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    # 3. Initialize Tokenizer
    logger.info(f"Initializing tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 4. Create Datasets
    logger.info("Creating PyTorch Datasets...")
    train_dataset = ChatbotDataset(
        train_df, tokenizer, Config.max_length, is_test=False
    )
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.max_length, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.max_length, is_test=True)

    # 5. Create DataLoaders
    logger.info("Creating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
    )
    return train_loader, val_loader, test_loader
