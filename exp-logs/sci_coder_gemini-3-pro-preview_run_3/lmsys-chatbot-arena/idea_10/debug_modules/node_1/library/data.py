import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

logger = get_logger("data")


class ChatbotDataset(Dataset):
    """
    Dataset class for the Siamese DeBERTa model.
    Tokenizes (Prompt, Response) pairs on-the-fly and computes scalar features.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to lists for faster access
        self.prompts = df["prompt"].astype(str).tolist()
        self.responses_a = df["response_a"].astype(str).tolist()
        self.responses_b = df["response_b"].astype(str).tolist()

        if not self.is_test:
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        res_a = self.responses_a[idx]
        res_b = self.responses_b[idx]

        # Tokenize Branch A: [CLS] Prompt [SEP] Response A [SEP]
        inputs_a = self.tokenizer(
            prompt,
            res_a,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # Tokenize Branch B: [CLS] Prompt [SEP] Response B [SEP]
        inputs_b = self.tokenizer(
            prompt,
            res_b,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # Helper to extract scalar lengths from tokenized sequence
        def get_token_lengths(input_ids, sep_token_id):
            # input_ids is shape (1, seq_len)
            # Find indices of [SEP] tokens
            sep_indices = (input_ids == sep_token_id).nonzero(as_tuple=True)[1]

            if len(sep_indices) >= 2:
                # [CLS] Prompt [SEP] Response [SEP] ...
                # Prompt len: from index 1 to first SEP (exclusive)
                p_len = sep_indices[0].item() - 1
                # Response len: from first SEP+1 to second SEP (exclusive)
                r_len = sep_indices[1].item() - sep_indices[0].item() - 1
            elif len(sep_indices) == 1:
                # Fallback if truncated aggressively
                p_len = sep_indices[0].item() - 1
                r_len = 0
            else:
                p_len = 0
                r_len = 0

            return max(0, p_len), max(0, r_len)

        sep_id = self.tokenizer.sep_token_id

        p_len_a, r_len_a = get_token_lengths(inputs_a["input_ids"], sep_id)
        p_len_b, r_len_b = get_token_lengths(inputs_b["input_ids"], sep_id)

        # We take the average prompt length (should be identical unless truncation differs slightly)
        p_len = (p_len_a + p_len_b) / 2.0

        # Log transform scalar features
        # Adding 1 to avoid log(0)
        scalars = torch.tensor(
            [np.log1p(p_len), np.log1p(r_len_a), np.log1p(r_len_b)], dtype=torch.float32
        )

        # Prepare output dictionary
        item = {
            "input_ids_a": inputs_a["input_ids"].squeeze(0),
            "attention_mask_a": inputs_a["attention_mask"].squeeze(0),
            "token_type_ids_a": inputs_a["token_type_ids"].squeeze(0),
            "input_ids_b": inputs_b["input_ids"].squeeze(0),
            "attention_mask_b": inputs_b["attention_mask"].squeeze(0),
            "token_type_ids_b": inputs_b["token_type_ids"].squeeze(0),
            "scalars": scalars,
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def process_dataframe(df_path, mode, load_cached_data=True):
    """
    Loads, processes, and augments data. Handles caching.
    """
    cache_file = os.path.join(Config.cache_dir, f"{mode}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        logger.info(f"Loading {mode} data from cache: {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}. Reprocessing.")

    # 2. Process from scratch
    logger.info(f"Processing {mode} data from {df_path}...")
    df = pd.read_csv(df_path)

    # Fill NaNs in text columns
    text_cols = ["prompt", "response_a", "response_b"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    # 3. Symmetric Augmentation (Train only)
    if mode == "train" and Config.augment_symmetric:
        logger.info("Applying symmetric augmentation (swapping A/B)...")
        df_aug = df.copy()

        # Swap responses
        df_aug = df_aug.rename(
            columns={
                "response_a": "response_b",
                "response_b": "response_a",
                "winner_model_a": "winner_model_b",
                "winner_model_b": "winner_model_a",
            }
        )

        # Concatenate
        df = pd.concat([df, df_aug], axis=0).reset_index(drop=True)
        logger.info(f"Augmented train size: {len(df)}")

    # 4. Save to cache
    try:
        df.to_parquet(cache_file, index=False)
        logger.info(f"Saved {mode} data to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Could not save cache: {e}")

    return df


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    # Initialize Tokenizer
    logger.info(f"Initializing tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name, use_fast=True)

    # Load DataFrames
    train_df = process_dataframe(Config.train_path, "train", load_cached_data)
    val_df = process_dataframe(Config.val_path, "val", load_cached_data)
    test_df = process_dataframe(Config.test_path, "test", load_cached_data)

    # Debug Mode: Downsample
    if debug or Config.debug:
        logger.info("Debug mode enabled: Downsampling data.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Create Datasets
    train_dataset = ChatbotDataset(train_df, tokenizer, Config.max_len, is_test=False)
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.max_len, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.max_len, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.physical_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.physical_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.physical_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
