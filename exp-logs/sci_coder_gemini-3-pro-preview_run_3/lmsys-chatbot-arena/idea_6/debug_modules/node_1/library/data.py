import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

logger = get_logger(name="data")


def load_and_cache_data(csv_path, cache_path, load_cached_data=True):
    """
    Loads data from CSV, optionally using a Parquet cache.
    Ensures text columns are strings and handles missing values.
    """
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        logger.info(f"Loading raw data from {csv_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Ensure text columns are strings and handle NaNs
        text_cols = ["prompt", "response_a", "response_b"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Save to cache
        logger.info(f"Saving data to cache at {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


class ChatbotDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_train=False, augment=False):
        """
        Dataset class for Siamese DeBERTa architecture.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.augment = augment

        # Pre-extract columns to numpy for faster access during training
        self.prompts = df["prompt"].values
        self.responses_a = df["response_a"].values
        self.responses_b = df["response_b"].values

        # Targets (only if training/validation)
        if "winner_model_a" in df.columns:
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        target = self.targets[idx] if self.targets is not None else None

        # Symmetric Augmentation: Swap A and B with 50% probability
        # Only applicable during training
        if self.is_train and self.augment and np.random.random() < 0.5:
            # Swap text
            resp_a, resp_b = resp_b, resp_a
            # Swap targets if they exist
            if target is not None:
                # Target format: [win_a, win_b, tie] -> [win_b, win_a, tie]
                target = np.array([target[1], target[0], target[2]], dtype=np.float32)

        # Tokenize Branch A
        # truncation='only_second' ensures prompt is preserved and only response is truncated if needed
        # Cite debug_lesson_5: We switch to truncation=True to allow prompt truncation if max_length is exceeded
        encoded_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # Tokenize Branch B
        encoded_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=True,
        )

        # Helper to extract tensors from tokenizer output
        def get_tensors(encoded):
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            # DeBERTa tokenizer returns token_type_ids where 0=Prompt, 1=Response (usually)
            # We rely on this for masking.
            if "token_type_ids" in encoded:
                token_type_ids = encoded["token_type_ids"].squeeze(0)
            else:
                # Fallback if tokenizer doesn't return type ids (unlikely for DeBERTa V3 if requested)
                token_type_ids = torch.zeros_like(input_ids)
            return input_ids, attention_mask, token_type_ids

        ids_a, mask_a, type_a = get_tensors(encoded_a)
        ids_b, mask_b, type_b = get_tensors(encoded_b)

        # Create Response Masks (1 for response tokens, 0 for prompt/padding)
        # We assume token_type_ids are 1 for the second sequence (Response)
        resp_mask_a = type_a * mask_a
        resp_mask_b = type_b * mask_b

        # Calculate Scalar Features (Log Lengths) based on token counts
        # Prompt length is total valid tokens minus response tokens
        # We calculate prompt length from Branch A (should be identical to B)
        len_prompt = (mask_a - resp_mask_a).sum().float()
        len_resp_a = resp_mask_a.sum().float()
        len_resp_b = resp_mask_b.sum().float()

        # Apply Log1p normalization
        scalars = torch.tensor(
            [torch.log1p(len_prompt), torch.log1p(len_resp_a), torch.log1p(len_resp_b)],
            dtype=torch.float32,
        )

        # Construct Output Dictionary
        item = {
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "response_mask_a": resp_mask_a,  # Used for isolated pooling
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "response_mask_b": resp_mask_b,
            "scalars": scalars,
        }

        if target is not None:
            item["target"] = torch.tensor(target, dtype=torch.float32)

        return item


def get_dataloaders(tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test sets.

    Args:
        tokenizer: Transformers tokenizer instance.
        load_cached_data (bool): Whether to use cached Parquet files.

    Returns:
        train_loader, val_loader, test_loader
    """
    # 1. Load DataFrames (with caching)
    train_df = load_and_cache_data(
        Config.TRAIN_PATH, Config.TRAIN_CACHE_PATH, load_cached_data
    )
    val_df = load_and_cache_data(
        Config.VAL_PATH, Config.VAL_CACHE_PATH, load_cached_data
    )
    test_df = load_and_cache_data(
        Config.TEST_PATH, Config.TEST_CACHE_PATH, load_cached_data
    )

    # 2. Debug Mode (Subsampling)
    if Config.DEBUG:
        logger.info("DEBUG mode enabled: Subsampling data.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # 3. Create Datasets
    train_dataset = ChatbotDataset(
        train_df,
        tokenizer,
        Config.MAX_LENGTH,
        is_train=True,
        augment=Config.AUGMENT_DATA,
    )

    val_dataset = ChatbotDataset(
        val_df, tokenizer, Config.MAX_LENGTH, is_train=False, augment=False
    )

    test_dataset = ChatbotDataset(
        test_df, tokenizer, Config.MAX_LENGTH, is_train=False, augment=False
    )

    # 4. Create DataLoaders
    # Using drop_last=True for training to avoid issues with batch norm or gradient accumulation on small last batches
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

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
