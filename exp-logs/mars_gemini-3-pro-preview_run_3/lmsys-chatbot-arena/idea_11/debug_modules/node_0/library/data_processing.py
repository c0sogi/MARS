import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("data_processing")


class SiameseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract columns to lists for faster access
        self.prompts = data["prompt"].tolist()
        self.responses_a = data["response_a"].tolist()
        self.responses_b = data["response_b"].tolist()

        if not self.is_test:
            self.targets = data[Config.TARGET_COLS].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = str(self.prompts[index])
        resp_a = str(self.responses_a[index])
        resp_b = str(self.responses_b[index])

        # Tokenize Branch A
        # We use truncation=True (longest_first) to be robust against massive prompts
        inputs_a = self.tokenizer(
            prompt,
            resp_a,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B
        inputs_b = self.tokenizer(
            prompt,
            resp_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Extract tensors and remove batch dimension
        ids_a = inputs_a["input_ids"].squeeze(0)
        mask_a = inputs_a["attention_mask"].squeeze(0)

        ids_b = inputs_b["input_ids"].squeeze(0)
        mask_b = inputs_b["attention_mask"].squeeze(0)

        # Helper to create token_type_ids manually
        # 0 for Prompt ([CLS] ... [SEP]), 1 for Response (... [SEP])
        def create_type_ids(input_ids, sep_id):
            sep_indices = (input_ids == sep_id).nonzero(as_tuple=True)[0]
            type_ids = torch.zeros_like(input_ids)
            if len(sep_indices) >= 1:
                # The first segment (Prompt) ends at the first SEP.
                # We mark everything after the first SEP as 1 (Response).
                first_sep = sep_indices[0]
                type_ids[first_sep + 1 :] = 1
            return type_ids

        sep_token_id = self.tokenizer.sep_token_id
        tt_ids_a = create_type_ids(ids_a, sep_token_id)
        tt_ids_b = create_type_ids(ids_b, sep_token_id)

        # Scalar Features: log(length)
        # Calculate lengths based on valid tokens in each segment
        len_prompt_a = ((mask_a == 1) & (tt_ids_a == 0)).sum().float()
        len_resp_a = ((mask_a == 1) & (tt_ids_a == 1)).sum().float()
        len_resp_b = ((mask_b == 1) & (tt_ids_b == 1)).sum().float()

        scalars = torch.tensor(
            [
                torch.log1p(len_prompt_a),
                torch.log1p(len_resp_a),
                torch.log1p(len_resp_b),
            ],
            dtype=torch.float,
        )

        item = {
            "input_ids_a": ids_a,
            "attention_mask_a": mask_a,
            "token_type_ids_a": tt_ids_a,
            "input_ids_b": ids_b,
            "attention_mask_b": mask_b,
            "token_type_ids_b": tt_ids_b,
            "scalars": scalars,
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[index], dtype=torch.float)

        return item


def process_data(load_cached_data=True):
    """
    Loads, preprocesses, and augments data.
    Handles caching to Parquet files.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_data.parquet")
    val_cache = os.path.join(cache_dir, "val_data.parquet")
    test_cache = os.path.join(cache_dir, "test_data.parquet")

    # Check cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        logger.info("Loading cached data from parquet files...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    logger.info("Processing data from scratch...")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Fill NaNs
    text_cols = ["prompt", "response_a", "response_b"]
    for df in [train_df, val_df, test_df]:
        for col in text_cols:
            df[col] = df[col].fillna("")

    # Symmetric Augmentation for Training Data
    logger.info(f"Original Train Shape: {train_df.shape}")

    swapped_train = train_df.copy()
    # Swap responses
    swapped_train["response_a"] = train_df["response_b"]
    swapped_train["response_b"] = train_df["response_a"]
    # Swap targets
    swapped_train["winner_model_a"] = train_df["winner_model_b"]
    swapped_train["winner_model_b"] = train_df["winner_model_a"]
    # winner_tie remains unchanged

    # Concatenate
    train_df = pd.concat([train_df, swapped_train], axis=0).reset_index(drop=True)
    logger.info(f"Augmented Train Shape: {train_df.shape}")

    # Save to cache
    logger.info("Saving processed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, val, and test.
    """
    # Load Data
    train_df, val_df, test_df = process_data(load_cached_data=load_cached_data)

    if debug:
        logger.info(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = SiameseDataset(
        train_df, tokenizer, Config.MAX_LENGTH, is_test=False
    )
    val_dataset = SiameseDataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    test_dataset = SiameseDataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

    # Create DataLoaders
    # drop_last=True for training to maintain consistent batch sizes for grad accumulation stability
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
