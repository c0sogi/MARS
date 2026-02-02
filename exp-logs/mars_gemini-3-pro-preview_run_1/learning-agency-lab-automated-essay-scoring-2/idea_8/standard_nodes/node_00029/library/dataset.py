import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.configuration import Config
from library.utilities import get_logger

logger = get_logger("Dataset")


def get_tokenizer():
    """
    Loads the tokenizer defined in the configuration.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_BACKBONE, use_fast=True)


class Collate:
    """
    Handles dynamic padding of batches.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Collates a list of dataset items into a batch.
        Assumes items are dictionaries with keys: 'input_ids', 'attention_mask', and optionally 'labels'.
        """
        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]

        # Dynamic padding
        # Convert to tensors first to use pad_sequence or manual padding
        # Here we use tokenizer.pad which handles list of lists/tensors
        batch_encoding = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding=True,
            return_tensors="pt",
        )

        output = {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
        }

        # Handle labels if present (Supervised task)
        if "labels" in batch[0]:
            labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float)
            output["labels"] = labels

        return output


class EssayDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning and Inference.
    """

    def __init__(self, df):
        self.df = df
        self.has_labels = "score" in df.columns

        # Ensure input_ids are lists (in case they were loaded as arrays from cache)
        # Pandas parquet sometimes loads lists as numpy arrays of objects
        if isinstance(self.df.iloc[0]["input_ids"], np.ndarray):
            self.df["input_ids"] = self.df["input_ids"].apply(list)
        if isinstance(self.df.iloc[0]["attention_mask"], np.ndarray):
            self.df["attention_mask"] = self.df["attention_mask"].apply(list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        item = {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}

        if self.has_labels:
            item["labels"] = float(row["score"])

        return item


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (Domain Adaptation).
    """

    def __init__(self, df):
        self.df = df

        # Ensure consistency
        if isinstance(self.df.iloc[0]["input_ids"], np.ndarray):
            self.df["input_ids"] = self.df["input_ids"].apply(list)
        if isinstance(self.df.iloc[0]["attention_mask"], np.ndarray):
            self.df["attention_mask"] = self.df["attention_mask"].apply(list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Return unmasked tokens; masking is handled by DataCollatorForLanguageModeling in the trainer
        return {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}


def _process_text(df, tokenizer, max_length):
    """
    Helper function to tokenize text columns.
    """
    logger.info(f"Tokenizing {len(df)} samples with max_length={max_length}...")

    # Fill NaNs just in case
    texts = df["full_text"].fillna("").astype(str).tolist()

    encodings = tokenizer(
        texts,
        max_length=max_length,
        padding=False,  # We use dynamic padding in Collate
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
    )

    # Assign back to dataframe
    df_processed = df.copy()
    df_processed["input_ids"] = encodings["input_ids"]
    df_processed["attention_mask"] = encodings["attention_mask"]

    return df_processed


def _get_cache_path(name):
    return os.path.join(Config.CACHE_DIR, f"{name}_processed.parquet")


def load_supervised_data(partition, tokenizer, load_cached_data=True, debug=False):
    """
    Loads, processes, and caches data for the supervised task.

    Args:
        partition (str): 'train', 'val', or 'test'.
        tokenizer: Transformers tokenizer.
        load_cached_data (bool): Whether to use cache.
        debug (bool): If True, subsamples data.

    Returns:
        EssayDataset: The ready-to-use dataset.
    """
    cache_path = _get_cache_path(partition)

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached {partition} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # Basic validation
            if "input_ids" in df.columns and "attention_mask" in df.columns:
                if debug:
                    df = df.head(Config.DEBUG_SAMPLE_SIZE)
                return EssayDataset(df)
            else:
                logger.warning(
                    f"Cache at {cache_path} is invalid (missing columns). Recomputing."
                )
        except Exception as e:
            logger.warning(f"Failed to load cache at {cache_path}: {e}. Recomputing.")

    # 2. Compute from Scratch
    logger.info(f"Processing {partition} data from scratch...")

    # Map partition to file path
    if partition == "train":
        path = Config.TRAIN_PATH
    elif partition == "val":
        path = Config.VAL_PATH
    elif partition == "test":
        path = Config.TEST_PATH
    else:
        raise ValueError(f"Unknown partition: {partition}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    df = pd.read_csv(path)

    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    # Tokenize
    df_processed = _process_text(df, tokenizer, Config.MAX_LENGTH)

    # 3. Save Cache (only if not debugging, to avoid overwriting full cache with subset)
    if not debug:
        try:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            logger.info(f"Saving {partition} data to {cache_path}")
            df_processed.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    return EssayDataset(df_processed)


def load_mlm_data(tokenizer, load_cached_data=True, debug=False):
    """
    Loads, combines, processes, and caches data for MLM pre-training.
    Combines Train and Test sets.

    Args:
        tokenizer: Transformers tokenizer.
        load_cached_data (bool): Whether to use cache.
        debug (bool): If True, subsamples data.

    Returns:
        MLMDataset: The ready-to-use dataset.
    """
    cache_path = _get_cache_path("mlm_combined")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached MLM data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            if debug:
                df = df.head(Config.DEBUG_SAMPLE_SIZE)
            return MLMDataset(df)
        except Exception as e:
            logger.warning(f"Failed to load MLM cache: {e}. Recomputing.")

    # 2. Compute from Scratch
    logger.info("Processing MLM data (Train + Test) from scratch...")

    df_train = pd.read_csv(Config.TRAIN_PATH)
    df_val = pd.read_csv(
        Config.VAL_PATH
    )  # Include val in MLM as it is part of training distribution
    df_test = pd.read_csv(Config.TEST_PATH)

    # Combine texts
    # We only need the text column
    combined_texts = pd.concat(
        [df_train[["full_text"]], df_val[["full_text"]], df_test[["full_text"]]],
        ignore_index=True,
    )

    if debug:
        combined_texts = combined_texts.head(Config.DEBUG_SAMPLE_SIZE)

    # Tokenize
    df_processed = _process_text(combined_texts, tokenizer, Config.MAX_LENGTH)

    # 3. Save Cache
    if not debug:
        try:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            logger.info(f"Saving MLM data to {cache_path}")
            df_processed.to_parquet(cache_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save MLM cache: {e}")

    return MLMDataset(df_processed)
