import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger

logger = get_logger("DataModule")


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Siamese Chatbot Preference Prediction.

    Structure:
    - Branch A: [CLS] Prompt [SEP] Response A [SEP]
    - Branch B: [CLS] Prompt [SEP] Response B [SEP]
    - Scalars: Log-transformed lengths of Prompt, Response A, Response B
    """

    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Ensure strings
        prompt = str(row["prompt"])
        resp_a = str(row["response_a"])
        resp_b = str(row["response_b"])

        # Tokenize Branch A
        # truncation="only_second" ensures the prompt is preserved if total length > max_len
        inputs_a = self.tokenizer(
            prompt,
            resp_a,
            truncation="only_second",
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize Branch B
        inputs_b = self.tokenizer(
            prompt,
            resp_b,
            truncation="only_second",
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Scalar features: Log-transformed character lengths (log(x+1))
        len_p = np.log1p(len(prompt))
        len_a = np.log1p(len(resp_a))
        len_b = np.log1p(len(resp_b))
        scalars = torch.tensor([len_p, len_a, len_b], dtype=torch.float)

        item = {
            "input_ids_a": inputs_a["input_ids"].squeeze(0),
            "attention_mask_a": inputs_a["attention_mask"].squeeze(0),
            "input_ids_b": inputs_b["input_ids"].squeeze(0),
            "attention_mask_b": inputs_b["attention_mask"].squeeze(0),
            "scalars": scalars,
        }

        if not self.is_test:
            # Targets: winner_model_a, winner_model_b, winner_tie
            labels = torch.tensor(
                [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]],
                dtype=torch.float,
            )
            item["labels"] = labels

        return item


def process_data(input_path, output_path, load_cached_data=True, is_train=False):
    """
    Loads, cleans, augments (if train), and caches data.
    """
    # 1. Try to load cache
    if load_cached_data and os.path.exists(output_path):
        logger.info(f"Loading cached data from {output_path}")
        try:
            df = pd.read_parquet(output_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Re-processing.")

    # 2. Process from scratch
    logger.info(f"Processing data from {input_path}")
    df = pd.read_csv(input_path)

    # Handle missing values
    text_cols = ["prompt", "response_a", "response_b"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # 3. Symmetric Augmentation for Training
    if is_train:
        logger.info("Applying symmetric augmentation (swapping A/B)...")
        # Create swapped copy
        df_swapped = df.copy()

        # Swap text columns
        df_swapped = df_swapped.rename(
            columns={
                "response_a": "response_b",
                "response_b": "response_a",
                "model_a": "model_b",
                "model_b": "model_a",
            }
        )

        # Swap target columns: winner_a <-> winner_b
        # winner_tie remains unchanged
        df_swapped = df_swapped.rename(
            columns={
                "winner_model_a": "winner_model_b",
                "winner_model_b": "winner_model_a",
            }
        )

        # Concatenate original and swapped
        df = pd.concat([df, df_swapped], axis=0).reset_index(drop=True)
        logger.info(f"Data size after augmentation: {len(df)}")

    # 4. Save to cache
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

    return df


def get_dataloaders(
    load_cached_data=True, debug=Config.DEBUG, batch_size=Config.TRAIN_BATCH_SIZE
):
    """
    Creates DataLoaders for Train, Validation, and Test sets.
    """
    logger.info("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Process Datasets
    logger.info("Preparing Training Data...")
    train_df = process_data(
        Config.TRAIN_DATA_PATH,
        Config.TRAIN_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_train=True,
    )

    logger.info("Preparing Validation Data...")
    val_df = process_data(
        Config.VAL_DATA_PATH,
        Config.VAL_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_train=False,
    )

    logger.info("Preparing Test Data...")
    test_df = process_data(
        Config.TEST_DATA_PATH,
        Config.TEST_CACHE_PATH,
        load_cached_data=load_cached_data,
        is_train=False,
    )

    # Debug Sampling
    if debug:
        logger.info(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Datasets
    train_dataset = ChatbotDataset(
        train_df, tokenizer, Config.MAX_LENGTH, is_test=False
    )
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

    # Instantiate DataLoaders
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
