import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config

# Suppress tokenizer warnings in forked processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Expects a DataFrame containing 'input_ids', 'attention_mask', and optionally 'score'.
    """

    def __init__(self, df, is_test=False):
        self.input_ids = df["input_ids"].tolist()
        self.attention_mask = df["attention_mask"].tolist()
        self.is_test = is_test

        if not self.is_test:
            # Ensure scores are floats for regression (MSELoss)
            self.labels = df["score"].astype(float).tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Convert lists to tensors
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def process_data(
    input_path, output_path, tokenizer, max_length, is_test=False, load_cached_data=True
):
    """
    Loads raw data, tokenizes it, and caches the result to a parquet file.

    Args:
        input_path (str): Path to the raw CSV file (metadata).
        output_path (str): Path to save/load the processed parquet file.
        tokenizer: Transformers tokenizer instance.
        max_length (int): Maximum sequence length.
        is_test (bool): Whether this is the test set (no labels).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with tokenized features.
    """
    # 1. Try to load from cache
    if load_cached_data and os.path.exists(output_path):
        try:
            df = pd.read_parquet(output_path)
            # Basic validation to ensure cache isn't corrupted
            required_cols = ["input_ids", "attention_mask"]
            if not is_test:
                required_cols.append("score")

            if all(col in df.columns for col in required_cols):
                print(f"Loaded cached data from {output_path}")
                return df
            else:
                print(f"Cache at {output_path} is missing columns. Reprocessing...")
        except Exception as e:
            print(f"Failed to load cache at {output_path}: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df_raw = pd.read_csv(input_path)

    # Fill NaN text if any (though EDA showed none)
    texts = df_raw["full_text"].fillna("").astype(str).tolist()

    # Tokenize
    # This returns a dictionary of lists (input_ids, attention_mask)
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    # Create processed DataFrame
    df_processed = pd.DataFrame(
        {
            "essay_id": df_raw["essay_id"],
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }
    )

    if not is_test:
        df_processed["score"] = df_raw["score"]

    # 3. Save to cache
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as parquet (pyarrow engine handles lists efficiently)
    df_processed.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Saved processed data to {output_path}")

    return df_processed


def get_dataloaders(load_cached_data=True):
    """
    Main function to prepare DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # Process Datasets
    # Train
    df_train = process_data(
        Config.TRAIN_PATH,
        train_cache,
        tokenizer,
        Config.MAX_LENGTH,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # Validation
    df_val = process_data(
        Config.VAL_PATH,
        val_cache,
        tokenizer,
        Config.MAX_LENGTH,
        is_test=False,
        load_cached_data=load_cached_data,
    )

    # Test
    df_test = process_data(
        Config.TEST_PATH,
        test_cache,
        tokenizer,
        Config.MAX_LENGTH,
        is_test=True,
        load_cached_data=load_cached_data,
    )

    # Create Dataset Objects
    train_dataset = EssayDataset(df_train, is_test=False)
    val_dataset = EssayDataset(df_val, is_test=False)
    test_dataset = EssayDataset(df_test, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to maintain consistent stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
