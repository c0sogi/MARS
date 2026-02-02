import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


def load_and_preprocess_data(load_cached_data=True):
    """
    Loads metadata, performs text preprocessing (concatenation/filling NaNs),
    and caches the result to Parquet files.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Define cache paths
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed.parquet")

    # 1. Try loading from cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            try:
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)
                test_df = pd.read_parquet(test_cache)
                return train_df, val_df, test_df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
        else:
            print("Cache not found. Processing data from scratch...")

    # 2. Load raw metadata
    print("Loading metadata CSVs...")
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # 3. Preprocess
    # We need to ensure text columns are strings and handle NaNs
    # Logic: Question = Title + " " + Body, Answer = Answer
    def process_df(df):
        # Fill NaNs
        for col in Config.TEXT_COLS:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        # Create consolidated text columns
        # Using a separator space
        df["text_q"] = df["question_title"] + " " + df["question_body"]
        df["text_a"] = df["answer"]
        return df

    print("Processing text columns...")
    train_df = process_df(train_df)
    val_df = process_df(val_df)
    test_df = process_df(test_df)

    # 4. Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


class QADataset(Dataset):
    """
    PyTorch Dataset for Question-Answer pairs.
    Handles tokenization and target extraction.
    """

    def __init__(self, df, tokenizer, max_length, is_test=False):
        """
        Args:
            df (pd.DataFrame): Processed dataframe containing 'text_q', 'text_a', and targets.
            tokenizer: HuggingFace tokenizer.
            max_length (int): Maximum sequence length.
            is_test (bool): If True, does not look for target columns.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Pre-extract lists to avoid dataframe overhead in __getitem__
        self.text_q = df["text_q"].tolist()
        self.text_a = df["text_a"].tolist()

        if not self.is_test:
            self.targets = df[Config.TARGET_COLS].values.astype("float32")

        self.qa_ids = df[Config.ID_COL].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text_q = self.text_q[idx]
        text_a = self.text_a[idx]

        # Tokenize
        # We pass text_q and text_a as a pair to handle [CLS] Q [SEP] A [SEP]
        # We do NOT use return_tensors="pt" here to get lists and access sequence_ids()
        inputs = self.tokenizer(
            text_q,
            text_a,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Generate Masks using sequence_ids
        # sequence_ids returns: [None, 0, 0, ..., None, 1, 1, ..., None]
        # 0 -> Question, 1 -> Answer, None -> Special Tokens
        seq_ids = inputs.sequence_ids()

        # Create binary masks (1 for segment, 0 otherwise)
        # We treat None as 0 (ignore special tokens in pooling)
        q_mask = [1.0 if s == 0 else 0.0 for s in seq_ids]
        a_mask = [1.0 if s == 1 else 0.0 for s in seq_ids]

        # Convert to tensors
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        q_mask = torch.tensor(q_mask, dtype=torch.float)
        a_mask = torch.tensor(a_mask, dtype=torch.float)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "q_mask": q_mask,
            "a_mask": a_mask,
            "qa_id": self.qa_ids[idx],
        }

        if "token_type_ids" in inputs:
            item["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def get_dataloaders(model_name, load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        model_name (str): Name of the HuggingFace model (for tokenizer).
        load_cached_data (bool): Whether to use cached processed dataframes.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Load Data
    train_df, val_df, test_df = load_and_preprocess_data(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Tokenizer
    print(f"Initializing tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Create Datasets
    train_dataset = QADataset(train_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    val_dataset = QADataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    test_dataset = QADataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

    # 4. Create DataLoaders
    # Use Config.NUM_WORKERS for parallel loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
