import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class StackExchangeDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): The dataframe containing the data.
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode

        # Extract text features, ensuring they are strings
        # Cite solution_lesson_node_00059: Concatenate title and body for early fusion
        titles = df["question_title"].astype(str).fillna("").values
        bodies = df["question_body"].astype(str).fillna("").values

        # Simple space concatenation. Transformer tokenizer handles the rest.
        self.questions = [t + " " + b for t, b in zip(titles, bodies)]
        self.answers = df["answer"].astype(str).fillna("").tolist()

        # Extract targets if not in test mode
        if self.mode != "test":
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)
        else:
            self.targets = None
            self.qa_ids = df["qa_id"].values

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = {
            "question": self.questions[idx],
            "answer": self.answers[idx],
        }

        if self.targets is not None:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.mode == "test":
            item["qa_id"] = self.qa_ids[idx]

        return item


class Collate:
    def __init__(self, tokenizer_name=Config.MODEL_NAME, max_len=Config.MAX_LEN):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __call__(self, batch):
        """
        Tokenizes question (title+body) and answer independently with dynamic padding.
        """
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]

        # Tokenize independently
        # padding=True applies dynamic padding to the longest sequence in the batch
        # truncation=True ensures we don't exceed model limits
        question_enc = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        answer_enc = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        batch_out = {
            "question_input_ids": question_enc["input_ids"],
            "question_attention_mask": question_enc["attention_mask"],
            "answer_input_ids": answer_enc["input_ids"],
            "answer_attention_mask": answer_enc["attention_mask"],
        }

        if "targets" in batch[0]:
            targets = torch.stack([item["targets"] for item in batch])
            batch_out["targets"] = targets

        if "qa_id" in batch[0]:
            batch_out["qa_ids"] = [item["qa_id"] for item in batch]

        return batch_out


def load_data(load_cached_data=True):
    """
    Loads data from cache or metadata CSVs.
    Handles caching logic using Parquet.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = Config.TRAIN_CACHE_PATH
    val_cache = Config.VAL_CACHE_PATH
    test_cache = Config.TEST_CACHE_PATH

    # Check if we should load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        try:
            print(f"Loading cached data from {Config.WORKING_DIR}...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    print("Loading data from metadata CSVs...")

    # Load from metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Preprocessing: Fill NaNs in text columns
    text_cols = ["question_title", "question_body", "answer"]
    for col in text_cols:
        train_df[col] = train_df[col].fillna("")
        val_df[col] = val_df[col].fillna("")
        test_df[col] = test_df[col].fillna("")

    # Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Creates and returns DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from cache.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    if debug:
        print("Debug mode: utilizing small subset of data.")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Initialize Datasets
    train_dataset = StackExchangeDataset(train_df, mode="train")
    val_dataset = StackExchangeDataset(val_df, mode="val")
    test_dataset = StackExchangeDataset(test_df, mode="test")

    # Initialize Collator
    collate_fn = Collate(tokenizer_name=Config.MODEL_NAME, max_len=Config.MAX_LEN)

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
