import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QuestDataset(Dataset):
    """
    Dataset class for the StackExchange Question-Answer task.
    Handles asymmetric tokenization for Question (Title+Body) and Answer branches.
    """

    def __init__(self, df, tokenizer, target_cols=None, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.mode = mode
        self.target_cols = target_cols

        # Extract feature columns
        # Fill NaNs with empty strings to prevent tokenization errors
        self.titles = df["question_title"].fillna("").astype(str).values
        self.bodies = df["question_body"].fillna("").astype(str).values
        self.answers = df["answer"].fillna("").astype(str).values
        self.qa_ids = df["qa_id"].values

        # Extract targets if not in test mode
        if self.mode != "test":
            if self.target_cols is None:
                raise ValueError("target_cols must be provided for train/val mode")
            self.labels = df[self.target_cols].values.astype(np.float32)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Question Branch: Pair tokenization (Title + Body)
        # This automatically adds separators: <s>Title</s></s>Body</s>
        # We do not pad here; padding is handled in collate_fn for dynamic batching
        q_encoded = self.tokenizer(
            text=self.titles[idx],
            text_pair=self.bodies[idx],
            truncation=True,
            max_length=Config.MAX_LEN_Q,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # 2. Answer Branch: Independent tokenization
        a_encoded = self.tokenizer(
            text=self.answers[idx],
            truncation=True,
            max_length=Config.MAX_LEN_A,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        sample = {
            "qa_id": self.qa_ids[idx],
            "input_ids_q": q_encoded["input_ids"],
            "attention_mask_q": q_encoded["attention_mask"],
            "input_ids_a": a_encoded["input_ids"],
            "attention_mask_a": a_encoded["attention_mask"],
        }

        if self.labels is not None:
            sample["labels"] = self.labels[idx]

        return sample


class Collate:
    """
    Custom collate function to handle dynamic padding for two separate input branches.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Unpack the batch
        input_ids_q = [
            {
                "input_ids": item["input_ids_q"],
                "attention_mask": item["attention_mask_q"],
            }
            for item in batch
        ]
        input_ids_a = [
            {
                "input_ids": item["input_ids_a"],
                "attention_mask": item["attention_mask_a"],
            }
            for item in batch
        ]

        # Dynamic padding using tokenizer.pad
        # This pads to the longest sequence in this specific batch
        batch_q = self.tokenizer.pad(input_ids_q, padding=True, return_tensors="pt")

        batch_a = self.tokenizer.pad(input_ids_a, padding=True, return_tensors="pt")

        result = {
            "qa_id": torch.tensor([item["qa_id"] for item in batch], dtype=torch.long),
            "input_ids_q": batch_q["input_ids"],
            "attention_mask_q": batch_q["attention_mask"],
            "input_ids_a": batch_a["input_ids"],
            "attention_mask_a": batch_a["attention_mask"],
        }

        if "labels" in batch[0]:
            result["labels"] = torch.tensor(
                np.array([item["labels"] for item in batch]), dtype=torch.float32
            )

        return result


def load_data(path, cache_name, load_cached_data=True):
    """
    Loads data from CSV or Parquet cache.
    Implements the required caching logic.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, cache_name)

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Reloading from source.")

    # 2. Load from source
    print(f"Loading data from source: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path)

    # 3. Save to cache
    print(f"Saving cache to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Factory function to create DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to use cached parquet files.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader, target_cols)
    """
    seed_everything(Config.SEED)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Identify Target Columns from Sample Submission
    # This ensures we have the correct order and names
    sample_sub = pd.read_csv(Config.SAMPLE_SUB_PATH)
    target_cols = [col for col in sample_sub.columns if col != "qa_id"]
    assert (
        len(target_cols) == Config.NUM_TARGETS
    ), "Mismatch in number of target columns"

    # Load Dataframes
    train_df = load_data(Config.TRAIN_PATH, "train_processed.parquet", load_cached_data)
    val_df = load_data(Config.VAL_PATH, "val_processed.parquet", load_cached_data)
    test_df = load_data(Config.TEST_PATH, "test_processed.parquet", load_cached_data)

    # Debug mode: subset data
    if debug:
        print("DEBUG MODE: Subsetting data...")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Create Datasets
    train_dataset = QuestDataset(train_df, tokenizer, target_cols, mode="train")
    val_dataset = QuestDataset(val_df, tokenizer, target_cols, mode="val")
    test_dataset = QuestDataset(test_df, tokenizer, target_cols=None, mode="test")

    # Create Collate Function
    collate_fn = Collate(tokenizer)

    # Create DataLoaders
    # Note: Config.TRAIN_BATCH_SIZE is the physical batch size
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

    return train_loader, val_loader, test_loader, target_cols
