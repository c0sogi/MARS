import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


def load_and_cache_data(path, cache_name, load_cached_data=True):
    """
    Loads data from CSV or Parquet cache.

    Args:
        path (str): Path to the source CSV file.
        cache_name (str): Name for the cached file (without extension).
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # print(f"Loading raw data from {path}")
    df = pd.read_csv(path)

    # Fill missing text values
    text_cols = ["question_title", "question_body", "answer"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Save to cache
    # print(f"Saving cache to {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


class QuestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            tokenizer: Transformers tokenizer.
            max_len (int): Maximum sequence length.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # Pre-extract columns to avoid overhead in __getitem__
        self.titles = df["question_title"].values
        self.bodies = df["question_body"].values
        self.answers = df["answer"].values
        self.qa_ids = df["qa_id"].values

        if self.mode != "test":
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Tokenize Question Stream: Title + [SEP] + Body
        # The tokenizer handles the special tokens automatically when two args are provided
        q_enc = self.tokenizer(
            title,
            body,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )

        # Tokenize Answer Stream: Answer
        a_enc = self.tokenizer(
            answer, truncation=True, max_length=self.max_len, add_special_tokens=True
        )

        item = {
            "input_ids_q": q_enc["input_ids"],
            "attention_mask_q": q_enc["attention_mask"],
            "input_ids_a": a_enc["input_ids"],
            "attention_mask_a": a_enc["attention_mask"],
        }

        if self.mode != "test":
            item["labels"] = self.targets[idx]

        if self.mode == "test":
            item["qa_id"] = self.qa_ids[idx]

        return item


class Collate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Dynamic padding for Question stream
        q_ids = [torch.tensor(item["input_ids_q"], dtype=torch.long) for item in batch]
        q_mask = [
            torch.tensor(item["attention_mask_q"], dtype=torch.long) for item in batch
        ]

        q_ids = pad_sequence(q_ids, batch_first=True, padding_value=self.pad_token_id)
        q_mask = pad_sequence(q_mask, batch_first=True, padding_value=0)

        # Dynamic padding for Answer stream
        a_ids = [torch.tensor(item["input_ids_a"], dtype=torch.long) for item in batch]
        a_mask = [
            torch.tensor(item["attention_mask_a"], dtype=torch.long) for item in batch
        ]

        a_ids = pad_sequence(a_ids, batch_first=True, padding_value=self.pad_token_id)
        a_mask = pad_sequence(a_mask, batch_first=True, padding_value=0)

        out = {
            "input_ids_q": q_ids,
            "attention_mask_q": q_mask,
            "input_ids_a": a_ids,
            "attention_mask_a": a_mask,
        }

        if "labels" in batch[0]:
            labels = torch.tensor(
                np.array([item["labels"] for item in batch]), dtype=torch.float
            )
            out["labels"] = labels

        if "qa_id" in batch[0]:
            out["qa_id"] = [item["qa_id"] for item in batch]

        return out


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached parquet files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Dataframes
    train_df = load_and_cache_data(
        Config.TRAIN_PATH, "train_processed", load_cached_data
    )
    val_df = load_and_cache_data(Config.VAL_PATH, "val_processed", load_cached_data)
    test_df = load_and_cache_data(Config.TEST_PATH, "test_processed", load_cached_data)

    if debug:
        train_df = train_df.iloc[:100].reset_index(drop=True)
        val_df = val_df.iloc[:100].reset_index(drop=True)
        test_df = test_df.iloc[:100].reset_index(drop=True)

    # Create Datasets
    train_dataset = QuestDataset(train_df, tokenizer, Config.MAX_LEN, mode="train")
    val_dataset = QuestDataset(val_df, tokenizer, Config.MAX_LEN, mode="val")
    test_dataset = QuestDataset(test_df, tokenizer, Config.MAX_LEN, mode="test")

    # Create Collator
    collator = Collate(pad_token_id=tokenizer.pad_token_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
