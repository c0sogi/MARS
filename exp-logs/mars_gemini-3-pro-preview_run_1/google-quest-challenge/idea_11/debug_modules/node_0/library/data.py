import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from library.config import Config
from library.utils import seed_everything


def load_data(load_cached_data: bool = True):
    """
    Loads data from cache (Parquet) or processes from raw CSVs (Metadata).
    Implements the required caching logic.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    train_cache = Config.TRAIN_CACHE
    val_cache = Config.VAL_CACHE
    test_cache = Config.TEST_CACHE

    # Check if we should and can load from cache
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print(f"Loading cached data from {Config.WORKING_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Processing data from metadata...")
        # Load from metadata
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Basic cleaning (fill NaNs in text columns)
        text_cols = ["question_title", "question_body", "answer"]
        for df in [train_df, val_df, test_df]:
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

        # Save to cache
        print(f"Saving processed data to {Config.WORKING_DIR}...")
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df


class QADataset(Dataset):
    """
    Dataset class for Question-Answer Labeling.
    Separates Question (Title + Body) and Answer streams for Dual-Encoder architecture.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        is_test: bool = False,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-fetch columns to numpy arrays for faster access
        self.titles = df["question_title"].values
        self.bodies = df["question_body"].values
        self.answers = df["answer"].values

        # Store targets if not test mode
        if not self.is_test:
            self.targets = df[Config.TARGET_COLS].values

        # Store IDs for tracking
        self.qa_ids = df["qa_id"].values if "qa_id" in df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Construct text inputs
        title = str(self.titles[idx])
        body = str(self.bodies[idx])
        answer = str(self.answers[idx])

        # Question Stream: Concatenate Title and Body
        q_text = title + " " + body

        # Answer Stream: Answer text
        a_text = answer

        # Tokenize Question Stream
        # We do NOT pad here; padding is handled dynamically in Collate
        q_inputs = self.tokenizer(
            q_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        # Tokenize Answer Stream
        a_inputs = self.tokenizer(
            a_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        item = {
            "q_input_ids": torch.tensor(q_inputs["input_ids"], dtype=torch.long),
            "q_attention_mask": torch.tensor(
                q_inputs["attention_mask"], dtype=torch.long
            ),
            "a_input_ids": torch.tensor(a_inputs["input_ids"], dtype=torch.long),
            "a_attention_mask": torch.tensor(
                a_inputs["attention_mask"], dtype=torch.long
            ),
        }

        # Add targets for training/validation
        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        # Add ID if available
        if self.qa_ids is not None:
            item["qa_id"] = self.qa_ids[idx]

        return item


class Collate:
    """
    Collator that implements dynamic padding.
    Pads sequences to the longest sequence in the batch rather than the global max length.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        # Extract question inputs
        q_input_ids = [item["q_input_ids"] for item in batch]
        q_attention_mask = [item["q_attention_mask"] for item in batch]

        # Extract answer inputs
        a_input_ids = [item["a_input_ids"] for item in batch]
        a_attention_mask = [item["a_attention_mask"] for item in batch]

        # Dynamic padding (batch_first=True)
        q_input_ids = pad_sequence(
            q_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        q_attention_mask = pad_sequence(
            q_attention_mask, batch_first=True, padding_value=0
        )

        a_input_ids = pad_sequence(
            a_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        a_attention_mask = pad_sequence(
            a_attention_mask, batch_first=True, padding_value=0
        )

        batch_out = {
            "q_input_ids": q_input_ids,
            "q_attention_mask": q_attention_mask,
            "a_input_ids": a_input_ids,
            "a_attention_mask": a_attention_mask,
        }

        # Stack labels if present
        if "labels" in batch[0]:
            batch_out["labels"] = torch.stack([item["labels"] for item in batch])

        # Pass through IDs if present (as a list)
        if "qa_id" in batch[0]:
            batch_out["qa_id"] = [item["qa_id"] for item in batch]

        return batch_out


def get_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    load_cached_data: bool = True,
    batch_size: int = Config.BATCH_SIZE,
):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        tokenizer: The transformer tokenizer.
        load_cached_data: Whether to use cached parquet files.
        batch_size: Batch size for dataloaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load dataframes
    train_df, val_df, test_df = load_data(load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = QADataset(train_df, tokenizer, Config.MAX_LEN, is_test=False)
    val_dataset = QADataset(val_df, tokenizer, Config.MAX_LEN, is_test=False)
    test_dataset = QADataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)

    # Initialize Collator
    collate_fn = Collate(pad_token_id=tokenizer.pad_token_id)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
