import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


def load_and_process_data(load_cached_data=True):
    """
    Loads data from metadata CSVs, performs text augmentation, and handles caching.

    Args:
        load_cached_data (bool): If True, attempts to load from parquet cache.

    Returns:
        tuple: (train_df, val_df, test_df, target_cols)
    """
    # Define cache paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    train_cache = os.path.join(Config.WORKING_DIR, "train_processed_pure.parquet")
    val_cache = os.path.join(Config.WORKING_DIR, "val_processed_pure.parquet")
    test_cache = os.path.join(Config.WORKING_DIR, "test_processed_pure.parquet")

    # Load target column names from sample submission to ensure correct order
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
    target_cols = [col for col in sample_sub.columns if col != "qa_id"]

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(train_cache)
            and os.path.exists(val_cache)
            and os.path.exists(test_cache)
        ):
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(train_cache)
            val_df = pd.read_parquet(val_cache)
            test_df = pd.read_parquet(test_cache)
            return train_df, val_df, test_df, target_cols
        else:
            print("Cache not found. Processing data from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # Load raw metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    # Helper for text augmentation
    def augment_text(df):
        # Fill NaNs
        df["question_title"] = df["question_title"].fillna("")
        df["question_body"] = df["question_body"].fillna("")
        df["answer"] = df["answer"].fillna("")
        df["category"] = df["category"].fillna("UNKNOWN")
        df["host"] = df["host"].fillna("UNKNOWN")

        # Construct Augmented Question
        # Format: "{question_title} {question_body}"
        df["question_text"] = df["question_title"] + " " + df["question_body"]
        return df

    # Apply processing
    print("Augmenting text features...")
    train_df = augment_text(train_df)
    val_df = augment_text(val_df)
    test_df = augment_text(test_df)

    # Save to cache
    print("Saving processed data to cache...")
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df, target_cols


class QuestDataset(Dataset):
    """
    PyTorch Dataset for StackExchange Question-Answer pairs.
    Handles independent tokenization of Question and Answer streams.
    """

    def __init__(self, df, tokenizer, target_cols=None, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.is_test = is_test

        # Extract texts
        self.questions = df["question_text"].tolist()
        self.answers = df["answer"].tolist()

        # Extract targets if available
        if not is_test:
            self.targets = df[target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = str(self.questions[idx])
        answer = str(self.answers[idx])

        # Tokenize Question
        # We do NOT pad here. Padding is handled in the collator for dynamic batching.
        q_encoded = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Tokenize Answer
        a_encoded = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "input_ids_q": q_encoded["input_ids"],
            "attention_mask_q": q_encoded["attention_mask"],
            "input_ids_a": a_encoded["input_ids"],
            "attention_mask_a": a_encoded["attention_mask"],
        }

        if not self.is_test:
            item["labels"] = self.targets[idx]

        # Pass qa_id for tracking if needed (useful for inference/debugging)
        if "qa_id" in self.df.columns:
            item["qa_id"] = self.df.iloc[idx]["qa_id"]

        return item


class Collate:
    """
    Custom Collator for Dynamic Padding.
    Pads sequences in a batch to the longest sequence in that specific batch.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract lists of inputs
        input_ids_q = [item["input_ids_q"] for item in batch]
        attention_mask_q = [item["attention_mask_q"] for item in batch]
        input_ids_a = [item["input_ids_a"] for item in batch]
        attention_mask_a = [item["attention_mask_a"] for item in batch]

        # Determine max lengths in this batch
        max_len_q = max(len(ids) for ids in input_ids_q)
        max_len_a = max(len(ids) for ids in input_ids_a)

        # Pad sequences
        def pad_seq(seqs, max_len, pad_val):
            padded = []
            for seq in seqs:
                pad_len = max_len - len(seq)
                padded.append(seq + [pad_val] * pad_len)
            return torch.tensor(padded, dtype=torch.long)

        batch_input_ids_q = pad_seq(input_ids_q, max_len_q, self.pad_token_id)
        batch_mask_q = pad_seq(attention_mask_q, max_len_q, 0)

        batch_input_ids_a = pad_seq(input_ids_a, max_len_a, self.pad_token_id)
        batch_mask_a = pad_seq(attention_mask_a, max_len_a, 0)

        output = {
            "input_ids_q": batch_input_ids_q,
            "attention_mask_q": batch_mask_q,
            "input_ids_a": batch_input_ids_a,
            "attention_mask_a": batch_mask_a,
        }

        # Handle labels
        if "labels" in batch[0]:
            labels = torch.tensor(
                np.array([item["labels"] for item in batch]), dtype=torch.float32
            )
            output["labels"] = labels

        # Handle qa_id
        if "qa_id" in batch[0]:
            output["qa_ids"] = [item["qa_id"] for item in batch]

        return output


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    # Load data
    train_df, val_df, test_df, target_cols = load_and_process_data(load_cached_data)

    if debug:
        print("DEBUG MODE: Truncating datasets...")
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        test_df = test_df.head(50)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Create Datasets
    train_dataset = QuestDataset(train_df, tokenizer, target_cols, is_test=False)
    val_dataset = QuestDataset(val_df, tokenizer, target_cols, is_test=False)
    test_dataset = QuestDataset(test_df, tokenizer, target_cols=None, is_test=True)

    # Create Collator
    collator = Collate(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, target_cols
