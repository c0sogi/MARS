import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.utils import set_seed

# Constants
CACHE_DIR = "./working/idea_16"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")


class QADataset(Dataset):
    """
    Dataset class for StackExchange Question-Answer pairs.
    Handles dual-stream tokenization for DistilRoBERTa.
    """

    def __init__(self, df, tokenizer, target_cols=None, max_length=512, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.target_cols = target_cols

        # Pre-extract text data to avoid overhead in __getitem__
        self.titles = self.df["question_title"].astype(str).tolist()
        self.bodies = self.df["question_body"].astype(str).tolist()
        self.answers = self.df["answer"].astype(str).tolist()

        if not self.is_test:
            self.labels = self.df[self.target_cols].values.astype("float32")

        self.qa_ids = self.df["qa_id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = self.titles[idx]
        body = self.bodies[idx]
        answer = self.answers[idx]

        # Stream A: Question (Title + Body)
        # Tokenizer handles [CLS] Title [SEP] Body [SEP] automatically with pair inputs
        q_enc = self.tokenizer(
            title,
            body,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # Stream B: Answer
        # [CLS] Answer [SEP]
        a_enc = self.tokenizer(
            answer,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        item = {
            "qa_id": self.qa_ids[idx],
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "a_input_ids": a_enc["input_ids"],
            "a_attention_mask": a_enc["attention_mask"],
        }

        if not self.is_test:
            item["labels"] = self.labels[idx]

        return item


class CollateQA:
    """
    Custom collator to handle dynamic padding for two separate input streams.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # Extract lists
        q_input_ids = [x["q_input_ids"] for x in batch]
        q_attention_mask = [x["q_attention_mask"] for x in batch]
        a_input_ids = [x["a_input_ids"] for x in batch]
        a_attention_mask = [x["a_attention_mask"] for x in batch]
        qa_ids = [x["qa_id"] for x in batch]

        # Dynamic padding function
        def pad_seqs(seqs, pad_val):
            max_len = max(len(s) for s in seqs)
            padded = []
            for s in seqs:
                pad_len = max_len - len(s)
                padded.append(s + [pad_val] * pad_len)
            return torch.tensor(padded, dtype=torch.long)

        # Pad everything
        batch_q_ids = pad_seqs(q_input_ids, self.pad_token_id)
        batch_q_mask = pad_seqs(q_attention_mask, 0)
        batch_a_ids = pad_seqs(a_input_ids, self.pad_token_id)
        batch_a_mask = pad_seqs(a_attention_mask, 0)

        result = {
            "qa_id": torch.tensor(qa_ids, dtype=torch.long),
            "q_input_ids": batch_q_ids,
            "q_attention_mask": batch_q_mask,
            "a_input_ids": batch_a_ids,
            "a_attention_mask": batch_a_mask,
        }

        # Handle labels if present
        if "labels" in batch[0]:
            labels = [x["labels"] for x in batch]
            result["labels"] = torch.tensor(labels, dtype=torch.float32)

        return result


def load_and_preprocess(load_cached_data=True):
    """
    Loads data from metadata or cache.
    Returns train_df, val_df, test_df, target_cols
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")

    # Identify target columns from sample submission
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    target_cols = [c for c in sample_sub.columns if c != "qa_id"]

    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        print("Loading cached data from parquet...")
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        print("Processing data from metadata...")
        # Load metadata
        train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

        # Fill NaNs in text columns
        text_cols = ["question_title", "question_body", "answer"]
        for df in [train_df, val_df, test_df]:
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)

        # Cache the processed dataframes
        train_df.to_parquet(train_cache, index=False)
        val_df.to_parquet(val_cache, index=False)
        test_df.to_parquet(test_cache, index=False)

    return train_df, val_df, test_df, target_cols


def get_dataloaders(
    batch_size=16, max_length=512, load_cached_data=True, num_workers=2, seed=42
):
    """
    Main entry point to get DataLoaders.
    """
    set_seed(seed)

    # Load Data
    train_df, val_df, test_df, target_cols = load_and_preprocess(
        load_cached_data=load_cached_data
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Create Datasets
    train_dataset = QADataset(
        train_df, tokenizer, target_cols, max_length, is_test=False
    )
    val_dataset = QADataset(val_df, tokenizer, target_cols, max_length, is_test=False)
    test_dataset = QADataset(test_df, tokenizer, target_cols, max_length, is_test=True)

    # Create Collator
    collator = CollateQA(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, target_cols
