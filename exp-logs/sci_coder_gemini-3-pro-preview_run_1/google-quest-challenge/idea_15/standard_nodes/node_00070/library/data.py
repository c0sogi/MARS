import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.utils import seed_everything

# Define constants
CACHE_DIR = "./working/idea_16/"
METADATA_DIR = "./metadata/"
MODEL_NAME = "distilroberta-base"


class QADataset(Dataset):
    def __init__(self, df, tokenizer, target_cols=None, max_len=512, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.target_cols = target_cols
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract text lists to avoid dataframe overhead in loop
        self.questions = df["question_text"].astype(str).tolist()
        self.answers = df["answer"].astype(str).tolist()
        self.qa_ids = df["qa_id"].tolist()

        if not self.is_test:
            self.labels = df[self.target_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Tokenize Question
        q_enc = self.tokenizer(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        # Tokenize Answer
        a_enc = self.tokenizer(
            answer,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            return_attention_mask=True,
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


class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Separate inputs
        q_features = [
            {"input_ids": b["q_input_ids"], "attention_mask": b["q_attention_mask"]}
            for b in batch
        ]
        a_features = [
            {"input_ids": b["a_input_ids"], "attention_mask": b["a_attention_mask"]}
            for b in batch
        ]

        # Dynamic padding using tokenizer
        q_batch = self.tokenizer.pad(q_features, padding="longest", return_tensors="pt")
        a_batch = self.tokenizer.pad(a_features, padding="longest", return_tensors="pt")

        batch_out = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "a_input_ids": a_batch["input_ids"],
            "a_attention_mask": a_batch["attention_mask"],
            "qa_id": torch.tensor([b["qa_id"] for b in batch], dtype=torch.long),
        }

        if "labels" in batch[0]:
            batch_out["labels"] = torch.tensor(
                np.array([b["labels"] for b in batch]), dtype=torch.float32
            )

        return batch_out


def get_target_columns(df):
    """Identifies target columns based on naming convention."""
    return [
        c
        for c in df.columns
        if (c.startswith("question_") or c.startswith("answer_"))
        and c not in ["question_title", "question_body", "answer", "question_text"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def preprocess_df(df):
    """Concatenates title and body, ensures text format."""
    # Fill NAs
    df["question_title"] = df["question_title"].fillna("")
    df["question_body"] = df["question_body"].fillna("")
    df["answer"] = df["answer"].fillna("")

    # Create combined question text
    df["question_text"] = df["question_title"] + " " + df["question_body"]

    return df


def prepare_loaders(load_cached_data=True, batch_size=16, max_len=512, seed=42):
    seed_everything(seed)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Define filenames
    train_cache = os.path.join(CACHE_DIR, "train_processed.parquet")
    val_cache = os.path.join(CACHE_DIR, "val_processed.parquet")
    test_cache = os.path.join(CACHE_DIR, "test_processed.parquet")

    train_df, val_df, test_df = None, None, None

    # 1. Try Loading Cache (Train/Val only)
    if load_cached_data:
        try:
            if os.path.exists(train_cache) and os.path.exists(val_cache):
                print("Loading cached train/val datasets from parquet...")
                train_df = pd.read_parquet(train_cache)
                val_df = pd.read_parquet(val_cache)

                if "question_text" not in train_df.columns:
                    print("Cache invalid. Processing from scratch.")
                    train_df, val_df = None, None
            else:
                print("Cache not found. Processing from scratch.")
        except Exception as e:
            print(f"Error loading cache: {e}. Processing from scratch.")

    # 2. Process Train/Val if not loaded
    if train_df is None:
        print("Loading metadata and processing train/val...")
        train_raw = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
        val_raw = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

        train_df = preprocess_df(train_raw)
        val_df = preprocess_df(val_raw)

        print("Saving processed train/val to cache...")
        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)

    # 3. Always Load Test from Input (Bypass Cache for Test)
    # Cite debug_lesson_2: Bypass Caching for Runtime-Swapped Test Sets
    print("Loading test data from ./input/test.csv...")
    try:
        test_raw = pd.read_csv("./input/test.csv")
    except FileNotFoundError:
        # Fallback for local testing if input/test.csv doesn't exist (e.g. only metadata exists)
        print("Warning: ./input/test.csv not found, falling back to metadata/test.csv")
        test_raw = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    test_df = preprocess_df(test_raw)

    # Identify targets
    target_cols = get_target_columns(train_df)
    print(f"Identified {len(target_cols)} target columns.")

    # Initialize Tokenizer
    print(f"Initializing tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create Datasets
    train_dataset = QADataset(
        train_df, tokenizer, target_cols, max_len=max_len, is_test=False
    )
    val_dataset = QADataset(
        val_df, tokenizer, target_cols, max_len=max_len, is_test=False
    )
    test_dataset = QADataset(
        test_df, tokenizer, target_cols=None, max_len=max_len, is_test=True
    )

    # Create Collator
    collator = DynamicPaddingCollator(tokenizer)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, target_cols
