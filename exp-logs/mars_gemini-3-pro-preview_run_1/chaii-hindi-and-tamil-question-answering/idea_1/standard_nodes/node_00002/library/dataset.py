import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    Custom Dataset for Question Answering using mT5.
    Implements Sliding Window strategy to handle long contexts.
    """

    def __init__(self, data, tokenizer, source_max_len, target_max_len, mode="train"):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.mode = mode  # 'train', 'val', 'test'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        question = row["question"]
        context = row["context"]

        # Tokenize question to determine available budget
        q_enc = self.tokenizer(question, add_special_tokens=False)
        q_ids = q_enc.input_ids

        # Reserve space for special tokens and formatting (approximate safety margin)
        budget = self.source_max_len - len(q_ids) - 20

        # Tokenize full context with offsets
        c_enc = self.tokenizer(
            context, add_special_tokens=False, return_offsets_mapping=True
        )
        c_ids = c_enc.input_ids

        best_window_text = context  # Default to full context if it fits

        if len(c_ids) > budget:
            # Need to select a window
            if self.mode == "train":
                # Oracle Strategy: Center around answer
                ans_start_char = int(row["answer_start"])
                token_idx = c_enc.char_to_token(ans_start_char)

                # If exact start not found (e.g. whitespace), search nearby
                if token_idx is None:
                    for offset in range(1, 20):
                        token_idx = c_enc.char_to_token(ans_start_char + offset)
                        if token_idx is not None:
                            break

                if token_idx is None:
                    token_idx = 0  # Fallback

                # Center the window
                half_window = budget // 2
                start_idx = max(0, token_idx - half_window)
                end_idx = min(len(c_ids), start_idx + budget)

                # Adjust if we hit the end
                if end_idx - start_idx < budget and start_idx > 0:
                    start_idx = max(0, end_idx - budget)

                # Extract text span using offsets
                span_start = c_enc.offset_mapping[start_idx][0]
                span_end = c_enc.offset_mapping[end_idx - 1][1]
                best_window_text = context[span_start:span_end]

            else:
                # Retrieval Strategy (Val/Test): Sliding window based on question overlap
                stride = budget // 2
                best_score = -1
                best_window_text = ""

                q_set = set(q_ids)

                for i in range(0, len(c_ids), stride):
                    end_i = min(len(c_ids), i + budget)
                    window_ids = c_ids[i:end_i]

                    # Simple overlap score (intersection count)
                    score = len(set(window_ids) & q_set)

                    if score > best_score:
                        best_score = score
                        span_start = c_enc.offset_mapping[i][0]
                        span_end = c_enc.offset_mapping[end_i - 1][1]
                        best_window_text = context[span_start:span_end]

                    if end_i == len(c_ids):
                        break

                if not best_window_text:
                    best_window_text = context[:2000]  # Fallback

        # Construct final input
        source_text = f"question: {question} context: {best_window_text}"

        # Tokenize final input
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "ids": row["id"],
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "context": context,  # Keep full context for post-processing
            "question": question,
        }

        # Handle targets
        if self.mode in ["train", "val"]:
            target_text = str(row["answer_text"])
            target_encoding = self.tokenizer(
                text_target=target_text,
                max_length=self.target_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = target_encoding["input_ids"].squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels
            item["answer_text"] = target_text

        return item


def _load_and_cache_data(file_path, cache_name, load_cached_data):
    """
    Helper function to load data from CSV or Parquet cache.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.parquet")

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    # 2. Load from source
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Ensure text columns are strings
    text_cols = ["context", "question"]
    if "answer_text" in df.columns:
        text_cols.append("answer_text")

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def get_dataloaders(load_cached_data=True):
    """
    Initializes the tokenizer and creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load DataFrames with Caching Logic
    train_df = _load_and_cache_data(
        Config.TRAIN_PATH, "train_processed", load_cached_data
    )
    val_df = _load_and_cache_data(Config.VAL_PATH, "val_processed", load_cached_data)
    test_df = _load_and_cache_data(Config.TEST_PATH, "test_processed", load_cached_data)

    # Debugging: Subsample if configured
    if Config.DEBUG:
        train_df = train_df.head(20)
        val_df = val_df.head(10)
        test_df = test_df.head(10)

    # Create Datasets
    train_dataset = QADataset(
        train_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        mode="train",
    )

    val_dataset = QADataset(
        val_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        mode="val",
    )

    test_dataset = QADataset(
        test_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        mode="test",
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader, tokenizer
