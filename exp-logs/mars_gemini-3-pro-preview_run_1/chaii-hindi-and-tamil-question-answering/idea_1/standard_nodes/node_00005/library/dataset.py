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
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.mode = mode  # 'train', 'val', 'test'

        if mode == "train":
            self.data = data
        else:
            # Cite solution_lesson_node_00004: Exhaustive Sliding Windows
            # Pre-process windows for validation and test to ensure full coverage
            self.samples = []
            # We'll use a conservative estimate for query length (e.g., 64 tokens)
            estimated_budget = source_max_len - 64 - 20
            self.doc_stride = estimated_budget // 2

            for _, row in data.iterrows():
                self._create_windows(row)

    def _create_windows(self, row):
        question = row["question"]
        context = row["context"]
        sample_id = row["id"]

        # Tokenize question to determine actual budget
        q_enc = self.tokenizer(question, add_special_tokens=False)
        q_ids = q_enc.input_ids
        budget = self.source_max_len - len(q_ids) - 20

        # Tokenize context
        c_enc = self.tokenizer(
            context, add_special_tokens=False, return_offsets_mapping=True
        )
        c_ids = c_enc.input_ids

        # If context fits, just one window
        if len(c_ids) <= budget:
            sample = {
                "id": sample_id,
                "context": context,
                "question": question,
                "window_text": context,
            }
            if "answer_text" in row:
                sample["answer_text"] = row["answer_text"]
            self.samples.append(sample)
            return

        # Sliding window
        for i in range(0, len(c_ids), self.doc_stride):
            end_i = min(len(c_ids), i + budget)

            span_start = c_enc.offset_mapping[i][0]
            span_end = c_enc.offset_mapping[end_i - 1][1]
            window_text = context[span_start:span_end]

            sample = {
                "id": sample_id,
                "context": context,
                "question": question,
                "window_text": window_text,
            }
            if "answer_text" in row:
                sample["answer_text"] = row["answer_text"]

            self.samples.append(sample)

            if end_i == len(c_ids):
                break

    def __len__(self):
        if self.mode == "train":
            return len(self.data)
        else:
            return len(self.samples)

    def __getitem__(self, index):
        if self.mode == "train":
            row = self.data.iloc[index]
            question = row["question"]
            context = row["context"]
            sample_id = row["id"]

            # Tokenize question
            q_enc = self.tokenizer(question, add_special_tokens=False)
            q_ids = q_enc.input_ids
            budget = self.source_max_len - len(q_ids) - 20

            # Tokenize context
            c_enc = self.tokenizer(
                context, add_special_tokens=False, return_offsets_mapping=True
            )
            c_ids = c_enc.input_ids

            best_window_text = context
            if len(c_ids) > budget:
                # Oracle Strategy: Center around answer
                ans_start_char = int(row["answer_start"])
                token_idx = c_enc.char_to_token(ans_start_char)

                if token_idx is None:
                    for offset in range(1, 20):
                        token_idx = c_enc.char_to_token(ans_start_char + offset)
                        if token_idx is not None:
                            break
                if token_idx is None:
                    token_idx = 0

                half_window = budget // 2
                start_idx = max(0, token_idx - half_window)
                end_idx = min(len(c_ids), start_idx + budget)

                if end_idx - start_idx < budget and start_idx > 0:
                    start_idx = max(0, end_idx - budget)

                span_start = c_enc.offset_mapping[start_idx][0]
                span_end = c_enc.offset_mapping[end_idx - 1][1]
                best_window_text = context[span_start:span_end]

            target_text = str(row["answer_text"])

        else:
            # Val/Test mode: use pre-calculated windows
            sample = self.samples[index]
            question = sample["question"]
            best_window_text = sample["window_text"]
            context = sample["context"]
            sample_id = sample["id"]
            if "answer_text" in sample:
                target_text = str(sample["answer_text"])
            else:
                target_text = None

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
            "ids": sample_id,
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "context": context,
            "question": question,
        }

        if target_text is not None:
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
