import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class QADataset(Dataset):
    """
    Custom Dataset for Question Answering using mT5.
    Formats input as: "question: <question> context: <context>"
    Implements Gold Window (Train) and Sliding Window (Test) strategies.
    Cite solution_lesson_node_00001: Context Truncation Pitfalls.
    """

    def __init__(self, data, tokenizer, source_max_len, target_max_len, mode="train"):
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.mode = mode  # 'train' or 'test'
        self.samples = self._process_data(data)

    def _process_data(self, df):
        samples = []
        # Cite solution_lesson_node_00005: Exact Token Mapping vs. Character Heuristics.
        # Use token-based windowing to maximize context usage and avoid truncation errors.

        # Approximate overhead for special tokens (<s> question </s> context </s>)
        SPECIAL_TOKENS_COUNT = 3

        for idx, row in df.iterrows():
            context = str(row["context"])
            question = str(row["question"])
            sample_id = row["id"]

            # 1. Tokenize Question to determine available space for context
            q_enc = self.tokenizer(question, add_special_tokens=False)
            q_len = len(q_enc["input_ids"])

            # Calculate max tokens available for context
            max_ctx_tokens = self.source_max_len - q_len - SPECIAL_TOKENS_COUNT
            if max_ctx_tokens < 50:
                max_ctx_tokens = 50

            # 2. Tokenize Context with Offsets
            # Returns input_ids and offset_mapping (start, end chars)
            ctx_enc = self.tokenizer(
                context,
                add_special_tokens=False,
                return_offsets_mapping=True,
                verbose=False,
            )
            ctx_ids = ctx_enc["input_ids"]
            offsets = ctx_enc["offset_mapping"]
            total_tokens = len(ctx_ids)

            if total_tokens == 0:
                continue

            # 3. Create Windows
            if self.mode == "train":
                # Gold Window: Center on answer
                ans_start = int(row["answer_start"])
                ans_text = str(row["answer_text"])
                ans_center = ans_start + len(ans_text) // 2

                # Find the token index corresponding to the answer center
                center_token_idx = 0
                for i, (start, end) in enumerate(offsets):
                    if start <= ans_center < end:
                        center_token_idx = i
                        break
                    if start > ans_center:
                        center_token_idx = max(0, i - 1)
                        break
                else:
                    center_token_idx = total_tokens - 1

                # Define window range (tokens)
                half_window = max_ctx_tokens // 2
                start_token = max(0, center_token_idx - half_window)
                end_token = min(total_tokens, start_token + max_ctx_tokens)

                # Shift back if we hit the end
                if end_token - start_token < max_ctx_tokens and start_token > 0:
                    start_token = max(0, end_token - max_ctx_tokens)

                # Map back to characters
                char_start = offsets[start_token][0]
                char_end = offsets[end_token - 1][1]
                window_context = context[char_start:char_end]

                samples.append(
                    {
                        "id": sample_id,
                        "context": window_context,
                        "question": question,
                        "answer_text": ans_text,
                    }
                )

            else:
                # Sliding Window (Test/Val)
                stride = max_ctx_tokens // 2
                for start_token in range(0, total_tokens, stride):
                    if start_token >= total_tokens and start_token != 0:
                        break

                    end_token = min(total_tokens, start_token + max_ctx_tokens)

                    char_start = offsets[start_token][0]
                    char_end = offsets[end_token - 1][1]
                    window_context = context[char_start:char_end]

                    # Filter tiny fragments
                    if len(window_context) < 10 and start_token != 0:
                        continue

                    sample = {
                        "id": sample_id,
                        "context": window_context,
                        "question": question,
                    }
                    if "answer_text" in row:
                        sample["answer_text"] = str(row["answer_text"])
                    samples.append(sample)

                    if end_token == total_tokens:
                        break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Input formatting
        source_text = f"question: {sample['question']} context: {sample['context']}"

        # Tokenize inputs
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Base item dictionary
        item = {
            "ids": sample["id"],
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "context": sample["context"],
            "question": sample["question"],
        }

        # Pass answer_text if available (for validation scoring)
        if "answer_text" in sample:
            item["answer_text"] = sample["answer_text"]

        # Handle targets (Training only)
        if self.mode == "train":
            target_text = sample["answer_text"]

            # Tokenize targets
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
    # Train: Use Gold Window (contains answer)
    train_dataset = QADataset(
        train_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        mode="train",
    )

    # Val: Use Sliding Window (mode="test") to prevent leakage and match inference logic
    val_dataset = QADataset(
        val_df,
        tokenizer,
        Config.MAX_SOURCE_LENGTH,
        Config.MAX_TARGET_LENGTH,
        mode="test",
    )

    # Test: Use Sliding Window for inference
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
