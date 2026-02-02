import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    WORKING_DIR,
    CACHE_DIR,
    MAX_LEN,
    SENTIMENT_MAP,
    BATCH_SIZE,
    NUM_WORKERS,
    DEBUG,
    DEBUG_SIZE,
    MODEL_NAME,
)
from library.utils import jaccard


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Converts dataframe text to padded indices and generates labels using RoBERTa tokenizer.
    """
    input_ids_list = []
    attention_masks_list = []
    start_ids_list = []
    end_ids_list = []
    sentiment_ids_list = []  # Kept for compatibility/analysis

    for _, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # Tokenize pair: (sentiment, text)
        # RoBERTa format: <s> sentiment </s> </s> text </s>
        encoded = tokenizer(
            sentiment,
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        sentiment_ids_list.append(SENTIMENT_MAP.get(sentiment, 1))

        if not is_test:
            selected_text = str(row["selected_text"])

            # Find character start/end of selected_text in text
            start_char = text.find(selected_text)

            if start_char == -1:
                # Fallback: try stripping
                start_char = text.find(selected_text.strip())

            if start_char != -1:
                end_char = start_char + len(selected_text)
            else:
                # Fallback: Use entire text if match fails (rare)
                start_char = 0
                end_char = len(text)

            # Find token indices corresponding to character span
            # We only look at tokens where sequence_id == 1 (the text part)
            token_start_index = 0
            token_end_index = 0

            # Use char_to_token to map character indices to token indices
            # sequence_index=1 refers to the second sequence (text)
            try:
                idx_start = encoded.char_to_token(1, start_char)
                idx_end = encoded.char_to_token(1, end_char - 1)  # inclusive end char
            except:
                idx_start = None
                idx_end = None

            # Handle cases where char_to_token returns None (e.g., whitespace)
            # Find nearest valid token for start
            if idx_start is None:
                # Iterate to find the first token of the text sequence
                text_start_idx = 0
                while (
                    text_start_idx < len(sequence_ids)
                    and sequence_ids[text_start_idx] != 1
                ):
                    text_start_idx += 1

                # If we couldn't map, default to start of text
                idx_start = text_start_idx
                # Try to refine: scan forward from start_char
                for i in range(start_char, end_char):
                    idx = encoded.char_to_token(1, i)
                    if idx is not None:
                        idx_start = idx
                        break

            # Find nearest valid token for end
            if idx_end is None:
                # Try to refine: scan backward from end_char
                for i in range(end_char - 1, start_char - 1, -1):
                    idx = encoded.char_to_token(1, i)
                    if idx is not None:
                        idx_end = idx
                        break

                if idx_end is None:
                    # Default to last token of sequence 1
                    text_end_idx = len(sequence_ids) - 1
                    while text_end_idx >= 0 and sequence_ids[text_end_idx] != 1:
                        text_end_idx -= 1
                    idx_end = text_end_idx

            # Ensure validity
            if idx_start is None or idx_end is None:
                # Total fallback
                idx_start = 0
                idx_end = 0

            start_ids_list.append(idx_start)
            end_ids_list.append(idx_end)
        else:
            start_ids_list.append(0)
            end_ids_list.append(0)

    return {
        "input_ids": np.array(input_ids_list, dtype=np.int64),
        "attention_masks": np.array(attention_masks_list, dtype=np.int64),
        "sentiment_ids": np.array(sentiment_ids_list, dtype=np.int64),
        "start_ids": np.array(start_ids_list, dtype=np.int64),
        "end_ids": np.array(end_ids_list, dtype=np.int64),
    }


class TweetDataset(Dataset):
    def __init__(self, df, data_dict):
        self.text_ids = df["textID"].values
        self.texts = df["text"].values
        self.sentiments = df["sentiment"].values

        self.input_ids = torch.tensor(data_dict["input_ids"])
        self.attention_masks = torch.tensor(data_dict["attention_masks"])
        self.sentiment_ids = torch.tensor(data_dict["sentiment_ids"])
        self.start_ids = torch.tensor(data_dict["start_ids"])
        self.end_ids = torch.tensor(data_dict["end_ids"])

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "sentiment_id": self.sentiment_ids[idx],
            "start_idx": self.start_ids[idx],
            "end_idx": self.end_ids[idx],
            "text": str(self.texts[idx]),
            "textID": str(self.text_ids[idx]),
            "sentiment": str(self.sentiments[idx]),
        }


def get_loaders(load_cached_data=True):
    """
    Main entry point to get data loaders. Handles caching, tokenizer loading, and batching.
    """
    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Load Dataframes from metadata
    df_train = pd.read_csv(TRAIN_PATH)
    df_val = pd.read_csv(VAL_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Handle missing values
    df_train.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)
    df_val.dropna(subset=["text", "sentiment", "selected_text"], inplace=True)
    df_test.dropna(subset=["text", "sentiment"], inplace=True)

    # Debug Mode
    if DEBUG:
        df_train = df_train.head(DEBUG_SIZE)
        df_val = df_val.head(DEBUG_SIZE)
        df_test = df_test.head(DEBUG_SIZE)

    # Load Tokenizer
    # We use cache_dir for the model artifacts if needed, but here we just load from pretrained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Helper to process or load cache
    def get_cached_or_process(df, split_name, is_test=False):
        # Note: We include MODEL_NAME in cache filename to prevent loading cache from different models
        cache_file = os.path.join(CACHE_DIR, f"{split_name}_{MODEL_NAME}_data.npz")

        if load_cached_data and os.path.exists(cache_file):
            try:
                loaded = np.load(cache_file)
                return {k: loaded[k] for k in loaded.files}
            except Exception:
                pass  # Fallback

        # Process data
        data = process_data(df, tokenizer, MAX_LEN, is_test)

        # Save to cache
        np.savez(cache_file, **data)
        return data

    # Get processed data dictionaries
    train_data = get_cached_or_process(df_train, "train", is_test=False)
    val_data = get_cached_or_process(df_val, "val", is_test=False)
    test_data = get_cached_or_process(df_test, "test", is_test=True)

    # Create Datasets
    train_dataset = TweetDataset(df_train, train_data)
    val_dataset = TweetDataset(df_val, val_data)
    test_dataset = TweetDataset(df_test, test_data)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, tokenizer
