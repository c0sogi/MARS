import os
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


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Converts dataframe text to padded indices and generates labels using Transformers tokenizer.
    Returns offset mapping for reconstruction.
    """
    input_ids_list = []
    attention_masks_list = []
    start_ids_list = []
    end_ids_list = []
    offset_mapping_list = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP] or <s> sentiment </s> </s> text </s>
        enc = tokenizer(
            sentiment,
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]
        sequence_ids = enc.sequence_ids()

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)
        offset_mapping_list.append(offsets)

        if not is_test:
            selected_text = str(row["selected_text"]).strip()

            # Find character start/end in the original text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: if selected_text not found, use whole text
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Find corresponding token indices
            # sequence_ids: None (special), 0 (sentiment), 1 (text)

            # Find start and end of the text segment in tokens
            token_start_index = 0
            while (
                token_start_index < len(sequence_ids)
                and sequence_ids[token_start_index] != 1
            ):
                token_start_index += 1

            token_end_index = len(sequence_ids) - 1
            while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # If text was truncated completely, handle it
            if token_start_index > token_end_index:
                start_ids_list.append(0)
                end_ids_list.append(0)
                continue

            # Check if the answer is within the span
            if (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                # Move i forward until we hit start_char
                i = token_start_index
                while i <= token_end_index and offsets[i][0] <= start_char:
                    i += 1
                start_idx = i - 1

                # Move j backward until we hit end_char
                j = token_end_index
                while j >= token_start_index and offsets[j][1] >= end_char:
                    j -= 1
                end_idx = j + 1

                start_ids_list.append(start_idx)
                end_ids_list.append(end_idx)
            else:
                # If truncated, point to CLS or 0
                start_ids_list.append(0)
                end_ids_list.append(0)
        else:
            start_ids_list.append(0)
            end_ids_list.append(0)

    return {
        "input_ids": np.array(input_ids_list, dtype=np.int64),
        "attention_masks": np.array(attention_masks_list, dtype=np.int64),
        "start_ids": np.array(start_ids_list, dtype=np.int64),
        "end_ids": np.array(end_ids_list, dtype=np.int64),
        "offset_mapping": np.array(offset_mapping_list, dtype=np.int64),
    }


class TweetDataset(Dataset):
    def __init__(self, df, data_dict):
        self.text_ids = df["textID"].values
        self.texts = df["text"].values
        self.sentiments = df["sentiment"].values

        self.input_ids = torch.tensor(data_dict["input_ids"])
        self.attention_masks = torch.tensor(data_dict["attention_masks"])
        self.start_ids = torch.tensor(data_dict["start_ids"])
        self.end_ids = torch.tensor(data_dict["end_ids"])
        self.offset_mapping = torch.tensor(data_dict["offset_mapping"])

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "start_idx": self.start_ids[idx],
            "end_idx": self.end_ids[idx],
            "offset_mapping": self.offset_mapping[idx],
            "text": str(self.texts[idx]),
            "textID": str(self.text_ids[idx]),
            "sentiment": str(self.sentiments[idx]),
        }


def get_loaders(load_cached_data=True):
    """
    Main entry point to get data loaders.
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Helper to process or load cache
    def get_cached_or_process(df, split_name, is_test=False):
        # We don't cache for this run to avoid compatibility issues with previous vocab-based cache
        data = process_data(df, tokenizer, MAX_LEN, is_test)
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
