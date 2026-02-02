import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class TweetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Fix for Parquet loading issue where nested lists become object arrays
        offsets = item["offsets"]
        if isinstance(offsets, np.ndarray):
            offsets = offsets.tolist()

        return {
            "ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(item["token_type_ids"], dtype=torch.long),
            "targets_start": torch.tensor(item["start_idx"], dtype=torch.long),
            "targets_end": torch.tensor(item["end_idx"], dtype=torch.long),
            "orig_tweet": item["text"],
            "sentiment": item["sentiment"],
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "textID": item["textID"],
        }


def process_data(df, tokenizer, max_len, is_test=False):
    """
    Processes the dataframe into a list of features suitable for the model.
    Computes token-level targets based on character-level selected_text.
    """
    data_list = []

    for _, row in df.iterrows():
        text = str(row["text"])
        textID = str(row["textID"])
        sentiment = str(row["sentiment"])

        # For test set, selected_text is not available
        selected_text = " "
        if not is_test:
            selected_text = str(row["selected_text"])

        # Tokenize with offsets
        # encode_plus handles: [CLS] sentiment [SEP] text [SEP] or <s> sentiment </s> </s> text </s>
        inputs = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # RoBERTa does not return token_type_ids by default, so we provide a default
        token_type_ids = inputs.get("token_type_ids", [0] * max_len)

        # Convert offsets to list of lists for safe serialization
        offsets = [list(o) for o in inputs["offset_mapping"]]

        start_idx = 0
        end_idx = 0

        if not is_test:
            # 1. Find character indices of selected_text in text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback if exact match fails (rare): use full text
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # 2. Identify which tokens correspond to the text part
            sequence_ids = inputs.sequence_ids()
            # sequence_id 0 is sentiment, 1 is text, None is special tokens
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not text_token_indices:
                # Safety fallback
                start_idx = 0
                end_idx = 0
            else:
                # 3. Find tokens overlapping with the character span
                # We want the first token that overlaps with start_char
                # and the last token that overlaps with end_char

                found_start = False
                token_start_index = text_token_indices[0]
                token_end_index = text_token_indices[-1]

                for i in text_token_indices:
                    # offsets[i] is [start_char_idx, end_char_idx] in the original text string
                    tok_start, tok_end = offsets[i]

                    # Check for start match
                    # A token covers the start if the start_char falls within it
                    # OR if it's the first token after the start (if start falls in a gap)
                    if not found_start:
                        if start_char < tok_end:
                            token_start_index = i
                            found_start = True

                    # Check for end match
                    # A token is part of the selection if it starts before the end_char
                    if tok_start < end_char:
                        token_end_index = i

                start_idx = token_start_index
                end_idx = token_end_index

        data_list.append(
            {
                "textID": textID,
                "text": text,
                "sentiment": sentiment,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "offsets": offsets,
            }
        )

    return data_list


def get_loaders(model_name, batch_size=None, load_cached_data=True, debug=False):
    """
    Creates DataLoaders for train, validation, and test sets.
    Handles caching of processed data to Parquet files.
    """
    if batch_size is None:
        batch_size = Config.TRAIN_BATCH_SIZE

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Construct cache paths based on model name to support the ensemble
    safe_name = model_name.replace("/", "_")
    cache_dir = Config.ARTIFACTS_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, f"cached_train_{safe_name}.parquet")
    val_cache = os.path.join(cache_dir, f"cached_val_{safe_name}.parquet")
    test_cache = os.path.join(cache_dir, f"cached_test_{safe_name}.parquet")

    # --- Train Data ---
    if load_cached_data and os.path.exists(train_cache):
        print(f"Loading cached train data from {train_cache}")
        df_train_processed = pd.read_parquet(train_cache)
        train_data = df_train_processed.to_dict("records")
    else:
        print(f"Processing train data for {model_name}...")
        df_train = pd.read_csv(Config.TRAIN_META_PATH)
        if debug:
            df_train = df_train.head(Config.DEBUG_SAMPLE_SIZE)
        train_data = process_data(df_train, tokenizer, Config.MAX_LEN, is_test=False)
        pd.DataFrame(train_data).to_parquet(train_cache)

    # --- Validation Data ---
    if load_cached_data and os.path.exists(val_cache):
        print(f"Loading cached val data from {val_cache}")
        df_val_processed = pd.read_parquet(val_cache)
        val_data = df_val_processed.to_dict("records")
    else:
        print(f"Processing val data for {model_name}...")
        df_val = pd.read_csv(Config.VAL_META_PATH)
        if debug:
            df_val = df_val.head(Config.DEBUG_SAMPLE_SIZE)
        val_data = process_data(df_val, tokenizer, Config.MAX_LEN, is_test=False)
        pd.DataFrame(val_data).to_parquet(val_cache)

    # --- Test Data ---
    if load_cached_data and os.path.exists(test_cache):
        print(f"Loading cached test data from {test_cache}")
        df_test_processed = pd.read_parquet(test_cache)
        test_data = df_test_processed.to_dict("records")
    else:
        print(f"Processing test data for {model_name}...")
        df_test = pd.read_csv(Config.TEST_META_PATH)
        if debug:
            df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)
        test_data = process_data(df_test, tokenizer, Config.MAX_LEN, is_test=True)
        pd.DataFrame(test_data).to_parquet(test_cache)

    # Create Datasets
    train_dataset = TweetDataset(train_data)
    val_dataset = TweetDataset(val_data)
    test_dataset = TweetDataset(test_data)

    # Create DataLoaders
    # We use standard shuffling for training to ensure robust convergence.
    # Static padding to MAX_LEN (128) is used, so smart batching is not strictly necessary for efficiency here.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
