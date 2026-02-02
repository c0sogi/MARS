import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Extraction.
    Wraps pre-processed tensors and metadata.
    """

    def __init__(self, data):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data.get("token_type_ids")
        self.offsets = data["offsets"]

        # Labels for training
        self.start_labels = data.get("start_labels")
        self.end_labels = data.get("end_labels")

        # Metadata for inference/validation
        self.orig_text = data.get("orig_text")
        self.sentiment = data.get("sentiment")
        self.selected_text = data.get("selected_text")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[idx], dtype=torch.long),
            "orig_text": str(self.orig_text[idx]),
            "sentiment": str(self.sentiment[idx]),
        }

        if self.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(
                self.token_type_ids[idx], dtype=torch.long
            )

        if self.start_labels is not None:
            item["start_labels"] = torch.tensor(
                self.start_labels[idx], dtype=torch.long
            )
            item["end_labels"] = torch.tensor(self.end_labels[idx], dtype=torch.long)
            item["selected_text"] = str(self.selected_text[idx])

        return item


def process_data(df, tokenizer, max_len, is_train=True, filter_neutral=False):
    """
    Tokenizes data and generates targets using Mask-Based Overlap.
    """
    # Arrays to store processed data
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    offsets_list = []
    start_labels_list = []
    end_labels_list = []
    orig_text_list = []
    sentiment_list = []
    selected_text_list = []

    # Filter neutral sentiment for training if requested
    if is_train and filter_neutral:
        initial_len = len(df)
        df = df[df["sentiment"] != "neutral"].copy()
        # print(f"Filtered neutral tweets: {initial_len} -> {len(df)}")

    for idx, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # --- Target Preparation ---
        start_idx = -1
        end_idx = -1
        selected_text = text  # Default to full text for inference

        if is_train:
            selected_text = str(row["selected_text"])

            # Find the character span of selected_text within text
            start_idx = text.find(selected_text)

            # Fallback: try stripping whitespace if exact match fails
            if start_idx == -1:
                stripped_sel = selected_text.strip()
                start_idx = text.find(stripped_sel)
                if start_idx != -1:
                    selected_text = stripped_sel

            # Alignment Filtering: If we still can't find it, skip this sample
            if start_idx == -1:
                continue

            end_idx = start_idx + len(selected_text)

        # --- Tokenization ---
        # Input format: [CLS] sentiment [SEP] text [SEP]
        encoding = tokenizer(
            sentiment,
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", [0] * max_len)
        offsets = encoding["offset_mapping"]
        sequence_ids = encoding.sequence_ids()

        # Identify tokens belonging to the 'text' part (sequence_id == 1)
        text_token_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]

        if not text_token_indices:
            continue

        # --- Label Generation (Mask-Based Overlap) ---
        if is_train:
            # Create a binary mask for the character span
            char_targets = np.zeros(len(text))
            char_targets[start_idx:end_idx] = 1

            target_tokens = []
            for i in text_token_indices:
                # offsets[i] contains (start_char, end_char) relative to the text string
                token_start, token_end = offsets[i]

                # Skip special tokens or zero-length tokens
                if token_start == token_end:
                    continue

                # Check for any overlap between token span and target char span
                if np.sum(char_targets[token_start:token_end]) > 0:
                    target_tokens.append(i)

            # If no tokens overlap (rare, but possible with weird tokenization/spacing), skip
            if not target_tokens:
                continue

            start_label = target_tokens[0]
            end_label = target_tokens[-1]

            start_labels_list.append(start_label)
            end_labels_list.append(end_label)
            selected_text_list.append(selected_text)

        # Append to lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        token_type_ids_list.append(token_type_ids)
        offsets_list.append(offsets)
        orig_text_list.append(text)
        sentiment_list.append(sentiment)

    # Convert to dictionary of numpy arrays
    data = {
        "input_ids": np.array(input_ids_list),
        "attention_mask": np.array(attention_mask_list),
        "token_type_ids": np.array(token_type_ids_list),
        "offsets": np.array(offsets_list),
        "orig_text": np.array(orig_text_list),
        "sentiment": np.array(sentiment_list),
    }

    if is_train:
        data["start_labels"] = np.array(start_labels_list)
        data["end_labels"] = np.array(end_labels_list)
        data["selected_text"] = np.array(selected_text_list)

    return data


def get_data(
    df,
    tokenizer,
    max_len,
    cache_path,
    load_cached_data=True,
    is_train=True,
    filter_neutral=False,
    debug=False,
):
    """
    Orchestrates data loading, processing, and caching.
    """
    # Handle debugging: subset data
    if debug:
        df = df.head(100).copy()
        cache_path += "_debug"

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Define filenames
    # Tensors are saved as .npy
    tensor_keys = ["input_ids", "attention_mask", "token_type_ids", "offsets"]
    if is_train:
        tensor_keys.extend(["start_labels", "end_labels"])

    npy_files = {k: f"{cache_path}_{k}.npy" for k in tensor_keys}
    # Text metadata saved as parquet to avoid pickle issues
    meta_file = f"{cache_path}_meta.parquet"

    # Check if cache exists
    all_exist = all(os.path.exists(f) for f in npy_files.values()) and os.path.exists(
        meta_file
    )

    if load_cached_data and all_exist:
        # print(f"Loading cached data from {cache_path}...")
        data = {}
        # Load tensors
        for k, f in npy_files.items():
            data[k] = np.load(f)
        # Load metadata
        df_meta = pd.read_parquet(meta_file)
        data["orig_text"] = df_meta["orig_text"].values
        data["sentiment"] = df_meta["sentiment"].values
        if is_train:
            data["selected_text"] = df_meta["selected_text"].values

        return TweetDataset(data)

    # Process data from scratch
    # print(f"Processing data (Cache miss)...")
    data = process_data(df, tokenizer, max_len, is_train, filter_neutral)

    # Save to cache
    # print(f"Saving data to {cache_path}...")
    for k, f in npy_files.items():
        np.save(f, data[k])

    # Save metadata
    meta_dict = {"orig_text": data["orig_text"], "sentiment": data["sentiment"]}
    if is_train:
        meta_dict["selected_text"] = data["selected_text"]

    pd.DataFrame(meta_dict).to_parquet(meta_file)

    return TweetDataset(data)
