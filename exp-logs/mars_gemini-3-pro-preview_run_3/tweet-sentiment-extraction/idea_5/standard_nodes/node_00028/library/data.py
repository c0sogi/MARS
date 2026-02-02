import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Stores pre-tokenized inputs and targets for the Dual-Head DeBERTa model.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        start_labels=None,
        end_labels=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.start_labels = start_labels
        self.end_labels = end_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        if self.start_labels is not None:
            item["start_labels"] = torch.tensor(
                self.start_labels[idx], dtype=torch.long
            )
            item["end_labels"] = torch.tensor(self.end_labels[idx], dtype=torch.long)

        return item


def process_data(
    df, tokenizer, max_len, is_test=False, cache_prefix="train", load_cached_data=True
):
    """
    Tokenizes data, generates targets, and handles caching to .npy files.
    Filters out 'neutral' rows for training/validation sets.
    """
    cache_dir = Config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    path_input_ids = os.path.join(cache_dir, f"{cache_prefix}_input_ids.npy")
    path_att_masks = os.path.join(cache_dir, f"{cache_prefix}_att_masks.npy")
    path_start = os.path.join(cache_dir, f"{cache_prefix}_start.npy")
    path_end = os.path.join(cache_dir, f"{cache_prefix}_end.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if is_test:
            if os.path.exists(path_input_ids) and os.path.exists(path_att_masks):
                return (
                    np.load(path_input_ids),
                    np.load(path_att_masks),
                    None,
                    None,
                )
        else:
            if (
                os.path.exists(path_input_ids)
                and os.path.exists(path_att_masks)
                and os.path.exists(path_start)
                and os.path.exists(path_end)
            ):
                return (
                    np.load(path_input_ids),
                    np.load(path_att_masks),
                    np.load(path_start),
                    np.load(path_end),
                )

    # 2. Process data from scratch

    # Filter 'neutral' sentiment for training/validation (but keep for test)
    if not is_test:
        df = df[df["sentiment"] != "neutral"].reset_index(drop=True)

    input_ids_list = []
    attention_masks_list = []
    start_labels_list = []
    end_labels_list = []

    for idx, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        # DeBERTa tokenizer handles the pair automatically
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offsets = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        input_ids_list.append(input_ids)
        attention_masks_list.append(attention_mask)

        if not is_test:
            selected_text = str(row["selected_text"])

            # Find the character start/end of the selected_text within the full text
            start_char = text.find(selected_text)
            if start_char == -1:
                # Fallback: if exact match not found, use full text (rare edge case)
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            # Identify tokens corresponding to the 'text' part (sequence_id == 1)
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if not text_token_indices:
                # Handle empty text case
                start_labels_list.append(0)
                end_labels_list.append(0)
                continue

            tokens_in_range = []
            for i in text_token_indices:
                o_start, o_end = offsets[i]
                # Check for overlap between token span and selected_text char span
                if o_start < end_char and o_end > start_char:
                    tokens_in_range.append(i)

            if not tokens_in_range:
                # Fallback if no tokens overlap (e.g., weird tokenization)
                s_idx = text_token_indices[0]
                e_idx = text_token_indices[-1]
            else:
                s_idx = tokens_in_range[0]
                e_idx = tokens_in_range[-1]

            start_labels_list.append(s_idx)
            end_labels_list.append(e_idx)

    # Convert lists to numpy arrays
    input_ids_np = np.array(input_ids_list)
    att_masks_np = np.array(attention_masks_list)

    # 3. Save to cache and return
    if is_test:
        np.save(path_input_ids, input_ids_np)
        np.save(path_att_masks, att_masks_np)
        return input_ids_np, att_masks_np, None, None
    else:
        start_np = np.array(start_labels_list)
        end_np = np.array(end_labels_list)

        np.save(path_input_ids, input_ids_np)
        np.save(path_att_masks, att_masks_np)
        np.save(path_start, start_np)
        np.save(path_end, end_np)

        return input_ids_np, att_masks_np, start_np, end_np


def get_dataloaders(tokenizer, batch_size=32, load_cached_data=True, debug=False):
    """
    Loads training and validation data and returns PyTorch DataLoaders.
    Handles debug mode and neutral filtering implicitly via process_data.
    """
    train_df = pd.read_csv(Config.TRAIN_FILE)
    val_df = pd.read_csv(Config.VAL_FILE)

    prefix_train = "train"
    prefix_val = "val"

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
        prefix_train = "train_debug"
        prefix_val = "val_debug"

    # Process Train Data
    t_ids, t_att, t_start, t_end = process_data(
        train_df,
        tokenizer,
        Config.MAX_LEN,
        is_test=False,
        cache_prefix=prefix_train,
        load_cached_data=load_cached_data,
    )

    # Process Validation Data
    v_ids, v_att, v_start, v_end = process_data(
        val_df,
        tokenizer,
        Config.MAX_LEN,
        is_test=False,
        cache_prefix=prefix_val,
        load_cached_data=load_cached_data,
    )

    train_dataset = TweetDataset(t_ids, t_att, t_start, t_end)
    val_dataset = TweetDataset(v_ids, v_att, v_start, v_end)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_loader(tokenizer, batch_size=32, load_cached_data=True):
    """
    Loads test data and returns a PyTorch DataLoader and the original DataFrame.
    """
    test_df = pd.read_csv(Config.TEST_FILE)

    te_ids, te_att, _, _ = process_data(
        test_df,
        tokenizer,
        Config.MAX_LEN,
        is_test=True,
        cache_prefix="test",
        load_cached_data=load_cached_data,
    )

    test_dataset = TweetDataset(te_ids, te_att)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader, test_df
