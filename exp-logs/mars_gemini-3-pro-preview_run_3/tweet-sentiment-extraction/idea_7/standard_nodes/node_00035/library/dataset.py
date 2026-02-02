import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    Dataset class for Sentiment Extraction.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        start_positions=None,
        end_positions=None,
        texts=None,
        offsets=None,
        sentiments=None,
        selected_texts=None,
        is_test=False,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.texts = texts
        self.offsets = offsets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.is_test = is_test

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if not self.is_test:
            item["start_positions"] = torch.tensor(
                self.start_positions[idx], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                self.end_positions[idx], dtype=torch.long
            )

            # Include metadata for validation scoring
            if self.texts is not None:
                item["text"] = self.texts[idx]
            if self.selected_texts is not None:
                item["selected_text"] = self.selected_texts[idx]
            if self.sentiments is not None:
                item["sentiment"] = self.sentiments[idx]
            if self.offsets is not None:
                item["offsets"] = torch.tensor(self.offsets[idx], dtype=torch.long)
        else:
            # Inference mode metadata
            if self.texts is not None:
                item["text"] = self.texts[idx]
            if self.sentiments is not None:
                item["sentiment"] = self.sentiments[idx]
            if self.offsets is not None:
                item["offsets"] = torch.tensor(self.offsets[idx], dtype=torch.long)

        return item


def process_data_to_arrays(df, tokenizer, max_len, is_test=False):
    """
    Tokenizes data and generates targets. Filters out invalid samples.
    """
    input_ids_list = []
    attention_mask_list = []
    start_positions_list = []
    end_positions_list = []
    offsets_list = []

    # Keep track of valid indices to filter the dataframe later
    valid_indices = []

    texts = df["text"].values
    sentiments = df["sentiment"].values

    if not is_test:
        selected_texts = df["selected_text"].values

    for i in range(len(df)):
        text = " " + " ".join(str(texts[i]).split())

        # Tokenize
        encoded = tokenizer.encode_plus(
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

        if is_test:
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            offsets_list.append(offsets)
            valid_indices.append(i)
        else:
            selected_text = " " + " ".join(str(selected_texts[i]).split())

            # Find char start/end
            start_idx = text.find(selected_text)

            # Alignment Filtering: If text not found, skip
            if start_idx == -1:
                continue

            end_idx = start_idx + len(selected_text)

            # Find token start/end
            token_start_index = 0
            token_end_index = 0

            # Look for the first token that overlaps with the start_idx
            found_start = False
            for idx, (o_start, o_end) in enumerate(offsets):
                if o_start == 0 and o_end == 0:
                    continue  # Skip special tokens if offset is 0,0 (unless it's CLS/SEP which we handle)
                if o_start < end_idx and o_end > start_idx:
                    if not found_start:
                        token_start_index = idx
                        found_start = True
                    token_end_index = idx

            # If we didn't find any overlapping tokens (rare but possible with aggressive truncation or bad alignment)
            if not found_start:
                continue

            # Ensure targets are within max_len
            if token_start_index >= max_len or token_end_index >= max_len:
                continue

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            start_positions_list.append(token_start_index)
            end_positions_list.append(token_end_index)
            offsets_list.append(offsets)
            valid_indices.append(i)

    return (
        np.array(input_ids_list),
        np.array(attention_mask_list),
        np.array(start_positions_list),
        np.array(end_positions_list),
        np.array(offsets_list),
        valid_indices,
    )


def get_data(load_cached_data=True):
    """
    Loads, cleans, and processes the training data.
    Handles caching to speed up subsequent runs.
    """
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    cache_df_path = os.path.join(cache_dir, "cached_df_pos_neg.parquet")
    cache_arrays_path = os.path.join(cache_dir, "cached_arrays_pos_neg.npz")

    # 1. Try to load cache
    if (
        load_cached_data
        and os.path.exists(cache_df_path)
        and os.path.exists(cache_arrays_path)
    ):
        print("Loading cached data...")
        df_filtered = pd.read_parquet(cache_df_path)
        arrays = np.load(cache_arrays_path)
        input_ids = arrays["input_ids"]
        attention_mask = arrays["attention_mask"]
        start_positions = arrays["start_positions"]
        end_positions = arrays["end_positions"]
        offsets = arrays["offsets"]
        return (
            df_filtered,
            input_ids,
            attention_mask,
            start_positions,
            end_positions,
            offsets,
        )

    # 2. Process from scratch
    print("Processing data from scratch...")

    # Load full original train set
    df = pd.read_csv(Config.original_train_path)

    # Drop NaNs
    df.dropna(subset=["text", "selected_text", "sentiment"], inplace=True)

    # Neutral Exclusion
    df = df[df["sentiment"] != "neutral"].reset_index(drop=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Process
    (
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
        offsets,
        valid_indices,
    ) = process_data_to_arrays(df, tokenizer, Config.max_len, is_test=False)

    # Filter DataFrame to match valid processed samples
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)

    # Save cache
    print(f"Saving cache to {cache_dir}...")
    df_filtered.to_parquet(cache_df_path, index=False)
    np.savez(
        cache_arrays_path,
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
        offsets=offsets,
    )

    return (
        df_filtered,
        input_ids,
        attention_mask,
        start_positions,
        end_positions,
        offsets,
    )


def get_loaders(fold):
    """
    Creates DataLoaders for a specific fold using StratifiedKFold.
    """
    seed_everything(Config.seed)

    # Load processed data
    df, input_ids, attention_mask, start_positions, end_positions, offsets = get_data(
        load_cached_data=True
    )

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # We stratify by sentiment (positive/negative)
    splits = list(skf.split(df, df["sentiment"]))
    train_idx, val_idx = splits[fold]

    # Create Train Dataset
    train_dataset = TweetDataset(
        input_ids=input_ids[train_idx],
        attention_mask=attention_mask[train_idx],
        start_positions=start_positions[train_idx],
        end_positions=end_positions[train_idx],
        # We don't strictly need metadata for training, but can pass if needed
        texts=None,
        offsets=None,
        sentiments=None,
        selected_texts=None,
        is_test=False,
    )

    # Create Validation Dataset (includes metadata for metric calculation)
    val_dataset = TweetDataset(
        input_ids=input_ids[val_idx],
        attention_mask=attention_mask[val_idx],
        start_positions=start_positions[val_idx],
        end_positions=end_positions[val_idx],
        texts=df.iloc[val_idx]["text"].values,
        offsets=offsets[val_idx],
        sentiments=df.iloc[val_idx]["sentiment"].values,
        selected_texts=df.iloc[val_idx]["selected_text"].values,
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Creates DataLoader for the test set.
    """
    df_test = pd.read_csv(Config.test_metadata_path)
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Preprocess test data
    input_ids_list = []
    attention_mask_list = []
    offsets_list = []

    texts = df_test["text"].values
    # Fill NaNs in text just in case, though metadata script handles it
    texts = [str(t) for t in texts]

    for text in texts:
        text = " " + " ".join(text.split())
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids_list.append(encoded["input_ids"])
        attention_mask_list.append(encoded["attention_mask"])
        offsets_list.append(encoded["offset_mapping"])

    test_dataset = TweetDataset(
        input_ids=np.array(input_ids_list),
        attention_mask=np.array(attention_mask_list),
        texts=texts,
        offsets=np.array(offsets_list),
        sentiments=df_test["sentiment"].values,
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return test_loader, df_test
