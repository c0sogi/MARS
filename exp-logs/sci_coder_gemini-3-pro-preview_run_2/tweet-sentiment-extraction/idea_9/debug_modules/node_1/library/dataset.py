import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class TweetDataset(Dataset):
    """
    PyTorch Dataset for Tweet Sentiment Extraction.
    Stores processed data and returns items for the DataLoader.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SmartBatchingCollate:
    """
    Collate function that implements Dynamic Padding.
    Pads the batch to the maximum sequence length found in that specific batch,
    rather than the global maximum length. This significantly speeds up training.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]

        # Determine max length in this batch
        max_len = max(len(ids) for ids in input_ids)

        # Pad sequences
        pad_id = self.tokenizer.pad_token_id
        padded_input_ids = []
        padded_mask = []
        padded_token_type = []

        for i in range(len(batch)):
            diff = max_len - len(input_ids[i])
            padded_input_ids.append(input_ids[i] + [pad_id] * diff)
            padded_mask.append(attention_mask[i] + [0] * diff)
            padded_token_type.append(token_type_ids[i] + [0] * diff)

        # Handle Targets
        if "start_token" in batch[0] and batch[0]["start_token"] is not None:
            start_tokens = torch.tensor(
                [item["start_token"] for item in batch], dtype=torch.long
            )
            end_tokens = torch.tensor(
                [item["end_token"] for item in batch], dtype=torch.long
            )
        else:
            start_tokens = None
            end_tokens = None

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(padded_token_type, dtype=torch.long),
            "start_tokens": start_tokens,
            "end_tokens": end_tokens,
            "text": [item["text"] for item in batch],
            "selected_text": [item.get("selected_text", "") for item in batch],
            "sentiment": [item["sentiment"] for item in batch],
            "offsets": [item["offsets"] for item in batch],
        }


class SmartBatchingSampler(Sampler):
    """
    Sampler that groups samples of similar lengths into batches.
    This works in tandem with SmartBatchingCollate to minimize padding tokens.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        # Precompute lengths for sorting
        self.lengths = [len(x["input_ids"]) for x in data_source]

    def __iter__(self):
        # Sort indices by sequence length
        indices = np.argsort(self.lengths)

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batches.append(indices[i : i + self.batch_size])

        # Shuffle the batches (so we don't train on all short then all long)
        np.random.shuffle(batches)

        # Flatten back to a list of indices
        shuffled_indices = [idx for batch in batches for idx in batch]
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.data_source)


def process_data(df, tokenizer, max_len):
    """
    Tokenizes data and calculates start/end token indices for the selected text.
    """
    data = []

    for idx, row in df.iterrows():
        text = str(row["text"])
        sentiment = str(row["sentiment"])
        selected_text = str(row["selected_text"]) if "selected_text" in row else None

        # Tokenize: [CLS] sentiment [SEP] text [SEP]
        encoded = tokenizer(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"]
        offsets = encoded["offset_mapping"]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "text": text,
            "sentiment": sentiment,
            "offsets": offsets,
        }

        # Calculate targets if selected_text is present
        if selected_text is not None and not pd.isna(selected_text):
            # Find character indices of selected_text in text
            start_idx = text.find(selected_text)
            end_idx = start_idx + len(selected_text)

            start_token = 0
            end_token = 0

            if start_idx != -1:
                # Identify which tokens correspond to the 'text' part (sequence_id == 1)
                seq_ids = encoded.sequence_ids()

                # Find the span of tokens that overlap with the character span
                found_start = False

                for i, seq_id in enumerate(seq_ids):
                    if seq_id != 1:
                        continue

                    o_start, o_end = offsets[i]

                    # Determine Start Token
                    # We look for the first token that overlaps with the start_idx
                    if not found_start:
                        if o_start <= start_idx < o_end:
                            start_token = i
                            found_start = True
                        elif o_start >= start_idx:
                            # Fallback: if start_idx was in a skipped space, take next token
                            start_token = i
                            found_start = True

                    # Determine End Token
                    # We update end_token as long as the token overlaps with the span
                    if o_start < end_idx:
                        end_token = i

                # Safety check
                if end_token < start_token:
                    end_token = start_token

            item["start_token"] = start_token
            item["end_token"] = end_token
            item["selected_text"] = selected_text
        else:
            item["start_token"] = None
            item["end_token"] = None
            item["selected_text"] = ""

        data.append(item)

    return data


def get_data(file_path, tokenizer, max_len, cache_dir, load_cached_data=True):
    """
    Loads data from CSV or Cache.
    Implements strict caching logic using Parquet.
    """
    # Construct cache filename
    filename = os.path.basename(file_path).replace(".csv", "")
    debug_suffix = "_debug" if Config.DEBUG else ""
    cache_file = os.path.join(
        cache_dir, f"cached_{filename}_{max_len}{debug_suffix}.parquet"
    )

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        try:
            df = pd.read_parquet(cache_file)
            # Convert DataFrame back to list of dicts
            data = df.to_dict(orient="records")
            # Ensure lists are actual lists (parquet might load as arrays)
            for item in data:
                item["input_ids"] = list(item["input_ids"])
                item["attention_mask"] = list(item["attention_mask"])
                item["token_type_ids"] = list(item["token_type_ids"])
                item["offsets"] = [tuple(x) for x in item["offsets"]]
            return data
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print(f"Processing data from {file_path}")
    df_raw = pd.read_csv(file_path)

    # Clean data
    if "selected_text" in df_raw.columns:
        df_raw = df_raw.dropna(subset=["text", "selected_text", "sentiment"])
    else:
        df_raw = df_raw.dropna(subset=["text", "sentiment"])

    # Handle Debug Mode
    if Config.DEBUG:
        df_raw = df_raw.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG Mode: Sampled {len(df_raw)} rows.")

    data = process_data(df_raw, tokenizer, max_len)

    # 3. Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    try:
        df_cache = pd.DataFrame(data)
        df_cache.to_parquet(cache_file)
        print(f"Saved cached data to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")

    return data


def create_loader(data, tokenizer, batch_size, shuffle=False, is_train=True):
    """
    Creates a DataLoader with Smart Batching.
    """
    ds = TweetDataset(data)
    collate = SmartBatchingCollate(tokenizer)

    if is_train:
        # Use SmartBatchingSampler for training
        # Note: batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        sampler = SmartBatchingSampler(data, batch_size)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        # For validation/test, standard loader is fine (sorted by length helps inference speed too)
        # But we keep it simple here.
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

    return loader
