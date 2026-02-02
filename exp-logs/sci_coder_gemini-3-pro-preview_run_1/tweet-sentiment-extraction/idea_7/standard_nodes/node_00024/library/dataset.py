import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.utils import normalize_text


def process_data(df, tokenizer, config, is_test=False):
    """
    Processes the dataframe into numpy arrays suitable for training/inference.
    Applies 'Normalize-First' strategy to ensure alignment between text and offsets.
    """
    n_samples = len(df)
    max_len = config.MAX_LEN

    # Pre-allocate arrays for efficiency
    input_ids = np.zeros((n_samples, max_len), dtype=np.int32)
    attention_mask = np.zeros((n_samples, max_len), dtype=np.int32)
    offsets = np.zeros((n_samples, max_len, 2), dtype=np.int32)

    start_indices = np.zeros(n_samples, dtype=np.int32)
    end_indices = np.zeros(n_samples, dtype=np.int32)

    for idx, row in df.iterrows():
        # Normalize-First: Collapse whitespace before tokenization
        text = normalize_text(row["text"])

        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        input_ids[idx] = encoded["input_ids"]
        attention_mask[idx] = encoded["attention_mask"]
        offsets[idx] = encoded["offset_mapping"]

        if not is_test:
            # Normalize target text to match input text
            selected_text = normalize_text(row["selected_text"])

            # Find exact character match in normalized text
            start_char = text.find(selected_text)

            if start_char == -1:
                # Fallback: use full text if not found (rare with normalization)
                start_char = 0
                end_char = len(text)
            else:
                end_char = start_char + len(selected_text)

            tokens_offsets = encoded["offset_mapping"]
            start_token = 0
            end_token = 0
            found_start = False

            # Map character indices to token indices
            for i, (o_start, o_end) in enumerate(tokens_offsets):
                if o_start == 0 and o_end == 0:
                    continue
                if o_start <= start_char < o_end:
                    start_token = i
                    found_start = True
                    break

            if found_start:
                for i, (o_start, o_end) in enumerate(tokens_offsets):
                    if o_start == 0 and o_end == 0:
                        continue
                    if o_start < end_char <= o_end:
                        end_token = i
                        break

            # Handle edge cases where end_token wasn't found (e.g., truncation)
            if end_token == 0:
                # Fallback to last valid token (excluding [SEP])
                end_token = np.sum(encoded["attention_mask"]) - 2

            start_indices[idx] = start_token
            end_indices[idx] = end_token

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offsets": offsets,
    }

    if not is_test:
        data["start_indices"] = start_indices
        data["end_indices"] = end_indices

    return data


def get_data(df, tokenizer, config, cache_name="train", load_cached_data=True):
    """
    Retrieves data from cache or processes it if cache is missing.
    Implements file-based caching using numpy format.
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Define file paths for cache
    files = {
        "input_ids": os.path.join(config.CACHE_DIR, f"{cache_name}_input_ids.npy"),
        "attention_mask": os.path.join(
            config.CACHE_DIR, f"{cache_name}_attention_mask.npy"
        ),
        "offsets": os.path.join(config.CACHE_DIR, f"{cache_name}_offsets.npy"),
    }

    # Determine if we need to handle targets based on cache name or explicit flag
    # Convention: 'test' in name implies inference mode (no targets)
    is_test = "test" in cache_name
    if not is_test:
        files["start_indices"] = os.path.join(
            config.CACHE_DIR, f"{cache_name}_start_indices.npy"
        )
        files["end_indices"] = os.path.join(
            config.CACHE_DIR, f"{cache_name}_end_indices.npy"
        )

    # Attempt to load from cache
    if load_cached_data:
        all_exist = all(os.path.exists(f) for f in files.values())
        if all_exist:
            print(f"Loading cached data for {cache_name}...")
            return {k: np.load(v) for k, v in files.items()}

    # Process data from scratch
    print(f"Processing data for {cache_name}...")
    data = process_data(df, tokenizer, config, is_test=is_test)

    # Save processed data to cache
    for k, v in data.items():
        if k in files:
            np.save(files[k], v)

    return data


class TweetDataset(Dataset):
    def __init__(self, data, config, is_test=False):
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.offsets = data["offsets"]
        self.config = config
        self.is_test = is_test

        if not is_test:
            self.start_indices = data["start_indices"]
            self.end_indices = data["end_indices"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        out = {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }

        if not self.is_test:
            start_idx = self.start_indices[item]
            end_idx = self.end_indices[item]
            seq_len = self.config.MAX_LEN

            # Generate Gaussian-smoothed soft targets
            x = torch.arange(seq_len, dtype=torch.float)
            sigma = self.config.SIGMA

            start_target = torch.exp(-0.5 * ((x - start_idx) / sigma) ** 2)
            end_target = torch.exp(-0.5 * ((x - end_idx) / sigma) ** 2)

            # Normalize to create a valid probability distribution
            start_target = start_target / (start_target.sum() + 1e-6)
            end_target = end_target / (end_target.sum() + 1e-6)

            out["start_targets"] = start_target
            out["end_targets"] = end_target

        return out
