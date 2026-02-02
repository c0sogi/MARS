import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import normalize_text


class TweetDataset(Dataset):
    def __init__(self, input_ids, attention_mask, start_targets, end_targets, offsets):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.start_targets = start_targets
        self.end_targets = end_targets
        self.offsets = offsets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.long),
            "start_targets": torch.tensor(self.start_targets[item], dtype=torch.float),
            "end_targets": torch.tensor(self.end_targets[item], dtype=torch.float),
            "offsets": torch.tensor(self.offsets[item], dtype=torch.long),
        }


def get_gaussian_target(target_idx, length, sigma=1.0):
    """
    Generates a Gaussian distribution centered at target_idx.
    """
    if target_idx == -1:
        return np.zeros(length)
    x = np.arange(length)
    return np.exp(-0.5 * ((x - target_idx) / sigma) ** 2)


def process_data(df, tokenizer, max_len, cache_prefix, load_cached_data=True):
    """
    Tokenizes data, generates targets, and caches the results to disk.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    path_input_ids = os.path.join(cache_dir, f"{cache_prefix}_input_ids.npy")
    path_attention_mask = os.path.join(cache_dir, f"{cache_prefix}_attention_mask.npy")
    path_start_targets = os.path.join(cache_dir, f"{cache_prefix}_start_targets.npy")
    path_end_targets = os.path.join(cache_dir, f"{cache_prefix}_end_targets.npy")
    path_offsets = os.path.join(cache_dir, f"{cache_prefix}_offsets.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(path_input_ids)
            and os.path.exists(path_attention_mask)
            and os.path.exists(path_start_targets)
            and os.path.exists(path_end_targets)
            and os.path.exists(path_offsets)
        ):

            print(
                f"Loading cached data from {cache_dir} with prefix '{cache_prefix}'..."
            )
            input_ids = np.load(path_input_ids)
            attention_mask = np.load(path_attention_mask)
            start_targets = np.load(path_start_targets)
            end_targets = np.load(path_end_targets)
            offsets = np.load(path_offsets)
            return input_ids, attention_mask, start_targets, end_targets, offsets

    print(f"Processing data for '{cache_prefix}'...")

    n_samples = len(df)
    input_ids = np.zeros((n_samples, max_len), dtype=int)
    attention_mask = np.zeros((n_samples, max_len), dtype=int)
    start_targets = np.zeros((n_samples, max_len), dtype=float)
    end_targets = np.zeros((n_samples, max_len), dtype=float)
    offsets_arr = np.zeros((n_samples, max_len, 2), dtype=int)

    for idx, (_, row) in enumerate(df.iterrows()):
        # Apply Normalize-First protocol
        text = normalize_text(str(row["text"]))
        sentiment = str(row["sentiment"])

        # Determine character-level targets
        start_char_idx = -1
        end_char_idx = -1

        if "selected_text" in row and pd.notna(row["selected_text"]):
            selected_text = normalize_text(str(row["selected_text"]))
            # Find the substring in the normalized text
            start_char_idx = text.find(selected_text)
            if start_char_idx != -1:
                end_char_idx = start_char_idx + len(selected_text)
            else:
                # Fallback: If exact match fails (rare), assume full text or skip
                # Given the robust normalization, this should be minimal.
                # We default to full text coverage for safety in training.
                start_char_idx = 0
                end_char_idx = len(text)

        # Tokenize: [CLS] Sentiment [SEP] Text [SEP]
        # Note: DeBERTa tokenizer handles special tokens automatically.
        encoded = tokenizer.encode_plus(
            sentiment,
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=True,
        )

        ids = encoded["input_ids"]
        mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]
        sequence_ids = encoded.sequence_ids()

        input_ids[idx] = ids
        attention_mask[idx] = mask
        offsets_arr[idx] = offset_mapping

        # Determine token-level targets
        token_start_idx = -1
        token_end_idx = -1

        if start_char_idx != -1:
            # Identify tokens belonging to the 'text' segment (sequence_id == 1)
            text_token_indices = [
                i for i, seq_id in enumerate(sequence_ids) if seq_id == 1
            ]

            if text_token_indices:
                # Find Start Token: Token containing the start_char_idx
                for t_idx in text_token_indices:
                    off_start, off_end = offset_mapping[t_idx]
                    if off_start <= start_char_idx < off_end:
                        token_start_idx = t_idx
                        break

                # Find End Token: Token containing the last character (end_char_idx - 1)
                target_end_char = max(0, end_char_idx - 1)
                for t_idx in text_token_indices:
                    off_start, off_end = offset_mapping[t_idx]
                    if off_start <= target_end_char < off_end:
                        token_end_idx = t_idx
                        break

                # Boundary snapping fallback
                if token_start_idx == -1:
                    token_start_idx = text_token_indices[0]
                if token_end_idx == -1:
                    token_end_idx = text_token_indices[-1]

        # Generate Gaussian Soft Targets
        if token_start_idx != -1 and token_end_idx != -1:
            start_targets[idx] = get_gaussian_target(
                token_start_idx, max_len, Config.SMOOTHING_SIGMA
            )
            end_targets[idx] = get_gaussian_target(
                token_end_idx, max_len, Config.SMOOTHING_SIGMA
            )

    # Save to cache
    np.save(path_input_ids, input_ids)
    np.save(path_attention_mask, attention_mask)
    np.save(path_start_targets, start_targets)
    np.save(path_end_targets, end_targets)
    np.save(path_offsets, offsets_arr)

    return input_ids, attention_mask, start_targets, end_targets, offsets_arr


def get_data_loaders(fold=0, load_cached_data=True, debug=False):
    """
    Creates train and validation DataLoaders.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Process or Load Data
    prefix = "train_debug" if debug else "train"
    input_ids, attention_mask, start_targets, end_targets, offsets = process_data(
        train_df, tokenizer, Config.MAX_LEN, prefix, load_cached_data=load_cached_data
    )

    # Stratified K-Fold Split
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )
    splits = list(skf.split(train_df, train_df["sentiment"]))
    train_idx, val_idx = splits[fold]

    # Filter Neutrals
    if Config.FILTER_NEUTRAL:
        # Get boolean mask for non-neutral rows
        is_not_neutral = train_df["sentiment"] != "neutral"

        # Filter training indices
        train_subset_mask = is_not_neutral.iloc[train_idx].values
        train_idx = train_idx[train_subset_mask]

        # Filter validation indices (to align loss calculation with training objective)
        val_subset_mask = is_not_neutral.iloc[val_idx].values
        val_idx = val_idx[val_subset_mask]

    # Create Datasets
    train_dataset = TweetDataset(
        input_ids[train_idx],
        attention_mask[train_idx],
        start_targets[train_idx],
        end_targets[train_idx],
        offsets[train_idx],
    )

    val_dataset = TweetDataset(
        input_ids[val_idx],
        attention_mask[val_idx],
        start_targets[val_idx],
        end_targets[val_idx],
        offsets[val_idx],
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
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
        drop_last=False,
    )

    return train_loader, val_loader


def get_eval_loader(df, prefix="eval", load_cached_data=True):
    """
    Creates a DataLoader for evaluation (validation or test) from a DataFrame.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    input_ids, attention_mask, start_targets, end_targets, offsets = process_data(
        df, tokenizer, Config.MAX_LEN, prefix, load_cached_data=load_cached_data
    )

    dataset = TweetDataset(
        input_ids, attention_mask, start_targets, end_targets, offsets
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def get_test_loader(load_cached_data=True, debug=False):
    """
    Legacy wrapper for compatibility.
    """
    test_df = pd.read_csv(Config.TEST_META)
    if debug:
        test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

    prefix = "test_debug" if debug else "test"
    loader = get_eval_loader(test_df, prefix=prefix, load_cached_data=load_cached_data)
    return loader, test_df
