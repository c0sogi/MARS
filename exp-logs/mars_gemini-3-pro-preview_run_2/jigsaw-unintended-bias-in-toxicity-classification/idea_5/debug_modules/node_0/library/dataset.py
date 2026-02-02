import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Serves pre-tokenized input_ids and attention_masks, along with multi-task targets.
    """

    def __init__(self, input_ids, attention_mask, targets=None, ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        self.ids = ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.targets is not None:
            # Targets include [toxicity_score, identity_1, identity_2, ...]
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.ids is not None:
            item["ids"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item


def process_data(
    metadata_path,
    text_source_path,
    cache_prefix,
    load_cached_data=True,
    is_test=False,
    debug=False,
):
    """
    Loads metadata, merges with text, tokenizes, and caches the result.

    Args:
        metadata_path (str): Path to the metadata CSV.
        text_source_path (str): Path to the source CSV containing text.
        cache_prefix (str): Prefix for cached filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.
        is_test (bool): Whether this is the test set (no targets).
        debug (bool): If True, processes a small subset.

    Returns:
        tuple: (input_ids, attention_mask, targets, ids)
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    # We append 'debug' to filename if debugging to avoid polluting real cache
    suffix = "_debug" if debug else ""
    path_input_ids = os.path.join(cache_dir, f"{cache_prefix}_input_ids{suffix}.npy")
    path_masks = os.path.join(cache_dir, f"{cache_prefix}_masks{suffix}.npy")
    path_targets = os.path.join(cache_dir, f"{cache_prefix}_targets{suffix}.npy")
    path_ids = os.path.join(cache_dir, f"{cache_prefix}_ids{suffix}.npy")

    # 1. Try Loading from Cache
    if load_cached_data:
        # Check if all required files exist
        files_exist = (
            os.path.exists(path_input_ids)
            and os.path.exists(path_masks)
            and os.path.exists(path_ids)
        )
        if not is_test:
            files_exist = files_exist and os.path.exists(path_targets)

        if files_exist:
            print(f"Loading {cache_prefix} data from cache...")
            input_ids = np.load(path_input_ids)
            attention_mask = np.load(path_masks)
            ids = np.load(path_ids)
            targets = np.load(path_targets) if not is_test else None
            return input_ids, attention_mask, targets, ids

    print(f"Processing {cache_prefix} data from scratch...")

    # 2. Load Metadata
    df_meta = pd.read_csv(metadata_path)
    if debug:
        df_meta = df_meta.iloc[:1000].copy()

    # 3. Load Text Source
    # We read only necessary columns to save memory
    # Note: text_source_path usually contains the full dataset (train.csv or test.csv)
    df_text = pd.read_csv(text_source_path, usecols=["id", "comment_text"])

    # 4. Merge
    # Left join ensures we strictly follow the metadata split
    df = df_meta.merge(df_text, on="id", how="left")

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("missing")

    # 5. Tokenize
    print(f"Tokenizing {len(df)} samples...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Use batch_encode_plus for speed
    # Convert series to list for tokenizer
    texts = df["comment_text"].astype(str).tolist()

    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    ids = df["id"].values

    # 6. Process Targets (if not test)
    targets = None
    if not is_test:
        # Primary Target: Toxicity
        # Shape: (N, 1)
        target_main = df[Config.TARGET_COL].values.reshape(-1, 1)

        # Auxiliary Targets: Identities
        # Fill NaNs with 0.0 (assuming NaN means identity not mentioned)
        # Shape: (N, num_identities)
        target_aux = df[Config.IDENTITY_COLUMNS].fillna(0.0).values

        # Combine: [Main, Aux1, Aux2, ...]
        targets = np.hstack([target_main, target_aux])

        # Save targets
        np.save(path_targets, targets)

    # 7. Save Cache
    np.save(path_input_ids, input_ids)
    np.save(path_masks, attention_mask)
    np.save(path_ids, ids)

    return input_ids, attention_mask, targets, ids


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for Train, Validation, and Test sets.

    Args:
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    debug = Config.DEBUG

    # --- Train Data ---
    train_ids, train_masks, train_targets, train_ids_arr = process_data(
        metadata_path=Config.TRAIN_METADATA_PATH,
        text_source_path=Config.TRAIN_TEXT_SOURCE,
        cache_prefix="train",
        load_cached_data=load_cached_data,
        is_test=False,
        debug=debug,
    )

    train_dataset = ToxicityDataset(
        train_ids, train_masks, train_targets, train_ids_arr
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Validation Data ---
    val_ids, val_masks, val_targets, val_ids_arr = process_data(
        metadata_path=Config.VALID_METADATA_PATH,
        text_source_path=Config.TRAIN_TEXT_SOURCE,  # Val comes from train.csv
        cache_prefix="val",
        load_cached_data=load_cached_data,
        is_test=False,
        debug=debug,
    )

    val_dataset = ToxicityDataset(val_ids, val_masks, val_targets, val_ids_arr)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Data ---
    test_ids, test_masks, _, test_ids_arr = process_data(
        metadata_path=Config.TEST_METADATA_PATH,
        text_source_path=Config.TEST_TEXT_SOURCE,
        cache_prefix="test",
        load_cached_data=load_cached_data,
        is_test=True,
        debug=debug,
    )

    test_dataset = ToxicityDataset(test_ids, test_masks, targets=None, ids=test_ids_arr)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
