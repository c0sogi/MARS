import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Returns input_ids, attention_mask, target (toxicity), and auxiliary identities.
    """

    def __init__(self, input_ids, attention_masks, targets=None, identities=None):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.identities = identities

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Convert numpy arrays to torch tensors
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.identities is not None:
            item["identities"] = torch.tensor(self.identities[idx], dtype=torch.float)

        return item


def _process_and_cache(split_name, metadata_path, text_path, load_cached_data):
    """
    Internal function to load, preprocess (tokenize), and cache data.
    """
    # Define cache file paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{split_name}_input_ids.npy")
    masks_path = os.path.join(cache_dir, f"{split_name}_masks.npy")
    sample_ids_path = os.path.join(cache_dir, f"{split_name}_ids.npy")
    targets_path = os.path.join(cache_dir, f"{split_name}_targets.npy")
    identities_path = os.path.join(cache_dir, f"{split_name}_identities.npy")

    # Determine if we need to load targets/identities (not for test)
    is_test = split_name == "test"

    # Check if cache exists
    cache_exists = (
        os.path.exists(ids_path)
        and os.path.exists(masks_path)
        and os.path.exists(sample_ids_path)
    )
    if not is_test:
        cache_exists = (
            cache_exists
            and os.path.exists(targets_path)
            and os.path.exists(identities_path)
        )

    # 1. Load from Cache
    if load_cached_data and cache_exists:
        print(f"Loading cached data for {split_name}...")
        input_ids = np.load(ids_path)
        attention_masks = np.load(masks_path)
        sample_ids = np.load(sample_ids_path)

        targets = None
        identities = None
        if not is_test:
            targets = np.load(targets_path)
            identities = np.load(identities_path)

        return input_ids, attention_masks, targets, identities, sample_ids

    # 2. Process from Scratch
    print(f"Processing data for {split_name} from scratch...")

    # Load Metadata
    print(f"  Loading metadata from {metadata_path}...")
    meta_df = pd.read_csv(metadata_path)

    # Load Raw Text (only necessary columns)
    print(f"  Loading text from {text_path}...")
    text_df = pd.read_csv(text_path, usecols=[Config.ID_COL, Config.TEXT_COL])

    # Merge Metadata with Text
    # Inner join ensures we only get the text for the IDs in this split
    df = meta_df.merge(text_df, on=Config.ID_COL, how="inner")

    # Handle missing text
    df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")

    # Debug Mode: Slice data
    if Config.DEBUG:
        print(f"  DEBUG mode: limiting to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # Tokenization
    print("  Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Process in chunks to manage memory
    chunk_size = 20000
    texts = df[Config.TEXT_COL].tolist()
    num_samples = len(texts)

    all_input_ids = []
    all_masks = []

    for i in range(0, num_samples, chunk_size):
        batch_texts = texts[i : i + chunk_size]
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="np",
        )
        all_input_ids.append(encoded["input_ids"])
        all_masks.append(encoded["attention_mask"])

    input_ids = np.concatenate(all_input_ids, axis=0).astype(np.int32)
    attention_masks = np.concatenate(all_masks, axis=0).astype(np.int32)
    sample_ids = df[Config.ID_COL].values

    # Save Inputs to Cache
    np.save(ids_path, input_ids)
    np.save(masks_path, attention_masks)
    np.save(sample_ids_path, sample_ids)

    targets = None
    identities = None

    if not is_test:
        # Process Targets
        targets = df[Config.TARGET_COL].values.astype(np.float32)

        # Process Identities
        # Fill NaNs with 0 (assuming NaN means identity not mentioned/annotated)
        ident_df = df[Config.IDENTITY_COLUMNS].fillna(0.0)
        identities = ident_df.values.astype(np.float32)

        # Save Labels to Cache
        np.save(targets_path, targets)
        np.save(identities_path, identities)

    print(f"  Processing complete. Saved to {cache_dir}")
    return input_ids, attention_masks, targets, identities, sample_ids


def get_dataloaders(load_cached_data=True):
    """
    Creates and returns DataLoaders for Train, Validation, and Test sets.
    Also returns the Test IDs for submission mapping.
    """
    # --------------------------------------------------------------------------
    # Train Set
    # --------------------------------------------------------------------------
    train_ids, train_masks, train_y, train_ident, _ = _process_and_cache(
        "train", Config.TRAIN_METADATA_PATH, Config.TRAIN_TEXT_PATH, load_cached_data
    )

    train_dataset = ToxicityDataset(train_ids, train_masks, train_y, train_ident)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,  # Drop last incomplete batch for stability
    )

    # --------------------------------------------------------------------------
    # Validation Set
    # --------------------------------------------------------------------------
    # Note: Validation text is also in train.csv
    val_ids, val_masks, val_y, val_ident, _ = _process_and_cache(
        "val", Config.VALID_METADATA_PATH, Config.TRAIN_TEXT_PATH, load_cached_data
    )

    val_dataset = ToxicityDataset(val_ids, val_masks, val_y, val_ident)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # --------------------------------------------------------------------------
    # Test Set
    # --------------------------------------------------------------------------
    test_ids, test_masks, _, _, test_sample_ids = _process_and_cache(
        "test", Config.TEST_METADATA_PATH, Config.TEST_TEXT_PATH, load_cached_data
    )

    test_dataset = ToxicityDataset(test_ids, test_masks)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, test_sample_ids
