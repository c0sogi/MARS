import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from library.config import (
    MODEL_NAME,
    MAX_LEN,
    WORKING_DIR,
    IDENTITY_COLUMNS,
    TARGET_COL,
)
from library.utils import load_processed_data


def tokenize_and_save(split, load_cached_data=True):
    """
    Tokenizes the text data for a given split and saves the resulting arrays to disk.

    Args:
        split (str): 'train', 'validation', or 'test'.
        load_cached_data (bool): Whether to use existing files if available.
    """
    # Define file paths
    input_ids_path = os.path.join(WORKING_DIR, f"{split}_input_ids.npy")
    masks_path = os.path.join(WORKING_DIR, f"{split}_masks.npy")
    ids_path = os.path.join(WORKING_DIR, f"{split}_ids.npy")
    targets_path = os.path.join(WORKING_DIR, f"{split}_targets.npy")
    identities_path = os.path.join(WORKING_DIR, f"{split}_identities.npy")

    # Check if all required files exist
    required_files = [input_ids_path, masks_path, ids_path]
    if split != "test":
        required_files.extend([targets_path, identities_path])

    files_exist = all(os.path.exists(f) for f in required_files)

    if load_cached_data and files_exist:
        return

    # Load data
    df = load_processed_data(split, load_cached_data=load_cached_data)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Prepare lists for collecting results
    all_input_ids = []
    all_masks = []

    # Tokenize in chunks to be memory efficient during processing
    text_data = df["comment_text"].astype(str).tolist()
    chunk_size = 20000

    for i in range(0, len(text_data), chunk_size):
        chunk = text_data[i : i + chunk_size]
        encoded = tokenizer.batch_encode_plus(
            chunk,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        all_input_ids.append(encoded["input_ids"].astype(np.int32))
        all_masks.append(encoded["attention_mask"].astype(np.int8))

    # Concatenate
    input_ids_np = np.concatenate(all_input_ids, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    ids_np = df["id"].values.astype(np.int64)

    # Save inputs
    np.save(input_ids_path, input_ids_np)
    np.save(masks_path, masks_np)
    np.save(ids_path, ids_np)

    # Save targets and identities for train/val
    if split != "test":
        targets_np = df[TARGET_COL].values.astype(np.float32)
        # Fill NaNs in identity columns with 0.0 (assuming not mentioned)
        identities_np = df[IDENTITY_COLUMNS].fillna(0.0).values.astype(np.float32)

        np.save(targets_path, targets_np)
        np.save(identities_path, identities_np)


class ToxicityDataset(Dataset):
    def __init__(self, split, load_cached_data=True, debug_size=None):
        """
        Dataset class for Toxicity Classification.

        Args:
            split (str): 'train', 'validation', or 'test'.
            load_cached_data (bool): Whether to load from cache.
            debug_size (int, optional): Limit dataset size for debugging.
        """
        self.split = split

        # Ensure data is processed and cached
        tokenize_and_save(split, load_cached_data)

        # Load arrays into memory
        self.input_ids = np.load(os.path.join(WORKING_DIR, f"{split}_input_ids.npy"))
        self.masks = np.load(os.path.join(WORKING_DIR, f"{split}_masks.npy"))
        self.ids = np.load(os.path.join(WORKING_DIR, f"{split}_ids.npy"))

        if split != "test":
            self.targets = np.load(os.path.join(WORKING_DIR, f"{split}_targets.npy"))
            self.identities = np.load(
                os.path.join(WORKING_DIR, f"{split}_identities.npy")
            )
        else:
            self.targets = None
            self.identities = None

        # Apply debug slicing
        if debug_size is not None:
            self.input_ids = self.input_ids[:debug_size]
            self.masks = self.masks[:debug_size]
            self.ids = self.ids[:debug_size]
            if self.targets is not None:
                self.targets = self.targets[:debug_size]
                self.identities = self.identities[:debug_size]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Convert to tensors
        input_id = torch.tensor(self.input_ids[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)

        if self.split != "test":
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            identity = torch.tensor(self.identities[idx], dtype=torch.float)
            return input_id, mask, target, identity
        else:
            # For test set, return ID to map predictions back
            id_val = torch.tensor(self.ids[idx], dtype=torch.long)
            return input_id, mask, id_val
