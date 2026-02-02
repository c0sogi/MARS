import os
import numpy as np
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.data_factory import process_and_cache_data


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Handles input IDs, attention masks, toxicity targets, auxiliary identity targets,
    sample weights, and IDs. Designed to work with memory-mapped numpy arrays for efficiency.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        targets=None,
        aux_targets=None,
        weights=None,
        ids=None,
    ):
        """
        Args:
            input_ids (np.ndarray): Array of token indices (N, seq_len).
            attention_masks (np.ndarray): Array of attention masks (N, seq_len).
            targets (np.ndarray, optional): Array of toxicity targets (N,).
            aux_targets (np.ndarray, optional): Array of auxiliary identity targets (N, num_aux).
            weights (np.ndarray, optional): Array of sample weights (N,).
            ids (np.ndarray, optional): Array of example IDs (N,).
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.aux_targets = aux_targets
        self.weights = weights
        self.ids = ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Convert numpy types to PyTorch tensors
        # input_ids and attention_mask are expected to be integers
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        # Add targets if available (Train/Val)
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        # Add auxiliary targets if available (Train/Val)
        if self.aux_targets is not None:
            item["aux_targets"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)

        # Add sample weights if available (Train)
        if self.weights is not None:
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)

        # Add IDs if available (Val/Test)
        if self.ids is not None:
            item["id"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item


def load_datasets(load_cached_data=True):
    """
    Loads training, validation, and test datasets.
    Ensures data is processed and cached, then loads it using memory-mapping.

    Args:
        load_cached_data (bool): If True, attempts to use existing cache.
                                 If False, forces reprocessing.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Check if critical cache files exist
    cache_exists = os.path.exists(Config.CACHE_TRAIN_INPUT_IDS) and os.path.exists(
        Config.CACHE_TEST_INPUT_IDS
    )

    # If cache is missing or reload requested, process data
    if not cache_exists or not load_cached_data:
        print("Cache missing or reload requested. Invoking data factory...")
        # We ignore the return value here because we want to load via mmap below
        process_and_cache_data(load_cached_data=False)
    else:
        print("Cache found. Loading memory-mapped arrays...")

    # Load arrays in read-only memory-map mode
    # This is efficient for large datasets as it doesn't load everything into RAM at once
    train_input_ids = np.load(Config.CACHE_TRAIN_INPUT_IDS, mmap_mode="r")
    train_attn_masks = np.load(Config.CACHE_TRAIN_ATTN_MASKS, mmap_mode="r")
    train_targets = np.load(Config.CACHE_TRAIN_TARGETS, mmap_mode="r")
    train_aux_targets = np.load(Config.CACHE_TRAIN_AUX_TARGETS, mmap_mode="r")
    train_weights = np.load(Config.CACHE_TRAIN_SAMPLE_WEIGHTS, mmap_mode="r")

    val_input_ids = np.load(Config.CACHE_VAL_INPUT_IDS, mmap_mode="r")
    val_attn_masks = np.load(Config.CACHE_VAL_ATTN_MASKS, mmap_mode="r")
    val_targets = np.load(Config.CACHE_VAL_TARGETS, mmap_mode="r")
    val_aux_targets = np.load(Config.CACHE_VAL_AUX_TARGETS, mmap_mode="r")
    val_ids = np.load(Config.CACHE_VAL_IDS, mmap_mode="r")

    test_input_ids = np.load(Config.CACHE_TEST_INPUT_IDS, mmap_mode="r")
    test_attn_masks = np.load(Config.CACHE_TEST_ATTN_MASKS, mmap_mode="r")
    test_ids = np.load(Config.CACHE_TEST_IDS, mmap_mode="r")

    # Instantiate Datasets
    train_dataset = ToxicityDataset(
        input_ids=train_input_ids,
        attention_masks=train_attn_masks,
        targets=train_targets,
        aux_targets=train_aux_targets,
        weights=train_weights,
    )

    val_dataset = ToxicityDataset(
        input_ids=val_input_ids,
        attention_masks=val_attn_masks,
        targets=val_targets,
        aux_targets=val_aux_targets,
        ids=val_ids,
    )

    test_dataset = ToxicityDataset(
        input_ids=test_input_ids,
        attention_masks=test_attn_masks,
        ids=test_ids,
    )

    return train_dataset, val_dataset, test_dataset
