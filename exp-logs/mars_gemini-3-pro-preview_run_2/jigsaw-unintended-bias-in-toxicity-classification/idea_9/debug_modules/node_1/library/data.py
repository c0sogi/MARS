import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from library.config import Config, load_and_preprocess_data, ToxDataset

# Alias the dataset class to satisfy the requirement that this module provides it,
# while reusing the existing implementation from the library.
ToxicityDataset = ToxDataset


def get_tokenizer(config):
    """
    Loads and returns the tokenizer based on the configuration.

    Args:
        config: Configuration object containing MODEL_NAME.

    Returns:
        tokenizer: Transformers tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return tokenizer


def prepare_data(config, load_cached_data=True):
    """
    Prepares the data: loads metadata, tokenizes, calculates weights, and caches results.

    This function leverages the existing logic in library.config which handles:
    1. Checking for cached .npy files in config.WORKING_DIR.
    2. If not found or load_cached_data=False:
       - Loads metadata from ./metadata
       - Merges with text from ./input
       - Tokenizes using RoBERTa tokenizer
       - Calculates sample weights (5.0 for identity mentions, 1.0 otherwise)
       - Saves processed arrays to cache.

    Args:
        config: Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        data (dict): Dictionary containing numpy arrays for ids, masks, targets, etc.
    """
    # Ensure the working directory exists as per requirements
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Delegate to the library function for consistent processing and caching
    data = load_and_preprocess_data(config, load_cached_data=load_cached_data)

    return data


def get_dataloaders(config, data):
    """
    Creates and returns DataLoaders for train, validation, and test sets.

    Args:
        config: Configuration object containing BATCH_SIZE.
        data (dict): Dictionary containing processed data arrays.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Train Dataset: Includes input_ids, masks, targets, sample weights, and aux targets (identities)
    train_dataset = ToxicityDataset(
        input_ids=data["train_ids"],
        attention_masks=data["train_masks"],
        targets=data["train_targets"],
        weights=data["train_weights"],
        aux_targets=data["train_aux"],
    )

    # Validation Dataset: Includes input_ids and masks.
    # Targets/Identities for metrics are handled separately in the evaluation loop.
    val_dataset = ToxicityDataset(
        input_ids=data["val_ids"], attention_masks=data["val_masks"]
    )

    # Test Dataset: Includes input_ids and masks.
    test_dataset = ToxicityDataset(
        input_ids=data["test_ids"], attention_masks=data["test_masks"]
    )

    # Create DataLoaders
    # Using 4 workers and pinned memory for efficiency as per the library configuration
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Double batch size for inference
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE * 2,  # Double batch size for inference
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
