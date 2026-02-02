import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Holds tokenized text, targets, sample weights, and auxiliary labels.
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        targets=None,
        weights=None,
        aux_identities=None,
        aux_identity_attack=None,
        is_test=False,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        self.weights = weights
        self.aux_identities = aux_identities
        self.aux_identity_attack = aux_identity_attack
        self.is_test = is_test

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if not self.is_test:
            # Primary target
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
            # Sample weight for loss scaling
            item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float)
            # Aux Head 1: Multi-label identities
            item["aux_identities"] = torch.tensor(
                self.aux_identities[idx], dtype=torch.float
            )
            # Aux Head 2: Identity attack subtype
            item["aux_identity_attack"] = torch.tensor(
                self.aux_identity_attack[idx], dtype=torch.float
            )

        return item


def get_tokenizer():
    """
    Returns the pretrained tokenizer defined in Config.
    """
    return AutoTokenizer.from_pretrained(Config.MODEL_NAME)


def calculate_sample_weights(df):
    """
    Calculates sample weights to prioritize 'Bias Trap' examples.
    Strategy:
    - Weight 5.0 (Config.WEIGHT_BIAS_TRAP) for examples with Identity Mentions.
    - Weight 1.0 (Config.WEIGHT_NORMAL) for others.
    """
    weights = np.ones(len(df), dtype=np.float32) * Config.WEIGHT_NORMAL

    # Identify rows with any identity mention (value >= 0.5)
    # Using the identity columns defined in Config
    identity_cols = Config.IDENTITY_COLUMNS

    # Check if any identity column is >= 0.5 (handling NaNs as 0.0)
    identity_mask = (df[identity_cols].fillna(0.0) >= 0.5).any(axis=1)

    # Apply higher weight to these examples to ensure the model focuses on
    # distinguishing toxic vs non-toxic within identity groups (Bias Traps).
    weights[identity_mask] = Config.WEIGHT_BIAS_TRAP

    return weights


def process_data(mode="train", load_cached_data=True, debug=False):
    """
    Loads raw data, performs tokenization, calculates weights/targets, and caches the result.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): If True, attempts to load from .npz cache.
        debug (bool): If True, uses a small subset of data.

    Returns:
        ToxicityDataset: The processed dataset ready for the DataLoader.
    """
    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Define cache filename
    prefix = "debug_" if debug else ""
    cache_file = os.path.join(Config.CACHE_DIR, f"{prefix}{mode}_processed.npz")

    # 1. Try Loading Cache
    if load_cached_data and os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            if mode == "test":
                return ToxicityDataset(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                    is_test=True,
                )
            else:
                return ToxicityDataset(
                    input_ids=data["input_ids"],
                    attention_mask=data["attention_mask"],
                    targets=data["targets"],
                    weights=data["weights"],
                    aux_identities=data["aux_identities"],
                    aux_identity_attack=data["aux_identity_attack"],
                    is_test=False,
                )
        except Exception as e:
            # If load fails, fall back to processing
            pass

    # 2. Load Source Data
    if mode == "train":
        df = pd.read_csv(Config.TRAIN_PATH)
    elif mode == "val":
        df = pd.read_csv(Config.VAL_PATH)
    elif mode == "test":
        df = pd.read_csv(Config.TEST_PATH)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Debug subset
    if debug:
        df = df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy().reset_index(drop=True)

    # Handle missing text
    df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")

    # 3. Tokenization
    tokenizer = get_tokenizer()
    encoded = tokenizer(
        df[Config.TEXT_COL].tolist(),
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 4. Process Labels (Train/Val only)
    if mode == "test":
        # Save cache for test
        np.savez(cache_file, input_ids=input_ids, attention_mask=attention_mask)
        return ToxicityDataset(input_ids, attention_mask, is_test=True)

    # Targets
    targets = df[Config.TARGET_COL].values.astype(np.float32)

    # Sample Weights
    weights = calculate_sample_weights(df)

    # Aux Identities (Multi-label)
    aux_identities = df[Config.IDENTITY_COLUMNS].fillna(0.0).values.astype(np.float32)

    # Aux Identity Attack (Single label)
    if Config.IDENTITY_ATTACK_COL in df.columns:
        aux_identity_attack = (
            df[Config.IDENTITY_ATTACK_COL].fillna(0.0).values.astype(np.float32)
        )
    else:
        aux_identity_attack = np.zeros(len(df), dtype=np.float32)

    # 5. Save Cache
    np.savez(
        cache_file,
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=targets,
        weights=weights,
        aux_identities=aux_identities,
        aux_identity_attack=aux_identity_attack,
    )

    return ToxicityDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        targets=targets,
        weights=weights,
        aux_identities=aux_identities,
        aux_identity_attack=aux_identity_attack,
        is_test=False,
    )


def make_loader(dataset, batch_size, mode="train"):
    """
    Creates a DataLoader for the dataset.

    Args:
        dataset (ToxicityDataset): The dataset to load.
        batch_size (int): Batch size.
        mode (str): 'train' or 'val'/'test'.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    if mode == "train":
        # Use WeightedRandomSampler for training to ensure batches contain
        # a dense mixture of bias trap examples (high weight).
        # This is crucial for the pairwise ranking loss to find valid pairs in a batch.
        weights = dataset.weights
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,  # Drop last incomplete batch to maintain batch size consistency
        )
    else:
        # Sequential sampling for validation/test
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

    return loader
