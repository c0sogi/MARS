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
    Handles input IDs, attention masks, toxicity targets, auxiliary identity targets,
    and sample weights.
    """

    def __init__(
        self,
        input_ids,
        attention_masks,
        targets=None,
        aux_targets=None,
        sample_weights=None,
        ids=None,
    ):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        self.aux_targets = aux_targets
        self.sample_weights = sample_weights
        self.ids = ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
        }

        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        if self.aux_targets is not None:
            item["aux_target"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)

        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(
                self.sample_weights[idx], dtype=torch.float
            )

        if self.ids is not None:
            item["id"] = torch.tensor(self.ids[idx], dtype=torch.long)

        return item


def _load_raw_data(metadata_path, text_path):
    """
    Loads metadata and merges it with the raw text content.
    """
    # Load metadata (contains labels and split info)
    meta_df = pd.read_csv(metadata_path)

    # Load text content (only ID and text to save memory)
    text_df = pd.read_csv(text_path, usecols=[Config.ID_COL, Config.TEXT_COL])

    # Merge on ID
    df = meta_df.merge(text_df, on=Config.ID_COL, how="left")
    df[Config.TEXT_COL] = df[Config.TEXT_COL].fillna("")

    # Debugging: Subsample if configured
    if Config.DEBUG:
        print(
            f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows from {metadata_path}"
        )
        df = df.sample(
            n=min(len(df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    return df


def _process_split(split_name, metadata_path, text_path, tokenizer, load_cached_data):
    """
    Handles loading, tokenization, and caching for a specific data split.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    files = {
        "input_ids": os.path.join(cache_dir, f"{split_name}_input_ids.npy"),
        "masks": os.path.join(cache_dir, f"{split_name}_masks.npy"),
        "ids": os.path.join(cache_dir, f"{split_name}_ids.npy"),
    }

    is_test = split_name == "test"
    if not is_test:
        files.update(
            {
                "targets": os.path.join(cache_dir, f"{split_name}_targets.npy"),
                "aux_targets": os.path.join(cache_dir, f"{split_name}_aux_targets.npy"),
                "weights": os.path.join(cache_dir, f"{split_name}_weights.npy"),
            }
        )

    # 1. Try Loading from Cache
    if load_cached_data:
        all_exist = all(os.path.exists(p) for p in files.values())
        if all_exist:
            print(f"Loading cached data for '{split_name}' from {cache_dir}...")
            data = {}
            for k, v in files.items():
                data[k] = np.load(v)
            return data

    # 2. Process from Scratch
    print(f"Processing data for '{split_name}' (Cache miss or force reload)...")
    df = _load_raw_data(metadata_path, text_path)

    print(f"Tokenizing {len(df)} samples...")
    encoded = tokenizer.batch_encode_plus(
        df[Config.TEXT_COL].tolist(),
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"]
    masks = encoded["attention_mask"]
    ids = df[Config.ID_COL].values

    # Save inputs
    np.save(files["input_ids"], input_ids)
    np.save(files["masks"], masks)
    np.save(files["ids"], ids)

    result = {"input_ids": input_ids, "masks": masks, "ids": ids}

    if not is_test:
        # Process Targets
        targets = df[Config.TARGET_COL].values

        # Process Auxiliary Targets (Identity Columns)
        aux_targets_list = []
        for col in Config.IDENTITY_COLUMNS:
            if col in df.columns:
                aux_targets_list.append(df[col].values)
            else:
                # Fallback if column missing (should not happen with correct metadata)
                aux_targets_list.append(np.zeros(len(df)))

        aux_targets = np.stack(aux_targets_list, axis=1)

        # Calculate Sample Weights
        # Strategy: Assign higher weight if any identity is mentioned (>= 0.5)
        has_identity = (aux_targets >= 0.5).any(axis=1)
        weights = np.ones(len(df), dtype=np.float32)
        weights[has_identity] = Config.IDENTITY_SAMPLE_WEIGHT

        # Save targets and weights
        np.save(files["targets"], targets)
        np.save(files["aux_targets"], aux_targets)
        np.save(files["weights"], weights)

        result["targets"] = targets
        result["aux_targets"] = aux_targets
        result["weights"] = weights

    return result


def get_dataloaders(load_cached_data=True):
    """
    Prepares and returns DataLoaders for Training and Validation sets.
    """
    print("Initializing Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # --- Training Data ---
    train_data = _process_split(
        "train",
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_TEXT_FILE,
        tokenizer,
        load_cached_data,
    )

    train_dataset = ToxicityDataset(
        train_data["input_ids"],
        train_data["masks"],
        train_data["targets"],
        train_data["aux_targets"],
        train_data["weights"],
        train_data["ids"],
    )

    # --- Validation Data ---
    # Note: Validation metadata points to 'train.csv' for text content
    val_data = _process_split(
        "val",
        Config.VAL_METADATA_PATH,
        Config.TRAIN_TEXT_FILE,
        tokenizer,
        load_cached_data,
    )

    val_dataset = ToxicityDataset(
        val_data["input_ids"],
        val_data["masks"],
        val_data["targets"],
        val_data["aux_targets"],
        val_data["weights"],
        val_data["ids"],
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
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(load_cached_data=True):
    """
    Prepares and returns DataLoader for the Test set.
    """
    print("Initializing Tokenizer for Test Set...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    test_data = _process_split(
        "test",
        Config.TEST_METADATA_PATH,
        Config.TEST_TEXT_FILE,
        tokenizer,
        load_cached_data,
    )

    test_dataset = ToxicityDataset(
        test_data["input_ids"],
        test_data["masks"],
        ids=test_data["ids"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
