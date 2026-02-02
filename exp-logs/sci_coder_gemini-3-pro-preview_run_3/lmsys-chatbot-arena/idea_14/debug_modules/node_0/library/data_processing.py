import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

from library.config import Config
from library.utils import set_seed


class SiameseDataset(Dataset):
    """
    PyTorch Dataset for Siamese DeBERTa model.
    Serves pre-processed tensors for Branch A and Branch B, along with scalar features.
    """

    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): Dictionary containing numpy arrays:
                              - input_ids: (N, 2, seq_len)
                              - attention_mask: (N, 2, seq_len)
                              - scalar_features: (N, 3)
                              - labels: (N, 3)
        """
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.scalar_features = data_dict["scalar_features"]
        self.labels = data_dict["labels"]
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "scalar_features": torch.tensor(
                self.scalar_features[idx], dtype=torch.float
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def process_dataframe(df, tokenizer, max_length):
    """
    Tokenizes data and computes scalar features.

    Args:
        df (pd.DataFrame): Dataframe containing prompt, response_a, response_b.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Maximum sequence length.

    Returns:
        dict: Dictionary of numpy arrays ready for the Dataset.
    """
    # 1. Scalar Features: Log-transformed lengths (char count)
    # Fill NaNs with empty string just in case
    prompts = df["prompt"].fillna("").astype(str).tolist()
    resps_a = df["response_a"].fillna("").astype(str).tolist()
    resps_b = df["response_b"].fillna("").astype(str).tolist()

    len_p = np.log(np.array([len(t) for t in prompts]) + 1)
    len_a = np.log(np.array([len(t) for t in resps_a]) + 1)
    len_b = np.log(np.array([len(t) for t in resps_b]) + 1)

    # Shape: (N, 3) -> [log_len_prompt, log_len_resp_a, log_len_resp_b]
    scalar_features = np.stack([len_p, len_a, len_b], axis=1).astype(np.float32)

    # 2. Tokenization
    # We use batch_encode_plus for speed.
    # Strategy: [CLS] Prompt [SEP] Response [SEP]
    # truncation="only_second" ensures Prompt is preserved.

    # Branch A
    enc_a = tokenizer(
        prompts,
        resps_a,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
        return_token_type_ids=False,
    )

    # Branch B
    enc_b = tokenizer(
        prompts,
        resps_b,
        truncation="only_second",
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
        return_token_type_ids=False,
    )

    # Stack branches: (N, 2, seq_len)
    input_ids = np.stack([enc_a["input_ids"], enc_b["input_ids"]], axis=1)
    attention_mask = np.stack(
        [enc_a["attention_mask"], enc_b["attention_mask"]], axis=1
    )

    # 3. Targets
    if "winner_model_a" in df.columns:
        labels = df[["winner_model_a", "winner_model_b", "winner_tie"]].values.astype(
            np.float32
        )
    else:
        # Dummy labels for test set
        labels = np.zeros((len(df), 3), dtype=np.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "scalar_features": scalar_features,
        "labels": labels,
    }


def load_and_process_data(config, tokenizer, split, load_cached_data=True):
    """
    Loads data, applies augmentation (if train), tokenizes, and caches results.

    Args:
        config (Config): Configuration object.
        tokenizer: Tokenizer instance.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Processed data dictionary.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"{split}_data.npz")

    # 1. Try Load Cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        try:
            loaded = np.load(cache_path)
            return {
                "input_ids": loaded["input_ids"],
                "attention_mask": loaded["attention_mask"],
                "scalar_features": loaded["scalar_features"],
                "labels": loaded["labels"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Load Raw Data
    print(f"Processing {split} data from scratch...")
    if split == "train":
        path = config.TRAIN_PATH
    elif split == "val":
        path = config.VAL_PATH
    else:
        path = config.TEST_PATH

    df = pd.read_csv(path)

    # Debug Subsampling
    if config.DEBUG:
        print(f"DEBUG mode: Subsampling {config.DEBUG_SAMPLE_SIZE} rows.")
        df = df.head(config.DEBUG_SAMPLE_SIZE)

    # 3. Augmentation (Train only)
    if split == "train" and config.USE_SYMMETRIC_AUGMENTATION:
        print("Applying Symmetric Augmentation (swapping A/B)...")
        df_swapped = df.copy()

        # Swap Responses
        df_swapped.rename(
            columns={"response_a": "temp_resp", "response_b": "response_a"},
            inplace=True,
        )
        df_swapped.rename(columns={"temp_resp": "response_b"}, inplace=True)

        # Swap Targets
        df_swapped.rename(
            columns={"winner_model_a": "temp_win", "winner_model_b": "winner_model_a"},
            inplace=True,
        )
        df_swapped.rename(columns={"temp_win": "winner_model_b"}, inplace=True)

        # Concatenate
        df = pd.concat([df, df_swapped], axis=0).reset_index(drop=True)
        print(f"Augmented Train Size: {len(df)}")

    # 4. Process
    data_dict = process_dataframe(df, tokenizer, config.MAX_LENGTH)

    # 5. Save Cache
    print(f"Saving processed {split} data to {cache_path}...")
    np.savez_compressed(
        cache_path,
        input_ids=data_dict["input_ids"],
        attention_mask=data_dict["attention_mask"],
        scalar_features=data_dict["scalar_features"],
        labels=data_dict["labels"],
    )

    return data_dict


def get_dataloaders(config, tokenizer, load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        config (Config): Configuration object.
        tokenizer: Tokenizer instance.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Ensure cache dir exists
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # --- Train ---
    train_data = load_and_process_data(config, tokenizer, "train", load_cached_data)
    train_dataset = SiameseDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # --- Validation ---
    val_data = load_and_process_data(config, tokenizer, "val", load_cached_data)
    val_dataset = SiameseDataset(val_data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test ---
    test_data = load_and_process_data(config, tokenizer, "test", load_cached_data)
    test_dataset = SiameseDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
