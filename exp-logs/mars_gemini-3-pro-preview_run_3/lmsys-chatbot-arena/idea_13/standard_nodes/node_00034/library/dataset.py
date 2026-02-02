import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import get_logger, seed_everything

# Initialize Logger
logger = get_logger("dataset", "dataset.log")


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for Siamese Chatbot Preference Prediction.
    Handles loading pre-processed arrays and applying symmetric augmentation.
    """

    def __init__(self, data, mode="train", augment=False):
        """
        Args:
            data (dict): Dictionary containing numpy arrays of inputs and targets.
            mode (str): 'train', 'val', or 'test'.
            augment (bool): Whether to apply symmetric augmentation (swap A/B).
        """
        self.mode = mode
        self.augment = augment

        # Inputs for Branch A
        self.input_ids_a = data["input_ids_a"]
        self.attention_mask_a = data["attention_mask_a"]
        self.token_type_ids_a = data["token_type_ids_a"]

        # Inputs for Branch B
        self.input_ids_b = data["input_ids_b"]
        self.attention_mask_b = data["attention_mask_b"]
        self.token_type_ids_b = data["token_type_ids_b"]

        # Scalars: [log_len_prompt, log_len_resp_a, log_len_resp_b]
        self.scalars = data["scalars"]

        # Targets: [winner_a, winner_b, tie]
        # Test set might not have targets
        if "targets" in data:
            self.targets = data["targets"]
        else:
            self.targets = None

        self.length = len(self.input_ids_a)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load sample data
        ids_a = self.input_ids_a[idx]
        mask_a = self.attention_mask_a[idx]
        type_a = self.token_type_ids_a[idx]

        ids_b = self.input_ids_b[idx]
        mask_b = self.attention_mask_b[idx]
        type_b = self.token_type_ids_b[idx]

        scalars = self.scalars[idx]  # [p, a, b]

        if self.targets is not None:
            target = self.targets[idx]
        else:
            target = np.zeros(3, dtype=np.float32)  # Dummy for test

        # Symmetric Augmentation: Randomly swap A and B
        # Only during training if enabled
        if self.mode == "train" and self.augment and np.random.rand() > 0.5:
            # Swap Inputs
            ids_a, ids_b = ids_b, ids_a
            mask_a, mask_b = mask_b, mask_a
            type_a, type_b = type_b, type_a

            # Swap Scalars: [p, a, b] -> [p, b, a]
            # scalars[0] is prompt, scalars[1] is A, scalars[2] is B
            scalars = np.array([scalars[0], scalars[2], scalars[1]], dtype=np.float32)

            # Swap Targets: [win_a, win_b, tie] -> [win_b, win_a, tie]
            target = np.array([target[1], target[0], target[2]], dtype=np.float32)

        # Convert to tensors
        return {
            "input_ids_a": torch.tensor(ids_a, dtype=torch.long),
            "attention_mask_a": torch.tensor(mask_a, dtype=torch.long),
            "token_type_ids_a": torch.tensor(type_a, dtype=torch.long),
            "input_ids_b": torch.tensor(ids_b, dtype=torch.long),
            "attention_mask_b": torch.tensor(mask_b, dtype=torch.long),
            "token_type_ids_b": torch.tensor(type_b, dtype=torch.long),
            "scalars": torch.tensor(scalars, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


def _tokenize_and_format(prompt, response, tokenizer, max_len):
    """
    Helper to tokenize prompt and response, enforce truncation preserving prompt,
    and generate token_type_ids.

    Structure: [CLS] Prompt [SEP] Response [SEP]
    Type IDs:   0     0      0     1        1
    """
    # Tokenize without special tokens first to handle lengths manually
    ids_p = tokenizer.encode(prompt, add_special_tokens=False)
    ids_r = tokenizer.encode(response, add_special_tokens=False)

    len_p = len(ids_p)
    len_r = len(ids_r)

    # Reserve space for [CLS], [SEP], [SEP]
    special_count = 3
    allowed_len = max_len - special_count

    # Truncation Logic
    if len_p + len_r > allowed_len:
        if len_p < allowed_len:
            # Prompt fits, truncate response
            len_r_new = allowed_len - len_p
            ids_r = ids_r[:len_r_new]
        else:
            # Prompt itself is too long. Truncate prompt, drop response (or keep minimal).
            # Priority is Prompt.
            ids_p = ids_p[:allowed_len]
            ids_r = []

    # Reconstruct
    cls_token = [tokenizer.cls_token_id]
    sep_token = [tokenizer.sep_token_id]

    input_ids = cls_token + ids_p + sep_token + ids_r + sep_token

    # Generate Token Type IDs
    # 0 for Context (CLS + Prompt + SEP)
    # 1 for Content (Response + SEP)
    len_part1 = 1 + len(ids_p) + 1  # CLS, P, SEP
    len_part2 = len(ids_r) + 1  # R, SEP

    token_type_ids = [0] * len_part1 + [1] * len_part2
    attention_mask = [1] * len(input_ids)

    # Padding
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        token_type_ids += [0] * pad_len  # Pad type usually 0, mask handles it
        attention_mask += [0] * pad_len

    return input_ids, attention_mask, token_type_ids, len_p, len(ids_r)


def preprocess_and_cache(
    csv_path, cache_path, tokenizer, max_len, load_cached_data=True, is_test=False
):
    """
    Loads CSV, tokenizes data, computes scalars, and caches to .npz.
    """
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        return np.load(cache_path)

    logger.info(f"Processing data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Fill NaNs in text columns to avoid errors
    df["prompt"] = df["prompt"].fillna("")
    df["response_a"] = df["response_a"].fillna("")
    df["response_b"] = df["response_b"].fillna("")

    # Initialize lists
    input_ids_a, attention_mask_a, token_type_ids_a = [], [], []
    input_ids_b, attention_mask_b, token_type_ids_b = [], [], []
    scalars = []
    targets = []

    # Process rows
    # Using simple loop as dataset size (40k) is manageable without multiprocessing complexity
    for idx, row in df.iterrows():
        p = str(row["prompt"])
        r_a = str(row["response_a"])
        r_b = str(row["response_b"])

        # Branch A
        ids_a, mask_a, type_a, len_p, len_ra = _tokenize_and_format(
            p, r_a, tokenizer, max_len
        )

        # Branch B
        ids_b, mask_b, type_b, _, len_rb = _tokenize_and_format(
            p, r_b, tokenizer, max_len
        )

        input_ids_a.append(ids_a)
        attention_mask_a.append(mask_a)
        token_type_ids_a.append(type_a)

        input_ids_b.append(ids_b)
        attention_mask_b.append(mask_b)
        token_type_ids_b.append(type_b)

        # Scalars: log1p of lengths
        # Note: len_p is consistent across branches due to truncation logic unless prompt > max_len
        # We use the computed lengths from tokenization
        s = [np.log1p(len_p), np.log1p(len_ra), np.log1p(len_rb)]
        scalars.append(s)

        # Targets
        if not is_test:
            t = [row["winner_model_a"], row["winner_model_b"], row["winner_tie"]]
            targets.append(t)

    # Convert to numpy arrays
    data = {
        "input_ids_a": np.array(input_ids_a, dtype=np.int32),
        "attention_mask_a": np.array(attention_mask_a, dtype=np.int8),
        "token_type_ids_a": np.array(token_type_ids_a, dtype=np.int8),
        "input_ids_b": np.array(input_ids_b, dtype=np.int32),
        "attention_mask_b": np.array(attention_mask_b, dtype=np.int8),
        "token_type_ids_b": np.array(token_type_ids_b, dtype=np.int8),
        "scalars": np.array(scalars, dtype=np.float32),
    }

    if not is_test:
        data["targets"] = np.array(targets, dtype=np.float32)

    # Save to cache
    logger.info(f"Saving processed data to {cache_path}")
    np.savez_compressed(cache_path, **data)

    return data


def get_dataloaders(load_cached_data=True):
    """
    Main entry point to get DataLoaders for Train, Val, and Test.
    """
    seed_everything(Config.SEED)

    # Initialize Tokenizer
    # We use the fast tokenizer to speed up processing
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)

    # Define Cache Paths
    train_cache = os.path.join(Config.CACHE_DIR, "train_data.npz")
    val_cache = os.path.join(Config.CACHE_DIR, "val_data.npz")
    test_cache = os.path.join(Config.CACHE_DIR, "test_data.npz")

    # 1. Train Data
    train_data = preprocess_and_cache(
        Config.TRAIN_PATH,
        train_cache,
        tokenizer,
        Config.MAX_LENGTH,
        load_cached_data,
        is_test=False,
    )

    train_dataset = ChatbotDataset(
        train_data, mode="train", augment=Config.SYMMETRIC_AUGMENTATION
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Validation Data
    val_data = preprocess_and_cache(
        Config.VAL_PATH,
        val_cache,
        tokenizer,
        Config.MAX_LENGTH,
        load_cached_data,
        is_test=False,
    )

    val_dataset = ChatbotDataset(val_data, mode="val", augment=False)

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Test Data
    test_data = preprocess_and_cache(
        Config.TEST_PATH,
        test_cache,
        tokenizer,
        Config.MAX_LENGTH,
        load_cached_data,
        is_test=True,
    )

    test_dataset = ChatbotDataset(test_data, mode="test", augment=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
