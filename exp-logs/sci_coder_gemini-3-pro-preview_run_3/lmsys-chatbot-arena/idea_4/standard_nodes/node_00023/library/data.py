import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from library.config import Config


class ChatbotDataset(Dataset):
    """
    PyTorch Dataset for the Chatbot Arena task.
    Constructs Cross-Encoder inputs: [CLS] Prompt [SEP] Response A [SEP] Response B [SEP]
    Cite solution_lesson_node_00006: Switching to Cross-Encoder for Transformer architectures.
    """

    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        self.prompts = data["prompt"].fillna("").astype(str).tolist()
        self.responses_a = data["response_a"].fillna("").astype(str).tolist()
        self.responses_b = data["response_b"].fillna("").astype(str).tolist()

        if not self.is_test:
            self.targets = data[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.astype("float32")

        # Pre-fetch special token IDs
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        resp_a = self.responses_a[idx]
        resp_b = self.responses_b[idx]

        # 1. Scalars (Cite solution_lesson_node_00004)
        feat_prompt = np.log1p(len(prompt))
        feat_resp_a = np.log1p(len(resp_a))
        feat_resp_b = np.log1p(len(resp_b))
        scalars = torch.tensor(
            [feat_prompt, feat_resp_a, feat_resp_b], dtype=torch.float32
        )

        # 2. Tokenization & Truncation
        # We manually tokenize to control truncation logic (Cite solution_lesson_node_00013)
        # Goal: [CLS] Prompt [SEP] Resp A [SEP] Resp B [SEP]

        ids_p = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids_a = self.tokenizer.encode(resp_a, add_special_tokens=False)
        ids_b = self.tokenizer.encode(resp_b, add_special_tokens=False)

        # Budget calculation
        # 4 special tokens: CLS, SEP, SEP, SEP
        max_seq_len = self.max_length
        budget = max_seq_len - 4

        len_p = len(ids_p)
        len_a = len(ids_a)
        len_b = len(ids_b)

        # Truncation Strategy:
        # Cap Prompt at 40% of budget to ensure responses have space
        # Split remaining space evenly between A and B

        prompt_cap = int(budget * 0.4)
        if len_p > prompt_cap:
            ids_p = ids_p[:prompt_cap]
            len_p = prompt_cap

        remaining = budget - len_p
        half_remaining = remaining // 2

        # Allocate space for A and B
        # If one is short, give more to the other
        if len_a <= half_remaining and len_b <= half_remaining:
            # Both fit
            pass
        elif len_a <= half_remaining:
            # A fits, B takes rest
            ids_b = ids_b[: (remaining - len_a)]
        elif len_b <= half_remaining:
            # B fits, A takes rest
            ids_a = ids_a[: (remaining - len_b)]
        else:
            # Both long, truncate both to half
            ids_a = ids_a[:half_remaining]
            ids_b = ids_b[: (remaining - len(ids_a))]

        # Construct Sequence
        input_ids = (
            [self.cls_token_id]
            + ids_p
            + [self.sep_token_id]
            + ids_a
            + [self.sep_token_id]
            + ids_b
            + [self.sep_token_id]
        )
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_length = max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "scalars": scalars,
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def load_and_process_data(mode, load_cached_data=True):
    """
    Loads data from metadata CSVs or cached Parquet files.
    Implements the required caching mechanism.

    Args:
        mode (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {mode} data from cache: {cache_path} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"Failed to load cache for {mode}: {e}. Reloading from source.")

    # 2. Load from source metadata
    if mode == "train":
        source_path = Config.TRAIN_DATA_PATH
    elif mode == "val":
        source_path = Config.VAL_DATA_PATH
    elif mode == "test":
        source_path = Config.TEST_DATA_PATH
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Loading {mode} data from metadata: {source_path}")
    df = pd.read_csv(source_path)

    # 3. Save to cache
    try:
        df.to_parquet(cache_path, index=False)
        print(f"Cached {mode} data to: {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to cache data: {e}")

    return df


def get_dataloaders(tokenizer, batch_size=None, load_cached_data=True):
    """
    Creates training and validation dataloaders.

    Args:
        tokenizer: The HuggingFace tokenizer.
        batch_size (int, optional): Batch size. Defaults to Config.TRAIN_BATCH_SIZE.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if batch_size is None:
        batch_size = Config.TRAIN_BATCH_SIZE

    # Load Data
    train_df = load_and_process_data("train", load_cached_data)
    val_df = load_and_process_data("val", load_cached_data)

    # Debug Subsampling
    if Config.DEBUG:
        print(f"DEBUG Mode: Subsampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Create Datasets
    train_ds = ChatbotDataset(train_df, tokenizer, Config.MAX_LENGTH, is_test=False)
    val_ds = ChatbotDataset(val_df, tokenizer, Config.MAX_LENGTH, is_test=False)

    # Create Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_dataloader(tokenizer, batch_size=None, load_cached_data=True):
    """
    Creates the test dataloader.

    Args:
        tokenizer: The HuggingFace tokenizer.
        batch_size (int, optional): Batch size. Defaults to Config.VALID_BATCH_SIZE.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        DataLoader: The test dataloader.
    """
    if batch_size is None:
        batch_size = Config.VALID_BATCH_SIZE

    test_df = load_and_process_data("test", load_cached_data)

    test_ds = ChatbotDataset(test_df, tokenizer, Config.MAX_LENGTH, is_test=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
