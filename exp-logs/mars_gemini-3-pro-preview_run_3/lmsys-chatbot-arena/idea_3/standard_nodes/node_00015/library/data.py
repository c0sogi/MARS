import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer
from library.config import Config
from library.utils import seed_everything


class ChatbotDataset(Dataset):
    """
    Dataset class for the Chatbot Arena task using Longformer.
    Handles tokenization and extraction of scalar features (length statistics).
    """

    def __init__(self, data, tokenizer, max_length=None, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length is not None else Config.MAX_LENGTH
        self.is_test = is_test

        # Unpack data arrays
        self.ids = data["id"]
        self.prompts = data["prompt"]
        self.responses_a = data["response_a"]
        self.responses_b = data["response_b"]

        if not self.is_test:
            self.targets = data["targets"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Ensure inputs are strings
        prompt = str(self.prompts[idx])
        resp_a = str(self.responses_a[idx])
        resp_b = str(self.responses_b[idx])

        # Tokenize components separately to calculate scalar features
        # We use the tokenizer to get accurate token counts
        tokens_prompt = self.tokenizer.tokenize(prompt)
        tokens_a = self.tokenizer.tokenize(resp_a)
        tokens_b = self.tokenizer.tokenize(resp_b)

        # Calculate explicit scalar features: Log-transformed token counts
        # These help the model distinguish length biases
        len_p = np.log1p(len(tokens_prompt))
        len_a = np.log1p(len(tokens_a))
        len_b = np.log1p(len(tokens_b))
        scalar_features = torch.tensor([len_p, len_a, len_b], dtype=torch.float32)

        # Construct the input sequence for Longformer
        # Format: <s> Prompt </s> Response A </s> Response B </s>
        bos_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token

        # Manually construct the token list to ensure correct structure
        full_tokens = (
            [bos_token]
            + tokens_prompt
            + [sep_token]
            + tokens_a
            + [sep_token]
            + tokens_b
            + [sep_token]
        )

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(full_tokens)

        # Truncate to max_length if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            # Note: We simply truncate the end. The Longformer's long context (2048/4096)
            # minimizes the risk of losing critical information compared to standard BERT.

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "scalar_features": scalar_features,
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


class CollateFn:
    """
    Custom collate function to handle dynamic padding and Global Attention Mask generation.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        scalar_features = torch.stack([item["scalar_features"] for item in batch])
        ids = [item["id"] for item in batch]

        # Dynamic padding to the longest sequence in the batch
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        # Create global_attention_mask for Longformer
        # 0: Local Attention, 1: Global Attention
        # We apply global attention to the <s> (CLS) token at index 0
        global_attention_mask = torch.zeros_like(input_ids_padded)
        global_attention_mask[:, 0] = 1

        batch_output = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "global_attention_mask": global_attention_mask,
            "scalar_features": scalar_features,
            "id": torch.tensor(ids, dtype=torch.long),
        }

        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            batch_output["labels"] = labels

        return batch_output


def load_and_cache_data(split, load_cached_data=True):
    """
    Loads data from CSV files or a cached NPZ file.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays of data.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, f"{split}_data.npz")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            if "id" not in loaded.files:
                raise ValueError("Cache missing required 'id' key")
            data = {key: loaded[key] for key in loaded.files}
            return data
        except Exception as e:
            print(f"Failed to load cache for {split}: {e}. Reloading from CSV.")

    # 2. Load from CSV
    if split == "train":
        file_path = Config.TRAIN_DATA_PATH
    elif split == "val":
        file_path = Config.VAL_DATA_PATH
    else:
        file_path = Config.TEST_DATA_PATH

    df = pd.read_csv(file_path)

    # Debugging subset
    if Config.DEBUG:
        df = df.head(Config.DEBUG_SIZE)

    # Extract text columns and handle NaNs
    data = {
        "id": df["id"].values,
        "prompt": df["prompt"].fillna("").values.astype(str),
        "response_a": df["response_a"].fillna("").values.astype(str),
        "response_b": df["response_b"].fillna("").values.astype(str),
    }

    # Extract targets for train/val
    if split != "test":
        targets = df[Config.TARGET_COLS].values.astype(np.float32)
        data["targets"] = targets

    # 3. Save to cache
    np.savez(cache_path, **data)

    return data


def get_dataloaders(load_cached_data=True):
    """
    Initializes the Tokenizer, Datasets, and DataLoaders.

    Args:
        load_cached_data (bool): Whether to use cached data files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Initialize Tokenizer
    tokenizer = LongformerTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Data
    train_data = load_and_cache_data("train", load_cached_data)
    val_data = load_and_cache_data("val", load_cached_data)
    test_data = load_and_cache_data("test", load_cached_data)

    # Create Datasets
    train_dataset = ChatbotDataset(train_data, tokenizer, is_test=False)
    val_dataset = ChatbotDataset(val_data, tokenizer, is_test=False)
    test_dataset = ChatbotDataset(test_data, tokenizer, is_test=True)

    # Initialize Collate Function
    collate_fn = CollateFn(tokenizer)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
