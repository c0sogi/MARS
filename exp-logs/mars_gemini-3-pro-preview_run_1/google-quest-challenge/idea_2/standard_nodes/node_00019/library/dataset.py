import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QADataset(Dataset):
    """
    PyTorch Dataset for Question-Answering regression task.
    Wraps pre-tokenized numpy arrays and converts them to tensors.
    """

    def __init__(
        self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, labels=None
    ):
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.a_input_ids = a_input_ids
        self.a_attention_mask = a_attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.q_input_ids)

    def __getitem__(self, idx):
        item = {
            "q_input_ids": torch.tensor(self.q_input_ids[idx], dtype=torch.long),
            "q_attention_mask": torch.tensor(
                self.q_attention_mask[idx], dtype=torch.long
            ),
            "a_input_ids": torch.tensor(self.a_input_ids[idx], dtype=torch.long),
            "a_attention_mask": torch.tensor(
                self.a_attention_mask[idx], dtype=torch.long
            ),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


def process_and_cache_data(df, tokenizer, max_len, cache_prefix, load_cached_data=True):
    """
    Tokenizes data and caches the results to disk as .npy files.

    Args:
        df (pd.DataFrame): Input dataframe containing text and optionally targets.
        tokenizer: Transformers tokenizer.
        max_len (int): Maximum sequence length.
        cache_prefix (str): Prefix for cache filenames (e.g., 'train', 'val').
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays for inputs and labels.
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define filenames
    files = {
        "q_input_ids": os.path.join(cache_dir, f"{cache_prefix}_q_input_ids.npy"),
        "q_mask": os.path.join(cache_dir, f"{cache_prefix}_q_mask.npy"),
        "a_input_ids": os.path.join(cache_dir, f"{cache_prefix}_a_input_ids.npy"),
        "a_mask": os.path.join(cache_dir, f"{cache_prefix}_a_mask.npy"),
        "labels": os.path.join(cache_dir, f"{cache_prefix}_labels.npy"),
    }

    # Check if we need to process labels
    has_labels = all(col in df.columns for col in Config.TARGET_COLS)

    # logic to determine if we can load from cache
    can_load = load_cached_data
    required_files = ["q_input_ids", "q_mask", "a_input_ids", "a_mask"]
    if has_labels:
        required_files.append("labels")

    for key in required_files:
        if not os.path.exists(files[key]):
            can_load = False
            break

    if can_load:
        try:
            data = {
                "q_input_ids": np.load(files["q_input_ids"]),
                "q_mask": np.load(files["q_mask"]),
                "a_input_ids": np.load(files["a_input_ids"]),
                "a_mask": np.load(files["a_mask"]),
            }
            if has_labels:
                data["labels"] = np.load(files["labels"])
            else:
                data["labels"] = None
            return data
        except Exception as e:
            print(f"Failed to load cache for {cache_prefix}: {e}. Recomputing...")
            # Fall through to computation

    # Compute from scratch
    print(f"Tokenizing {cache_prefix} data...")

    # Prepare text
    # Question branch: Title + Body
    # We use a space to separate. The tokenizer handles special tokens (CLS/SEP) via batch_encode_plus
    q_text = (
        df["question_title"].fillna("") + " " + df["question_body"].fillna("")
    ).tolist()
    a_text = df["answer"].fillna("").tolist()

    # Tokenize Question Branch
    q_encoded = tokenizer.batch_encode_plus(
        q_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    # Tokenize Answer Branch
    a_encoded = tokenizer.batch_encode_plus(
        a_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )

    # Save to cache
    np.save(files["q_input_ids"], q_encoded["input_ids"])
    np.save(files["q_mask"], q_encoded["attention_mask"])
    np.save(files["a_input_ids"], a_encoded["input_ids"])
    np.save(files["a_mask"], a_encoded["attention_mask"])

    labels_np = None
    if has_labels:
        labels_np = df[Config.TARGET_COLS].values.astype(np.float32)
        np.save(files["labels"], labels_np)

    data = {
        "q_input_ids": q_encoded["input_ids"],
        "q_mask": q_encoded["attention_mask"],
        "a_input_ids": a_encoded["input_ids"],
        "a_mask": a_encoded["attention_mask"],
        "labels": labels_np,
    }

    return data


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        debug (bool): If True, uses a small subset of data.
        load_cached_data (bool): Whether to use cached numpy files.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_PATH)
    val_df = pd.read_csv(Config.VAL_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)

    if debug:
        train_df = train_df.iloc[:100]
        val_df = val_df.iloc[:100]
        test_df = test_df.iloc[:100]
        # Modify prefix to avoid overwriting full cache with debug data
        prefix_suffix = "_debug"
    else:
        prefix_suffix = ""

    # Process Train
    train_data = process_and_cache_data(
        train_df, tokenizer, Config.MAX_LEN, f"train{prefix_suffix}", load_cached_data
    )
    train_dataset = QADataset(
        train_data["q_input_ids"],
        train_data["q_mask"],
        train_data["a_input_ids"],
        train_data["a_mask"],
        train_data["labels"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Process Val
    val_data = process_and_cache_data(
        val_df, tokenizer, Config.MAX_LEN, f"val{prefix_suffix}", load_cached_data
    )
    val_dataset = QADataset(
        val_data["q_input_ids"],
        val_data["q_mask"],
        val_data["a_input_ids"],
        val_data["a_mask"],
        val_data["labels"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Process Test
    test_data = process_and_cache_data(
        test_df, tokenizer, Config.MAX_LEN, f"test{prefix_suffix}", load_cached_data
    )
    test_dataset = QADataset(
        test_data["q_input_ids"],
        test_data["q_mask"],
        test_data["a_input_ids"],
        test_data["a_mask"],
        labels=None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
