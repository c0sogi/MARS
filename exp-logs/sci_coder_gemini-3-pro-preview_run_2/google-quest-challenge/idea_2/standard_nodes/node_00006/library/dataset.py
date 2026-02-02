import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class QUESTDataset(Dataset):
    """
    PyTorch Dataset for Siamese DistilRoBERTa model.
    Holds pre-tokenized inputs for Question and Answer streams.
    """

    def __init__(
        self, q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, targets=None
    ):
        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.a_input_ids = a_input_ids
        self.a_attention_mask = a_attention_mask
        self.targets = targets

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

        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def _tokenize_texts(texts, tokenizer, max_length):
    """
    Helper function to tokenize a list of texts.
    Returns numpy arrays for input_ids and attention_mask.
    """
    encoded = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def get_dataloaders(tokenizer=None, load_cached_data=True, debug=False):
    """
    Loads data, processes/tokenizes it (with caching), and returns DataLoaders.

    Args:
        tokenizer: Pre-trained tokenizer instance. If None, loads distilroberta-base.
        load_cached_data (bool): Whether to load processed arrays from disk if available.
        debug (bool): If True, subsets the data to a small number of samples for debugging.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists for caching
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Define cache file paths
    cache_files = {
        "train_q_ids": os.path.join(Config.WORKING_DIR, "train_q_ids.npy"),
        "train_q_mask": os.path.join(Config.WORKING_DIR, "train_q_mask.npy"),
        "train_a_ids": os.path.join(Config.WORKING_DIR, "train_a_ids.npy"),
        "train_a_mask": os.path.join(Config.WORKING_DIR, "train_a_mask.npy"),
        "train_labels": os.path.join(Config.WORKING_DIR, "train_labels.npy"),
        "val_q_ids": os.path.join(Config.WORKING_DIR, "val_q_ids.npy"),
        "val_q_mask": os.path.join(Config.WORKING_DIR, "val_q_mask.npy"),
        "val_a_ids": os.path.join(Config.WORKING_DIR, "val_a_ids.npy"),
        "val_a_mask": os.path.join(Config.WORKING_DIR, "val_a_mask.npy"),
        "val_labels": os.path.join(Config.WORKING_DIR, "val_labels.npy"),
        "test_q_ids": os.path.join(Config.WORKING_DIR, "test_q_ids.npy"),
        "test_q_mask": os.path.join(Config.WORKING_DIR, "test_q_mask.npy"),
        "test_a_ids": os.path.join(Config.WORKING_DIR, "test_a_ids.npy"),
        "test_a_mask": os.path.join(Config.WORKING_DIR, "test_a_mask.npy"),
        "test_qa_ids": os.path.join(
            Config.WORKING_DIR, "test_qa_ids.npy"
        ),  # Cache QA IDs for submission mapping
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(path) for path in cache_files.values())

    data = {}

    if load_cached_data and cache_exists:
        print("Loading processed data from cache...")
        for key, path in cache_files.items():
            data[key] = np.load(path)
    else:
        print("Processing data from scratch...")
        # Load Raw Data
        train_df = pd.read_csv(Config.TRAIN_PATH)
        val_df = pd.read_csv(Config.VAL_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)

        # Helper to prepare text streams
        def prepare_text(df):
            # Stream 1: Title + SEP + Body
            q_text = (
                df["question_title"].fillna("").astype(str)
                + f" {tokenizer.sep_token} "
                + df["question_body"].fillna("").astype(str)
            ).tolist()
            # Stream 2: Answer
            a_text = df["answer"].fillna("").astype(str).tolist()
            return q_text, a_text

        # Process Train
        train_q_text, train_a_text = prepare_text(train_df)
        data["train_q_ids"], data["train_q_mask"] = _tokenize_texts(
            train_q_text, tokenizer, Config.MAX_LENGTH
        )
        data["train_a_ids"], data["train_a_mask"] = _tokenize_texts(
            train_a_text, tokenizer, Config.MAX_LENGTH
        )
        data["train_labels"] = train_df[Config.TARGET_COLS].values.astype(np.float32)

        # Process Val
        val_q_text, val_a_text = prepare_text(val_df)
        data["val_q_ids"], data["val_q_mask"] = _tokenize_texts(
            val_q_text, tokenizer, Config.MAX_LENGTH
        )
        data["val_a_ids"], data["val_a_mask"] = _tokenize_texts(
            val_a_text, tokenizer, Config.MAX_LENGTH
        )
        data["val_labels"] = val_df[Config.TARGET_COLS].values.astype(np.float32)

        # Process Test
        test_q_text, test_a_text = prepare_text(test_df)
        data["test_q_ids"], data["test_q_mask"] = _tokenize_texts(
            test_q_text, tokenizer, Config.MAX_LENGTH
        )
        data["test_a_ids"], data["test_a_mask"] = _tokenize_texts(
            test_a_text, tokenizer, Config.MAX_LENGTH
        )
        data["test_qa_ids"] = test_df["qa_id"].values

        # Save to cache
        print("Saving processed data to cache...")
        for key, arr in data.items():
            np.save(cache_files[key], arr)

    # Debug Slicing
    if debug:
        print("Debug mode: Slicing dataset to 100 samples.")
        slice_size = 100
        for key in data:
            data[key] = data[key][:slice_size]

    # Create Datasets
    train_dataset = QUESTDataset(
        data["train_q_ids"],
        data["train_q_mask"],
        data["train_a_ids"],
        data["train_a_mask"],
        data["train_labels"],
    )

    val_dataset = QUESTDataset(
        data["val_q_ids"],
        data["val_q_mask"],
        data["val_a_ids"],
        data["val_a_mask"],
        data["val_labels"],
    )

    test_dataset = QUESTDataset(
        data["test_q_ids"],
        data["test_q_mask"],
        data["test_a_ids"],
        data["test_a_mask"],
        targets=None,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
