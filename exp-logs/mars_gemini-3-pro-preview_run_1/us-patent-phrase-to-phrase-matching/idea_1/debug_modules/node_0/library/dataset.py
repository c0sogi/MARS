import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


def load_and_preprocess(split, tokenizer=None, load_cached_data=True):
    """
    Loads data, preprocesses (tokenizes), and caches the result.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer: HuggingFace tokenizer. If None, loaded from Config.
        load_cached_data (bool): Whether to try loading from cache.

    Returns:
        dict: Dictionary containing numpy arrays of tokenized data.
    """
    # 1. Setup Paths and Directories
    os.makedirs(Config.working_dir, exist_ok=True)

    debug_suffix = "_debug" if Config.debug else ""
    cache_filename = f"cached_{split}{debug_suffix}.npz"
    cache_path = os.path.join(Config.working_dir, cache_filename)

    # 2. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            # Load the npz file.
            # We convert the NpzFile object to a dict to keep data in memory after closing.
            with np.load(cache_path) as cached_data:
                data = {key: cached_data[key] for key in cached_data.files}
            return data
        except Exception:
            # If loading fails, proceed to compute from scratch
            pass

    # 3. Compute from scratch
    if split == "train":
        input_path = Config.train_path
    elif split == "val":
        input_path = Config.val_path
    elif split == "test":
        input_path = Config.test_path
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load CSV
    df = pd.read_csv(input_path)

    # Handle Debug mode
    if Config.debug:
        df = df.head(Config.debug_subset_size)

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Prepare inputs for tokenization
    # We pair Context with Anchor, and Context with Target.
    # Transformer tokenizers handle pairs as: [CLS] Sequence A [SEP] Sequence B [SEP]
    contexts = df["context"].astype(str).tolist()
    anchors = df["anchor"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()

    # Tokenize Anchor Inputs (Context + Anchor)
    anchor_encodings = tokenizer(
        contexts,
        anchors,
        padding="max_length",
        truncation=True,
        max_length=Config.max_length,
        return_tensors="np",
    )

    # Tokenize Target Inputs (Context + Target)
    target_encodings = tokenizer(
        contexts,
        targets,
        padding="max_length",
        truncation=True,
        max_length=Config.max_length,
        return_tensors="np",
    )

    # Construct data dictionary
    data = {
        "anchor_input_ids": anchor_encodings["input_ids"],
        "anchor_attention_mask": anchor_encodings["attention_mask"],
        "target_input_ids": target_encodings["input_ids"],
        "target_attention_mask": target_encodings["attention_mask"],
    }

    # Add scores if available (Train/Val)
    if "score" in df.columns:
        data["scores"] = df["score"].values.astype(np.float32)

    # Add IDs (Train/Val/Test) - needed for submission
    if "id" in df.columns:
        data["ids"] = df["id"].values.astype(str)

    # 4. Save to cache
    np.savez(cache_path, **data)

    return data


class PhraseDataset(Dataset):
    def __init__(self, split, tokenizer=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            tokenizer: HuggingFace tokenizer instance.
        """
        self.split = split

        # Load processed data (cached or computed)
        data = load_and_preprocess(
            split, tokenizer=tokenizer, load_cached_data=Config.load_cached_data
        )

        # Assign data to attributes
        self.anchor_input_ids = data["anchor_input_ids"]
        self.anchor_attention_mask = data["anchor_attention_mask"]
        self.target_input_ids = data["target_input_ids"]
        self.target_attention_mask = data["target_attention_mask"]

        # Optional attributes
        self.scores = data["scores"] if "scores" in data else None
        self.ids = data["ids"] if "ids" in data else None

    def __len__(self):
        return len(self.anchor_input_ids)

    def __getitem__(self, idx):
        item = {
            "anchor_input_ids": torch.tensor(
                self.anchor_input_ids[idx], dtype=torch.long
            ),
            "anchor_attention_mask": torch.tensor(
                self.anchor_attention_mask[idx], dtype=torch.long
            ),
            "target_input_ids": torch.tensor(
                self.target_input_ids[idx], dtype=torch.long
            ),
            "target_attention_mask": torch.tensor(
                self.target_attention_mask[idx], dtype=torch.long
            ),
        }

        if self.scores is not None:
            item["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        if self.ids is not None:
            item["id"] = str(self.ids[idx])

        return item
