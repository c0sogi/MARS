import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import Config


class ToxicityDataset(Dataset):
    """
    Dataset class for Toxicity Classification.
    Handles loading metadata, merging with raw text, tokenization, and caching.
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_len: int = Config.max_len,
        load_cached_data: bool = True,
        debug: bool = Config.debug,
        debug_subset_size: int = Config.debug_subset_size,
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'.
            tokenizer (PreTrainedTokenizer): Transformers tokenizer.
            max_len (int): Maximum sequence length.
            load_cached_data (bool): Whether to load from cache if available.
            debug (bool): Whether to run in debug mode (subset of data).
            debug_subset_size (int): Number of samples in debug mode.
        """
        self.split = split
        self.max_len = max_len
        self.debug = debug

        # Ensure working directory exists
        os.makedirs(Config.working_dir, exist_ok=True)

        # Define cache file paths
        # We include 'debug' in filename to avoid loading partial data for full run
        suffix = "_debug" if debug else ""
        self.cache_input_ids = os.path.join(
            Config.working_dir, f"{split}{suffix}_input_ids.npy"
        )
        self.cache_attn_mask = os.path.join(
            Config.working_dir, f"{split}{suffix}_attention_mask.npy"
        )
        self.cache_labels = os.path.join(
            Config.working_dir, f"{split}{suffix}_labels.npy"
        )

        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        else:
            self.tokenizer = tokenizer

        # Try loading from cache
        if load_cached_data and self._check_cache_exists():
            print(f"Loading {split} data from cache...")
            self.input_ids = np.load(self.cache_input_ids)
            self.attention_mask = np.load(self.cache_attn_mask)

            if split in ["train", "val"]:
                self.labels = np.load(self.cache_labels)
            else:
                self.labels = None
        else:
            print(f"Processing {split} data from scratch...")
            self._process_data(debug, debug_subset_size)

    def _check_cache_exists(self):
        """Checks if all necessary cache files exist."""
        files = [self.cache_input_ids, self.cache_attn_mask]
        if self.split in ["train", "val"]:
            files.append(self.cache_labels)
        return all(os.path.exists(f) for f in files)

    def _process_data(self, debug, debug_subset_size):
        """Loads raw data, tokenizes, and saves to cache."""
        # 1. Load Metadata
        if self.split == "train":
            meta_path = Config.train_meta_path
        elif self.split == "val":
            meta_path = Config.val_meta_path
        elif self.split == "test":
            meta_path = Config.test_meta_path
        else:
            raise ValueError(f"Invalid split: {self.split}")

        meta_df = pd.read_csv(meta_path)

        # Handle Debug Mode
        if debug:
            meta_df = meta_df.iloc[:debug_subset_size].copy()

        # 2. Merge with Raw Text
        # Identify unique source files needed
        source_files = meta_df["source_file"].unique()

        merged_dfs = []
        for src_file in source_files:
            # Load raw data
            raw_path = os.path.join(Config.input_dir, src_file)
            raw_df = pd.read_csv(raw_path)

            # Filter metadata for this source
            subset_meta = meta_df[meta_df["source_file"] == src_file]

            # Merge on ID
            # We use inner join to attach text to the metadata rows
            merged = pd.merge(
                subset_meta, raw_df[["id", "comment_text"]], on="id", how="left"
            )
            merged_dfs.append(merged)

        df = pd.concat(merged_dfs, ignore_index=True)

        # Ensure text is string and handle NaNs
        df["comment_text"] = df["comment_text"].fillna("").astype(str)

        # 3. Tokenize
        print(f"Tokenizing {len(df)} texts...")
        encoded = self.tokenizer.batch_encode_plus(
            df["comment_text"].tolist(),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )

        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

        # 4. Extract Labels (if applicable)
        if self.split in ["train", "val"]:
            self.labels = df[Config.target_cols].values.astype(np.float32)
        else:
            self.labels = None

        # 5. Save to Cache
        print(f"Saving {self.split} data to cache at {Config.working_dir}...")
        np.save(self.cache_input_ids, self.input_ids)
        np.save(self.cache_attn_mask, self.attention_mask)

        if self.labels is not None:
            np.save(self.cache_labels, self.labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
