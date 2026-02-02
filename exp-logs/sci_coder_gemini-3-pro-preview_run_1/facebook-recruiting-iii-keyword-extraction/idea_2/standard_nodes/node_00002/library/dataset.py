import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional

from library.config import Config


class StackExchangeDataset(Dataset):
    """
    PyTorch Dataset for Stack Exchange Tag Prediction.
    Loads data from metadata and raw CSV files, merges them, and caches the result.
    Prepares data for an EmbeddingBag-based model (FastText architecture).
    """

    def __init__(
        self,
        metadata_path: str,
        tokenizer,
        split_name: str = "train",
        load_cached_data: bool = True,
    ):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            tokenizer (TextProcessor): Instance of the tokenizer.
            split_name (str): Name of the split (train/val/test) for caching purposes.
            load_cached_data (bool): Whether to load from parquet cache if available.
        """
        self.tokenizer = tokenizer
        self.split_name = split_name

        # Ensure working directory exists for caching
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        self.cache_path = os.path.join(
            Config.WORKING_DIR, f"cached_{split_name}.parquet"
        )

        self.data = self._load_data(metadata_path, load_cached_data)

        # Check if we have labels
        self.has_tags = "Tags" in self.data.columns

    def _load_data(self, metadata_path: str, load_cached_data: bool) -> pd.DataFrame:
        """
        Loads data with caching mechanism.
        """
        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"[{self.split_name}] Loading cached data from {self.cache_path}...")
            try:
                df = pd.read_parquet(self.cache_path)
                return df
            except Exception as e:
                print(f"[{self.split_name}] Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"[{self.split_name}] Loading metadata from {metadata_path}...")
        df_meta = pd.read_csv(metadata_path)

        if df_meta.empty:
            return df_meta

        # Identify source file (assuming all rows in metadata come from the same source file type)
        # The metadata generation script adds 'file_path' column.
        source_file = df_meta["file_path"].iloc[0]
        full_source_path = os.path.join(Config.INPUT_DIR, source_file)

        print(f"[{self.split_name}] Reading raw text from {full_source_path}...")
        # Read Id, Title, Body
        # We use string dtype for text to avoid issues with mixed types
        try:
            df_text = pd.read_csv(
                full_source_path,
                usecols=["Id", "Title", "Body"],
                dtype={"Id": "int64", "Title": "object", "Body": "object"},
            )
        except ValueError:
            # Fallback if columns are different (unlikely given problem spec)
            df_text = pd.read_csv(full_source_path)

        # Merge metadata with text
        print(f"[{self.split_name}] Merging metadata and text...")
        df_merged = df_meta.merge(df_text, on="Id", how="inner")

        # Fill NaNs in text columns
        df_merged["Title"] = df_merged["Title"].fillna("")
        df_merged["Body"] = df_merged["Body"].fillna("")

        # 3. Save to cache
        print(f"[{self.split_name}] Saving cache to {self.cache_path}...")
        df_merged.to_parquet(self.cache_path, index=False)

        return df_merged

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], np.ndarray, int]:
        """
        Returns:
            indices (List[int]): Token indices for Title + Body.
            target (np.ndarray): Multi-hot encoded tags (or zeros if test).
            id (int): Question ID.
        """
        row = self.data.iloc[idx]

        title = str(row["Title"])
        body = str(row["Body"])
        q_id = int(row["Id"])

        # Tokenize (returns list of lists, we take the first one)
        # We pass single items as lists to reuse the batch encoding method
        indices = self.tokenizer.encode_text([title], [body])[0]

        if self.has_tags:
            tags_str = str(row["Tags"])
            # encode_tags returns (batch_size, num_tags), take first
            target = self.tokenizer.encode_tags([tags_str])[0]
        else:
            # Zero vector for test set
            num_tags = self.tokenizer.get_num_tags()
            target = np.zeros(num_tags, dtype=np.float32)

        return indices, target, q_id

    @staticmethod
    def collate_fn(
        batch: List[Tuple[List[int], np.ndarray, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Custom collate function for EmbeddingBag.

        Args:
            batch: List of tuples (indices, target, id)

        Returns:
            text_indices: 1D Tensor of concatenated indices.
            offsets: 1D Tensor of starting positions.
            targets: 2D Tensor of targets (batch_size, num_tags).
            ids: 1D Tensor of question IDs.
        """
        indices_list = [item[0] for item in batch]
        targets_list = [item[1] for item in batch]
        ids_list = [item[2] for item in batch]

        # 1. Create offsets and flat indices
        # EmbeddingBag requires a 1D tensor of all indices and a 1D tensor of offsets
        offsets = [0]
        for seq in indices_list[:-1]:
            offsets.append(offsets[-1] + len(seq))

        offsets_tensor = torch.tensor(offsets, dtype=torch.long)

        # Flatten indices
        flat_indices = [idx for seq in indices_list for idx in seq]
        text_indices_tensor = torch.tensor(flat_indices, dtype=torch.long)

        # 2. Stack targets
        targets_tensor = torch.tensor(np.array(targets_list), dtype=torch.float32)

        # 3. Stack IDs
        ids_tensor = torch.tensor(ids_list, dtype=torch.long)

        return text_indices_tensor, offsets_tensor, targets_tensor, ids_tensor
