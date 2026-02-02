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

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], np.ndarray, int]:
        """
        Returns:
            t_indices (List[int]): Token indices for Title.
            b_indices (List[int]): Token indices for Body.
            target (np.ndarray): Multi-hot encoded tags (or zeros if test).
            id (int): Question ID.
        """
        row = self.data.iloc[int(idx)]

        title = str(row["Title"])
        body = str(row["Body"])
        q_id = int(row["Id"])

        # Tokenize (returns list of tuples, we take the first one)
        t_indices, b_indices = self.tokenizer.encode_text([title], [body])[0]

        if self.has_tags:
            tags_str = str(row["Tags"])
            target = self.tokenizer.encode_tags([tags_str])[0]
        else:
            num_tags = self.tokenizer.get_num_tags()
            target = np.zeros(num_tags, dtype=np.float32)

        return t_indices, b_indices, target, q_id

    @staticmethod
    def collate_fn(
        batch: List[Tuple[List[int], List[int], np.ndarray, int]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Custom collate function for EmbeddingBag with separate Title and Body.
        """
        t_indices_list = [item[0] for item in batch]
        b_indices_list = [item[1] for item in batch]
        targets_list = [item[2] for item in batch]
        ids_list = [item[3] for item in batch]

        def process_stream(indices_list):
            offsets = [0]
            for seq in indices_list[:-1]:
                offsets.append(offsets[-1] + len(seq))
            offsets_tensor = torch.tensor(offsets, dtype=torch.long)
            flat_indices = [idx for seq in indices_list for idx in seq]
            text_indices_tensor = torch.tensor(flat_indices, dtype=torch.long)
            return text_indices_tensor, offsets_tensor

        t_text, t_offsets = process_stream(t_indices_list)
        b_text, b_offsets = process_stream(b_indices_list)

        targets_tensor = torch.tensor(np.array(targets_list), dtype=torch.float32)
        ids_tensor = torch.tensor(ids_list, dtype=torch.long)

        return t_text, t_offsets, b_text, b_offsets, targets_tensor, ids_tensor
