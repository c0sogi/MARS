import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import List, Dict, Any, Optional, Union
from library.config import Config


def preprocess_data(
    input_path: str,
    output_path: str,
    load_cached_data: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Loads data with caching mechanism strictly following the requirements.

    Args:
        input_path: Path to the source CSV file.
        output_path: Path to the cache Parquet file.
        load_cached_data: Whether to attempt loading from cache.
        debug: If True, subsamples the data for rapid iteration.

    Returns:
        pd.DataFrame: The loaded data.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Logic Flow: 1. Try to load cache
    if load_cached_data and os.path.exists(output_path):
        try:
            df = pd.read_parquet(output_path)
            # If debug is on but cached file is full size (or vice versa),
            # we might ideally reload, but strict logic says if load_cached_data is True, load it.
            # We assume the cache matches the debug state or user manages cache invalidation.
            return df
        except Exception as e:
            print(f"Failed to load cache from {output_path}: {e}")
            # Fallthrough to process from scratch

    # Logic Flow: 2. Process from scratch
    df = pd.read_csv(input_path)

    # Handle Debugging
    if debug:
        df = df.sample(n=min(100, len(df)), random_state=Config.seed).reset_index(
            drop=True
        )

    # Save to cache
    try:
        df.to_parquet(output_path, index=False)
    except Exception as e:
        print(f"Failed to save cache to {output_path}: {e}")

    return df


class EssayDataset(Dataset):
    """
    Dataset class for Essay Scoring using Ordinal Classification.
    """

    def __init__(
        self,
        data_path: str,
        processed_path: str,
        load_cached_data: bool = True,
        is_test: bool = False,
        debug: bool = False,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        """
        Args:
            data_path: Path to the metadata CSV.
            processed_path: Path for caching the processed parquet file.
            load_cached_data: Whether to use cached data.
            is_test: Whether this is the test set (no labels).
            debug: Whether to run in debug mode (subset of data).
            tokenizer: Optional pre-initialized tokenizer. If None, loads from Config.
        """
        self.is_test = is_test
        self.debug = debug
        self.max_length = Config.max_length
        self.num_labels = Config.num_labels

        # Load Tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

        # Load Data
        self.df = preprocess_data(
            input_path=data_path,
            output_path=processed_path,
            load_cached_data=load_cached_data,
            debug=debug,
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = row["full_text"]
        essay_id = row["essay_id"]

        # Tokenize
        # We do not pad here; padding is handled in the Collate function for dynamic batching
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }

        # Handle Targets for Train/Val
        if not self.is_test:
            score = row["score"]
            sample["labels"] = torch.tensor(score, dtype=torch.float32)
        else:
            # Dummy labels for test set to maintain consistent structure if needed
            sample["labels"] = torch.tensor(0.0, dtype=torch.float32)

        return sample


class Collate:
    """
    Collator for dynamic padding of batches.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        essay_ids = [item["essay_id"] for item in batch]

        # Dynamic padding using tokenizer
        # Convert lists to tensors implicitly via pad
        batch_inputs = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        output = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "essay_ids": essay_ids,
        }

        # Stack labels if present
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
            output["labels"] = labels

        return output
