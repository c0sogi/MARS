import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.utils import get_metadata


class StackExchangeDataset(Dataset):
    def __init__(
        self,
        split="train",
        tokenizer_name="roberta-base",
        max_length=512,
        cache_dir="./working/idea_22",
        load_cached_data=True,
        debug=False,
        num_debug_samples=100,
    ):
        """
        Dataset class for StackExchange Question-Answer pairs.

        Args:
            split (str): 'train', 'val', or 'test'.
            tokenizer_name (str): Name of the pre-trained tokenizer.
            max_length (int): Maximum sequence length for tokenization.
            cache_dir (str): Directory to store cached parquet files.
            load_cached_data (bool): Whether to try loading from cache.
            debug (bool): If True, restricts dataset size for debugging.
            num_debug_samples (int): Number of samples to use in debug mode.
        """
        self.split = split
        self.max_length = max_length
        self.debug = debug
        self.num_debug_samples = num_debug_samples
        self.cache_dir = cache_dir

        # Initialize tokenizer
        # We use use_fast=True to ensure we get sequence_ids() method
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        # Load and process data
        self.data = self._process_data(load_cached_data)

        # Identify target columns
        # Based on the prompt, targets are the last 30 columns in train.csv
        # We can identify them by filtering for 'question_' and 'answer_' prefixes
        # excluding the text features.
        all_cols = self.data.columns.tolist()
        text_cols = ["question_title", "question_body", "answer"]
        self.target_cols = [
            c
            for c in all_cols
            if (c.startswith("question_") or c.startswith("answer_"))
            and c not in text_cols
            and pd.api.types.is_numeric_dtype(self.data[c])
        ]
        # Ensure we have exactly 30 targets if we are in train/val
        if self.split in ["train", "val"] and len(self.target_cols) != 30:
            # Fallback or warning could go here, but we assume metadata is correct
            pass

    def _process_data(self, load_cached_data):
        """
        Loads data from metadata, processes it, and handles caching.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = os.path.join(self.cache_dir, f"{self.split}_processed.parquet")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if self.debug:
                    df = df.head(self.num_debug_samples)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Re-processing.")

        # 2. Process from scratch
        df = get_metadata(self.split)

        # Fill NaNs in text columns to avoid tokenizer errors
        text_cols = ["question_title", "question_body", "answer"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        # Save to cache (before debug slicing to cache the full set)
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            print(f"Failed to save cache: {e}")

        # Apply debug slicing
        if self.debug:
            df = df.head(self.num_debug_samples)

        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Extract text
        q_title = row.get("question_title", "")
        q_body = row.get("question_body", "")
        answer = row.get("answer", "")

        # --- Question Branch Processing ---
        # Tokenize Title + Body as a pair
        # Truncation strategy: 'only_second' to preserve Title (Intent)
        q_enc = self.tokenizer(
            q_title,
            q_body,
            truncation="only_second",
            max_length=self.max_length,
            padding=False,  # Dynamic padding in collate_fn
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids_q = q_enc["input_ids"]
        attention_mask_q = q_enc["attention_mask"]

        # Generate Partition Masks
        # sequence_ids: None (special), 0 (title), 1 (body)
        seq_ids = q_enc.sequence_ids()

        # Create binary masks
        title_mask = [1 if s == 0 else 0 for s in seq_ids]
        body_mask = [1 if s == 1 else 0 for s in seq_ids]

        # --- Answer Branch Processing ---
        a_enc = self.tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        input_ids_a = a_enc["input_ids"]
        attention_mask_a = a_enc["attention_mask"]

        # --- Targets ---
        labels = None
        if self.split in ["train", "val"]:
            # Extract target values
            labels = row[self.target_cols].values.astype(np.float32)

        # Return dict
        item = {
            "qa_id": row["qa_id"],
            "input_ids_q": torch.tensor(input_ids_q, dtype=torch.long),
            "attention_mask_q": torch.tensor(attention_mask_q, dtype=torch.long),
            "title_mask": torch.tensor(
                title_mask, dtype=torch.float
            ),  # Float for pooling math
            "body_mask": torch.tensor(
                body_mask, dtype=torch.float
            ),  # Float for pooling math
            "input_ids_a": torch.tensor(input_ids_a, dtype=torch.long),
            "attention_mask_a": torch.tensor(attention_mask_a, dtype=torch.long),
        }

        if labels is not None:
            item["labels"] = torch.tensor(labels, dtype=torch.float)

        return item


def collate_fn(batch):
    """
    Custom collate function to handle dynamic padding for multiple inputs.
    """
    # Keys to stack/pad
    keys = batch[0].keys()
    output = {}

    # QA IDs (list)
    output["qa_id"] = [b["qa_id"] for b in batch]

    # Helper for padding
    def pad_tensor(vecs, pad_val=0):
        return torch.nn.utils.rnn.pad_sequence(
            vecs, batch_first=True, padding_value=pad_val
        )

    # Question Branch
    output["input_ids_q"] = pad_tensor(
        [b["input_ids_q"] for b in batch], pad_val=1
    )  # 1 is <pad> for RoBERTa
    output["attention_mask_q"] = pad_tensor(
        [b["attention_mask_q"] for b in batch], pad_val=0
    )
    output["title_mask"] = pad_tensor([b["title_mask"] for b in batch], pad_val=0)
    output["body_mask"] = pad_tensor([b["body_mask"] for b in batch], pad_val=0)

    # Answer Branch
    output["input_ids_a"] = pad_tensor([b["input_ids_a"] for b in batch], pad_val=1)
    output["attention_mask_a"] = pad_tensor(
        [b["attention_mask_a"] for b in batch], pad_val=0
    )

    # Labels
    if "labels" in batch[0]:
        output["labels"] = torch.stack([b["labels"] for b in batch])

    return output
