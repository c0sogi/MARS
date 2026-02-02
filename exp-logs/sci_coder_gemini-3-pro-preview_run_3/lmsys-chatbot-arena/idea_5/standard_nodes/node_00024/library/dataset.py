import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import Optional, List, Dict
from library.config import Config


class ChatbotArenaDataset(Dataset):
    """
    PyTorch Dataset for the Chatbot Arena task.
    Serves Siamese inputs (Branch A and Branch B) along with scalar features and targets.
    """

    def __init__(self, df: pd.DataFrame, is_test: bool = False):
        self.df = df
        self.is_test = is_test

        # Pre-convert columns to lists to avoid overhead during __getitem__
        # Branch A
        self.input_ids_a = df["input_ids_a"].tolist()
        self.attention_mask_a = df["attention_mask_a"].tolist()
        self.response_mask_a = df["response_mask_a"].tolist()

        # Branch B
        self.input_ids_b = df["input_ids_b"].tolist()
        self.attention_mask_b = df["attention_mask_b"].tolist()
        self.response_mask_b = df["response_mask_b"].tolist()

        # Scalars
        self.scalars = df["scalars"].tolist()

        # Targets
        if not self.is_test:
            self.targets = df[
                ["winner_model_a", "winner_model_b", "winner_tie"]
            ].values.tolist()
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Stack Branch A and Branch B
        # Shape: [2, max_length]
        input_ids = torch.tensor(
            [self.input_ids_a[idx], self.input_ids_b[idx]], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [self.attention_mask_a[idx], self.attention_mask_b[idx]], dtype=torch.long
        )
        response_mask = torch.tensor(
            [self.response_mask_a[idx], self.response_mask_b[idx]], dtype=torch.long
        )

        # Scalars: [log(prompt_len), log(resp_a_len), log(resp_b_len)]
        scalars = torch.tensor(self.scalars[idx], dtype=torch.float)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "scalars": scalars,
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item


def _preprocess_batch(
    batch_df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> pd.DataFrame:
    """
    Helper function to tokenize a batch of text and extract features.
    """
    prompts = batch_df["prompt"].fillna("").astype(str).tolist()
    resp_a = batch_df["response_a"].fillna("").astype(str).tolist()
    resp_b = batch_df["response_b"].fillna("").astype(str).tolist()

    # Tokenize Branch A: Prompt + Response A
    enc_a = tokenizer(
        prompts,
        resp_a,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,  # Ensure we can get sequence_ids logic if needed, though we use sequence_ids() method
    )

    # Tokenize Branch B: Prompt + Response B
    enc_b = tokenizer(
        prompts,
        resp_b,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
    )

    # Storage for processed rows
    processed_rows = []

    for i in range(len(prompts)):
        # --- Branch A Processing ---
        ids_a = enc_a.input_ids[i]
        mask_a = enc_a.attention_mask[i]
        seq_ids_a = enc_a.sequence_ids(i)

        # Generate Response Mask A (1 for Response tokens, 0 otherwise)
        # sequence_ids: None (special), 0 (prompt), 1 (response)
        resp_mask_a = [1 if s == 1 else 0 for s in seq_ids_a]

        # Calculate Lengths for Scalars (based on token counts)
        # Prompt length is count of 0s
        len_prompt = sum(1 for s in seq_ids_a if s == 0)
        len_resp_a = sum(1 for s in seq_ids_a if s == 1)

        # --- Branch B Processing ---
        ids_b = enc_b.input_ids[i]
        mask_b = enc_b.attention_mask[i]
        seq_ids_b = enc_b.sequence_ids(i)

        resp_mask_b = [1 if s == 1 else 0 for s in seq_ids_b]
        len_resp_b = sum(1 for s in seq_ids_b if s == 1)

        # --- Scalar Features ---
        # Log-transformed lengths: [log(prompt), log(resp_a), log(resp_b)]
        # Add 1 to avoid log(0)
        scalar_feats = [
            np.log1p(len_prompt),
            np.log1p(len_resp_a),
            np.log1p(len_resp_b),
        ]

        processed_rows.append(
            {
                "input_ids_a": ids_a,
                "attention_mask_a": mask_a,
                "response_mask_a": resp_mask_a,
                "input_ids_b": ids_b,
                "attention_mask_b": mask_b,
                "response_mask_b": resp_mask_b,
                "scalars": scalar_feats,
            }
        )

    return pd.DataFrame(processed_rows)


def load_dataset(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    load_cached_data: bool = True,
    limit: Optional[int] = None,
) -> ChatbotArenaDataset:
    """
    Loads, processes, and caches the dataset.

    Args:
        split (str): 'train', 'val', or 'test'.
        tokenizer: HuggingFace tokenizer.
        load_cached_data (bool): Whether to try loading from cache.
        limit (int, optional): If set, limits the dataset size (for debugging).

    Returns:
        ChatbotArenaDataset: The ready-to-use PyTorch dataset.
    """
    # Determine paths
    if split == "train":
        input_path = Config.TRAIN_DATA_PATH
        cache_path = Config.TRAIN_CACHE_FILE
    elif split == "val":
        input_path = Config.VAL_DATA_PATH
        cache_path = Config.VAL_CACHE_FILE
    elif split == "test":
        input_path = Config.TEST_DATA_PATH
        cache_path = Config.TEST_CACHE_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    # Ensure output directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Try loading cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {split} data from {cache_path}...")
        df = pd.read_parquet(cache_path)

        # Apply limit if requested
        if limit is not None:
            df = df.head(limit)

        return ChatbotArenaDataset(df, is_test=(split == "test"))

    # Process from scratch
    print(f"Processing {split} data from {input_path}...")
    raw_df = pd.read_csv(input_path)

    if limit is not None:
        raw_df = raw_df.head(limit)

    # Process in chunks to avoid excessive memory usage during tokenization
    chunk_size = 1000
    processed_chunks = []

    # Iterate through raw dataframe
    for start_idx in range(0, len(raw_df), chunk_size):
        end_idx = min(start_idx + chunk_size, len(raw_df))
        batch_df = raw_df.iloc[start_idx:end_idx]

        processed_chunk = _preprocess_batch(batch_df, tokenizer, Config.MAX_LENGTH)

        # Copy targets/IDs if they exist
        if "id" in batch_df.columns:
            processed_chunk["id"] = batch_df["id"].values

        if split != "test":
            processed_chunk["winner_model_a"] = batch_df["winner_model_a"].values
            processed_chunk["winner_model_b"] = batch_df["winner_model_b"].values
            processed_chunk["winner_tie"] = batch_df["winner_tie"].values

        processed_chunks.append(processed_chunk)

    final_df = pd.concat(processed_chunks, ignore_index=True)

    # Save to cache
    print(f"Saving processed {split} data to {cache_path}...")
    final_df.to_parquet(cache_path, index=False)

    return ChatbotArenaDataset(final_df, is_test=(split == "test"))
