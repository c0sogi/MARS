import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from library.config import CFG
from library.cpc_loader import CPCLoader


class CPCDataset(Dataset):
    """
    Dataset class for Patent Phrase Matching.
    Prepares inputs for the model by combining Context, Anchor, and Target.
    """

    def __init__(self, df, tokenizer, config=CFG, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

        # Pre-extract data for faster access in __getitem__
        self.anchors = df["anchor"].values.astype(str)
        self.targets = df["target"].values.astype(str)
        self.contexts = df["context_text"].values.astype(str)

        if self.mode != "test":
            self.labels = df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct input text segments
        # Segment A: Context + Anchor (The "Premise")
        # Segment B: Target (The "Hypothesis")
        # This structure allows the Cross-Encoder to evaluate the Target against the Contextualized Anchor.
        text_a = context + " " + anchor
        text_b = target

        inputs = self.tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.config.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors=None,  # Return python lists, convert to tensor manually
        )

        # Convert to tensors
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs:
            output["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if self.mode != "test":
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return output, label
        else:
            return output


def prepare_data(config=CFG, split="train", load_cached_data=True):
    """
    Loads, processes, and caches the dataset for a specific split.
    Handles CPC context expansion and Parquet caching.

    Args:
        config: Configuration object containing paths and settings.
        split: The dataset split to load ('train', 'val', 'test').
        load_cached_data: If True, attempts to load from the working directory cache.

    Returns:
        pd.DataFrame: The processed dataframe containing 'context_text'.
    """
    # Determine file paths based on split
    if split == "train":
        input_path = config.train_path
        cache_path = config.train_cache_path
    elif split == "val":
        input_path = config.val_path
        cache_path = config.val_cache_path
    elif split == "test":
        input_path = config.test_path
        cache_path = config.test_cache_path
    else:
        raise ValueError(f"Unknown split: {split}")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            # Apply debug sampling if needed
            if config.debug:
                df = df.iloc[: config.debug_sample_size].reset_index(drop=True)
            return df
        except Exception as e:
            # If cache loading fails, proceed to processing
            pass

    # 2. Process data from scratch
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path)

    # Load and expand CPC contexts
    cpc_loader = CPCLoader(config)
    # We pass load_cached_data to leverage the CPCLoader's own caching mechanism
    context_map = cpc_loader.get_cpc_texts(df, load_cached_data=load_cached_data)

    # Merge context descriptions into the main dataframe
    df = df.merge(context_map, on="context", how="left")

    # Fill any missing context texts with empty string
    df["context_text"] = df["context_text"].fillna("")

    # 3. Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")

    # 4. Apply Debug Subsampling
    if config.debug:
        df = df.iloc[: config.debug_sample_size].reset_index(drop=True)

    return df
