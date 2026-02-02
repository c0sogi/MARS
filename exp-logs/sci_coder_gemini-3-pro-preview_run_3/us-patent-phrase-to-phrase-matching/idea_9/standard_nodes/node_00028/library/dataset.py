import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library import utils


def load_data(partition, load_cached_data=True):
    """
    Loads the dataset for a specific partition (train, val, test).
    Implements caching to Parquet to avoid re-processing context texts.

    Args:
        partition (str): One of 'train', 'val', 'test'.
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    cache_path = os.path.join(Config.working_dir, f"{partition}_cache.parquet")

    # 1. Try loading from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {partition} data from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
    else:
        # 2. Process from metadata
        print(f"Processing {partition} data from metadata...")
        source_path = os.path.join(Config.metadata_dir, f"{partition}.csv")

        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Metadata file not found: {source_path}")

        df = pd.read_csv(source_path)

        # Expand Context Codes to Text
        # We pre-compute this to save time during training
        print("Expanding context codes...")
        cpc_texts = utils.get_cpc_texts()

        # Ensure context is string
        df["context"] = df["context"].astype(str)

        # Apply expansion
        df["context_text"] = df["context"].apply(
            lambda x: utils.get_expanded_cpc_text(x, cpc_texts)
        )

        # Save to cache
        print(f"Saving {partition} data to cache: {cache_path}")
        df.to_parquet(cache_path, index=False)

    # 3. Handle Debug Mode
    if Config.debug:
        print(
            f"Debug mode enabled. Sampling {Config.debug_sample_size} rows from {partition}."
        )
        # Reset index to avoid issues with __getitem__ indexing
        df = df.head(Config.debug_sample_size).reset_index(drop=True)

    print(f"Final {partition} dataset shape: {df.shape}")
    return df


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for Phrase Matching.
    Tokenizes inputs in the format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Prepare text components
        # Ensure texts are strings and lowercased for consistency
        anchor = str(row["anchor"]).lower()
        target = str(row["target"]).lower()
        context_text = str(row["context_text"]).lower()

        # Construct Input Sequence
        # We want the model to see: Context <SEP> Anchor <SEP> Target
        # Using the tokenizer's text_pair functionality:
        # Text A: Context
        # Text B: Anchor + SEP + Target

        sep = self.tokenizer.sep_token
        text_pair = f"{anchor} {sep} {target}"

        inputs = self.tokenizer(
            context_text,
            text_pair,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by return_tensors='pt'
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": row["id"],
        }

        # Add targets if available (Train/Val sets)
        if "score" in row:
            # Regression target
            item["labels"] = torch.tensor(row["score"], dtype=torch.float)

        return item


def get_dataset(partition, tokenizer, load_cached_data=True):
    """
    Factory function to create a PhraseDataset.

    Args:
        partition (str): 'train', 'val', or 'test'.
        tokenizer: Pre-trained tokenizer instance.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        PhraseDataset: The configured dataset.
    """
    df = load_data(partition, load_cached_data=load_cached_data)
    dataset = PhraseDataset(df, tokenizer, Config.max_len)
    return dataset
