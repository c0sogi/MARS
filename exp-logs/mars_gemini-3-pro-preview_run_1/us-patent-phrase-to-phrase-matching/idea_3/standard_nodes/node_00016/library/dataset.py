import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from library.config import CFG
from library.cpc_mapper import get_cpc_texts


def get_data(cfg, load_cached_data=True):
    """
    Prepares the training and testing data.

    1. Loads metadata from ./metadata/train.csv and ./metadata/val.csv (combined for CV).
    2. Loads metadata from ./metadata/test.csv.
    3. Maps CPC codes to full descriptions using get_cpc_texts.
    4. Generates Stratified Group K-Fold splits.
    5. Caches the processed DataFrames to Parquet files.

    Args:
        cfg: Configuration object containing paths and settings.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        tuple: (train_df, test_df)
    """
    # Ensure working directory exists
    os.makedirs(cfg.WORKING_DIR, exist_ok=True)

    # Define cache filenames based on debug mode
    suffix = "_debug" if cfg.debug else ""
    train_cache_path = os.path.join(cfg.WORKING_DIR, f"train_processed{suffix}.parquet")
    test_cache_path = os.path.join(cfg.WORKING_DIR, f"test_processed{suffix}.parquet")

    # 1. Try Loading from Cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        try:
            train_df = pd.read_parquet(train_cache_path)
            test_df = pd.read_parquet(test_cache_path)
            return train_df, test_df
        except Exception as e:
            print(f"Cache load failed: {e}. Recomputing...")

    # 2. Compute from Scratch

    # Load Metadata
    # Combine train and val metadata for full Cross-Validation
    # We use the metadata files which are already split, but for CV we need the full set
    # to perform our own StratifiedGroupKFold
    df_train_meta = pd.read_csv(cfg.TRAIN_METADATA)
    df_val_meta = pd.read_csv(cfg.VAL_METADATA)
    train_df = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

    test_df = pd.read_csv(cfg.TEST_METADATA)

    # Load CPC Descriptions
    # Pass load_cached_data to the mapper as well
    cpc_texts = get_cpc_texts(cfg, load_cached_data=load_cached_data)

    # Map Context Codes to Descriptions
    train_df["context_text"] = train_df["context"].map(cpc_texts).fillna("")
    test_df["context_text"] = test_df["context"].map(cpc_texts).fillna("")

    # Debug Mode: Subset Data
    if cfg.debug:
        # Take a small subset
        train_df = train_df.head(1000).reset_index(drop=True)
        test_df = test_df.head(100).reset_index(drop=True)

    # Create Folds
    # We use StratifiedGroupKFold to prevent leakage (group by anchor) and maintain class balance (stratify by score)
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed
    )

    train_df["fold"] = -1

    # Note: StratifiedGroupKFold expects discrete y. 'score' is float but has discrete levels (0, 0.25, 0.5, 0.75, 1.0).
    # We convert to string to ensure sklearn treats it as multiclass rather than continuous.
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(train_df, train_df["score"].astype(str), groups=train_df["anchor"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    # 3. Save to Cache
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return train_df, test_df


class PhraseDataset(Dataset):
    """
    PyTorch Dataset for the Phrase Similarity Task.
    Tokenizes inputs in the format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
    """

    def __init__(self, cfg, df, tokenizer, mode="train"):
        """
        Args:
            cfg: Configuration object.
            df: DataFrame containing the data.
            tokenizer: Pre-trained tokenizer.
            mode: 'train' (returns labels) or 'test' (no labels).
        """
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.mode = mode

        # Extract columns to numpy arrays for efficient indexing
        self.anchors = self.df["anchor"].values.astype(str)
        self.targets = self.df["target"].values.astype(str)
        self.contexts = self.df["context_text"].values.astype(str)

        if self.mode == "train":
            self.labels = self.df["score"].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context = self.contexts[idx]

        # Construct Input
        # We want: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # DeBERTa tokenizer(text, text_pair) produces [CLS] text [SEP] text_pair [SEP]
        # So we set:
        # text = context
        # text_pair = anchor + [SEP] + target

        sep = self.tokenizer.sep_token
        text = context
        text_pair = anchor + sep + target

        inputs = self.tokenizer(
            text,
            text_pair,
            add_special_tokens=True,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return lists
        )

        # Convert to tensors
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs:
            item["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if self.mode == "train":
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item
