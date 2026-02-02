import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from library.config import CFG
from library.cpc_utils import get_cpc_texts
from library.utils import seed_everything


def get_folds(cfg=CFG, load_cached_data=True):
    """
    Generates or loads the training data with stratified k-fold splits.

    Args:
        cfg: Configuration class containing paths and settings.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: DataFrame containing the training data with a 'fold' column.
    """
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    cache_path = os.path.join(cfg.output_dir, "folds.parquet")

    # 1. Attempt to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            print(f"Loading folds from cache: {cache_path}")
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load folds from cache: {e}. Regenerating...")

    # 2. Generate Folds
    print("Generating stratified k-folds...")
    seed_everything(cfg.seed)

    # Load training metadata
    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(f"Train file not found at {cfg.train_path}")

    df = pd.read_csv(cfg.train_path)

    # Create 'score_cat' for stratification to ensure discrete classes
    df["score_cat"] = df["score"].astype(str)

    kf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(df, df["score_cat"])):
        df.loc[val_idx, "fold"] = fold

    df = df.drop(columns=["score_cat"])

    # 3. Save to cache
    print(f"Saving folds to cache: {cache_path}")
    df.to_parquet(cache_path, index=False)

    return df


class CPCDataset(Dataset):
    def __init__(self, df, tokenizer, cpc_texts, max_len=128, is_test=False):
        """
        Dataset class for Phrase Matching.

        Args:
            df (pd.DataFrame): DataFrame containing 'anchor', 'target', 'context', and optionally 'score'.
            tokenizer (PreTrainedTokenizerBase): Transformer tokenizer.
            cpc_texts (dict): Dictionary mapping context codes to description strings.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is a test set (no labels).
        """
        self.df = df
        self.tokenizer = tokenizer
        self.cpc_texts = cpc_texts
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to arrays for faster access
        self.contexts = df["context"].values
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.ids = df["id"].values
        if not self.is_test:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        context_code = self.contexts[idx]
        anchor = self.anchors[idx]
        target = self.targets[idx]

        # Retrieve context description
        context_text = self.cpc_texts.get(context_code, "")

        # Construct Input Sequence
        # Structure: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        # We pass context as the first segment, and anchor+target as the second segment.
        sep = self.tokenizer.sep_token
        text = context_text
        text_pair = f"{anchor}{sep}{target}"

        inputs = self.tokenizer(
            text,
            text_pair,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,  # Padding handled by collator
            truncation=True,
            return_token_type_ids=True,
        )

        # Prepare output dictionary
        output = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }

        if "token_type_ids" in inputs:
            output["token_type_ids"] = torch.tensor(
                inputs["token_type_ids"], dtype=torch.long
            )

        if not self.is_test:
            score = self.scores[idx]
            output["labels"] = torch.tensor(score, dtype=torch.float)

        output["ids"] = self.ids[idx]

        return output


class CustomCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def __call__(self, features):
        # Separate non-tensor features (like 'ids') which DataCollatorWithPadding might not handle
        ids = [f.pop("ids") for f in features]

        # Collate tensor features using dynamic padding
        batch = self.pad_collator(features)

        # Add ids back to the batch object
        batch["ids"] = ids
        return batch


def get_collate_fn(tokenizer):
    """
    Returns the collate function for dynamic padding.
    """
    return CustomCollator(tokenizer=tokenizer)
