import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from library.config import Config
from library.utils import seed_everything


def get_cpc_texts(load_cached_data=True):
    """
    Parses the CPC description file to map codes (e.g., 'A47') to their full text descriptions.
    Implements caching using Parquet.
    """
    cache_path = os.path.join(Config.working_dir, "cpc_texts.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            df_cpc = pd.read_parquet(cache_path)
            # Convert to dictionary for fast lookup
            return dict(zip(df_cpc["code"], df_cpc["text"]))
        except Exception as e:
            print(f"Failed to load cached CPC texts: {e}. Recomputing...")

    # 2. Compute from scratch
    cpc_path = Config.cpc_context_path
    cpc_data = {}

    if os.path.exists(cpc_path):
        with open(cpc_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Assuming format: "CODE Description text..."
                # We split by the first space
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    code, text = parts
                    cpc_data[code] = text
                else:
                    # Fallback if format is unexpected
                    pass
    else:
        print(f"Warning: CPC description file not found at {cpc_path}")

    # 3. Save to cache
    os.makedirs(Config.working_dir, exist_ok=True)
    df_cpc = pd.DataFrame(list(cpc_data.items()), columns=["code", "text"])
    df_cpc.to_parquet(cache_path, index=False)

    return cpc_data


def get_folds(df, n_splits=5):
    """
    Creates Stratified Group K-Folds.
    Groups by 'anchor' to prevent leakage.
    Stratifies by 'score' to maintain class balance.
    """
    seed_everything(Config.seed)

    # Create a new column for folds
    df["fold"] = -1

    # Map scores to integers for stratification (0.0 -> 0, 0.25 -> 1, etc.)
    # This ensures sklearn treats them as discrete classes
    score_mapper = {val: i for i, val in enumerate(sorted(df["score"].unique()))}
    y_strat = df["score"].map(score_mapper)

    # Group by anchor
    groups = df["anchor"]

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=Config.seed
    )

    for fold_id, (_, val_idx) in enumerate(sgkf.split(df, y_strat, groups)):
        df.loc[val_idx, "fold"] = fold_id

    return df


class PearsonDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        """
        Dataset class for the Pearson Correlation task.

        Args:
            df (pd.DataFrame): Dataframe containing 'anchor', 'target', 'context', and optionally 'score'.
                               Should ideally have 'context_text' merged beforehand, but we handle it if passed.
            tokenizer: HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            is_test (bool): Whether this is a test set (no targets).
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-tokenize special tokens to save time
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Retrieve texts
        anchor = str(row["anchor"]).lower()
        target = str(row["target"]).lower()
        # If 'context_text' is merged, use it. Otherwise fallback to raw context code.
        context = str(row.get("context_text", row["context"])).lower()

        # Tokenize components separately without special tokens
        # We manually construct the sequence to ensure: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        tok_ctx = self.tokenizer.encode(context, add_special_tokens=False)
        tok_anc = self.tokenizer.encode(anchor, add_special_tokens=False)
        tok_tar = self.tokenizer.encode(target, add_special_tokens=False)

        # Construct Input IDs
        # Format: [CLS] Context [SEP] Anchor [SEP] Target [SEP]
        input_ids = (
            [self.cls_token_id]
            + tok_ctx
            + [self.sep_token_id]
            + tok_anc
            + [self.sep_token_id]
            + tok_tar
            + [self.sep_token_id]
        )

        # Truncation
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            # Ensure last token is SEP if we truncated (optional but good practice)
            if input_ids[-1] != self.sep_token_id:
                input_ids[-1] = self.sep_token_id

        # Padding
        mask_len = len(input_ids)
        padding_len = self.max_len - mask_len

        input_ids = input_ids + [self.pad_token_id] * padding_len
        attention_mask = [1] * mask_len + [0] * padding_len

        # Convert to tensors
        d = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        # Add label if not test
        if not self.is_test:
            score = float(row["score"])
            d["label"] = torch.tensor(score, dtype=torch.float)

        return d
