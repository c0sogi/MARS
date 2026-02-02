import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer

from library.config import Config
from library.utils import seed_everything


def get_cpc_texts():
    """
    Parses the CPC description file to map codes to their natural language descriptions.

    Returns:
        dict: A dictionary mapping CPC codes (e.g., 'A47') to descriptions.
    """
    contexts = {}
    if not os.path.exists(Config.cpc_path):
        print(f"Warning: CPC description file not found at {Config.cpc_path}")
        return contexts

    with open(Config.cpc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The format is typically "CODE Description"
            # We split on the first space
            parts = line.split(" ", 1)
            if len(parts) == 2:
                code, description = parts
                contexts[code] = description.lower()  # Normalize to lowercase
            else:
                # Handle cases where the line might not match expected format
                pass
    return contexts


def make_folds(df, n_folds=5, seed=42):
    """
    Creates Stratified Group K-Folds.
    Groups by 'anchor' to prevent leakage.
    Stratifies by 'score' to ensure balanced label distribution.
    """
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # StratifiedGroupKFold expects discrete labels for stratification.
    # The scores are discrete (0, 0.25, 0.5, 0.75, 1.0), so we can use them directly as y.
    # We convert score to string or int for stratification if needed, but float works with sklearn usually.
    # However, to be safe and explicit with class balancing:
    y = df["score"].astype(str)

    df["fold"] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y, groups=df["anchor"])):
        df.loc[val_idx, "fold"] = fold

    return df


def prepare_data(load_cached_data=True):
    """
    Loads, processes, and splits the data. Implements caching using Parquet.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, test_df)
            train_df contains 'fold', 'context_text', and original columns.
            test_df contains 'context_text' and original columns.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    cache_train_path = os.path.join(Config.working_dir, "train_folds.parquet")
    cache_test_path = os.path.join(Config.working_dir, "test_processed.parquet")

    # 1. Try Loading Cache
    if load_cached_data:
        if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
            print("Loading data from cache...")
            train_df = pd.read_parquet(cache_train_path)
            test_df = pd.read_parquet(cache_test_path)
            return train_df, test_df
        else:
            print("Cache not found. Processing from scratch...")
    else:
        print("Forcing data processing from scratch...")

    # 2. Load Metadata
    # We combine train and val from metadata to create our own Cross-Validation folds
    # consistent with the Config.n_folds parameter.
    df_train_meta = pd.read_csv(Config.train_path)
    df_val_meta = pd.read_csv(Config.val_path)
    train_df = pd.concat([df_train_meta, df_val_meta], axis=0).reset_index(drop=True)

    test_df = pd.read_csv(Config.test_path)

    # 3. Inject Context Descriptions
    cpc_texts = get_cpc_texts()

    # Map context codes to descriptions. Fallback to empty string if code not found.
    train_df["context_text"] = train_df["context"].map(cpc_texts).fillna("")
    test_df["context_text"] = test_df["context"].map(cpc_texts).fillna("")

    # 4. Create Folds
    train_df = make_folds(train_df, n_folds=Config.n_folds, seed=Config.seed)

    # 5. Save to Cache
    train_df.to_parquet(cache_train_path, index=False)
    test_df.to_parquet(cache_test_path, index=False)
    print(f"Data processed and saved to {Config.working_dir}")

    return train_df, test_df


class PhraseDataset(Dataset):
    """
    Dataset class for Semantic Similarity.
    Implements Native Pair Encoding:
    - Segment A: Context Description + " " + Anchor
    - Segment B: Target
    """

    def __init__(self, df, tokenizer, max_length=140, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # Pre-extract columns to lists for faster access
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.contexts = df["context_text"].values
        self.ids = df["id"].values
        if self.is_train:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context_text = self.contexts[idx]

        # Native Pair Encoding Strategy
        # Text A: Context + Anchor
        # Text B: Target
        # We add a space between context and anchor.
        first_segment = str(context_text) + " " + str(anchor)
        second_segment = str(target)

        # Tokenization
        # We pass text and text_pair to let the tokenizer handle special tokens (SEP) and token_type_ids
        inputs = self.tokenizer(
            text=first_segment,
            text_pair=second_segment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,  # Return python lists, converted to tensor by collator or manually below
            return_token_type_ids=True,  # Explicitly request token_type_ids for DeBERTa
        )

        # Convert to tensors
        item = {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
        }

        if self.is_train:
            label = torch.tensor(self.scores[idx], dtype=torch.float)
            item["labels"] = label

        # We can also return the ID for tracking during inference
        item["id"] = self.ids[idx]

        return item
