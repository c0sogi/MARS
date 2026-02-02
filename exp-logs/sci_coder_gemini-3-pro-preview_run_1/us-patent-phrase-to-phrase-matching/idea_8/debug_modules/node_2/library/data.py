import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer, DataCollatorWithPadding
from library.config import CFG
from library.utils import seed_everything
from library.cpc_utils import get_cpc_texts


class PearsonDataset(Dataset):
    """
    Dataset class for Patent Phrase Similarity.
    Implements Native Pair Encoding:
    Segment A: Context Description + ". " + Anchor
    Segment B: Target
    """

    def __init__(self, df, tokenizer, max_len=133, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

        # Pre-extract columns to lists/arrays for faster access
        self.anchors = df["anchor"].values
        self.targets = df["target"].values
        self.context_texts = df["context_text"].values
        self.ids = df["id"].values
        if self.is_train:
            self.scores = df["score"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.anchors[idx]
        target = self.targets[idx]
        context_text = self.context_texts[idx]

        # Construct Segment A: Context + Anchor
        # We add a period to separate the context description from the anchor
        text_a = f"{context_text}. {anchor}"
        text_b = target

        # Tokenize with Native Pair Encoding
        # The tokenizer handles [CLS], [SEP] placement and token_type_ids
        inputs = self.tokenizer(
            text=text_a,
            text_pair=text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors=None,  # Return python lists for DataCollator
        )

        output = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        if "token_type_ids" in inputs:
            output["token_type_ids"] = inputs["token_type_ids"]

        if self.is_train:
            output["labels"] = torch.tensor(self.scores[idx], dtype=torch.float)

        return output


def preprocess_data(cfg=CFG, load_cached_data=True):
    """
    Loads raw metadata, merges CPC descriptions, creates folds, and caches the result.
    Strictly follows the caching logic requirement.
    """
    train_cache_path = os.path.join(cfg.working_dir, "train_folds.parquet")
    test_cache_path = os.path.join(cfg.working_dir, "test_processed.parquet")

    # Ensure working directory exists
    os.makedirs(cfg.working_dir, exist_ok=True)

    # 1. Try to load from cache
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
            print(f"Failed to load cache: {e}. Reprocessing...")

    # 2. Process from scratch
    print("Preprocessing data and creating folds...")

    # Load metadata
    # Note: We combine metadata/train.csv and metadata/val.csv to recreate the full training set
    # for Cross-Validation splitting, as we want to control the folds ourselves.
    # However, the task description implies we should use the provided splits or create our own.
    # Given we are doing K-Fold, we should merge train and val metadata if they are pre-split,
    # or just use the provided train.csv if it represents the full dev set.
    # Looking at the metadata generation script, train.csv and val.csv are splits of the full train.
    # To perform 5-fold CV properly, we should concatenate them.

    df_train_meta = pd.read_csv(cfg.train_file)
    df_val_meta = pd.read_csv(cfg.val_file)
    train_df = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    test_df = pd.read_csv(cfg.test_file)

    # Get CPC Context Descriptions
    cpc_texts = get_cpc_texts(cfg)

    # Map Context Codes to Descriptions
    train_df["context_text"] = train_df["context"].map(cpc_texts)
    test_df["context_text"] = test_df["context"].map(cpc_texts)

    # Handle missing context mappings (if any)
    train_df["context_text"] = train_df["context_text"].fillna("")
    test_df["context_text"] = test_df["context_text"].fillna("")

    # Create Stratified Group K-Folds
    # Groups: Anchor (prevent leakage)
    # Stratify: Score (balance classes)
    sgkf = StratifiedGroupKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )

    # We create a new column 'fold'
    train_df["fold"] = -1

    # Since score is continuous-like but discrete (0, 0.25...), we can stratify on it directly.
    # However, StratifiedGroupKFold expects integer targets often, but works with floats if discrete.
    # To be safe, we can map scores to integers or strings for stratification.
    stratify_target = train_df["score"].astype(str)

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(train_df, stratify_target, groups=train_df["anchor"])
    ):
        train_df.loc[val_idx, "fold"] = fold

    # 3. Save to Cache
    train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    print(f"Data processed and saved to {cfg.working_dir}")

    return train_df, test_df


def prepare_loaders(fold, tokenizer, cfg=CFG, debug=False):
    """
    Prepares DataLoaders for a specific fold.
    """
    # Load processed data
    train_df, _ = preprocess_data(cfg, load_cached_data=True)

    # Filter by fold
    df_train = train_df[train_df["fold"] != fold].reset_index(drop=True)
    df_val = train_df[train_df["fold"] == fold].reset_index(drop=True)

    # Debug mode: subset data
    if debug:
        df_train = df_train.sample(n=100, random_state=cfg.seed).reset_index(drop=True)
        df_val = df_val.sample(n=50, random_state=cfg.seed).reset_index(drop=True)

    # Create Datasets
    train_dataset = PearsonDataset(df_train, tokenizer, cfg.max_len, is_train=True)
    val_dataset = PearsonDataset(df_val, tokenizer, cfg.max_len, is_train=True)

    # Data Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,  # Can use larger batch size for val, but sticking to config
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_inference_loader(tokenizer, cfg=CFG):
    """
    Prepares DataLoader for the test set.
    """
    # Load processed data
    _, test_df = preprocess_data(cfg, load_cached_data=True)

    # Create Dataset
    test_dataset = PearsonDataset(test_df, tokenizer, cfg.max_len, is_train=False)

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.inference_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        drop_last=False,
    )

    return test_loader
