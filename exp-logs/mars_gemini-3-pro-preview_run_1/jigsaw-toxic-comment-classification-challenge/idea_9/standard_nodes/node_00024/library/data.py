import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from library.config import Config
from library.utils import seed_everything


def get_tokenizer():
    """
    Loads the DeBERTa-v3 tokenizer based on the configuration.
    """
    return AutoTokenizer.from_pretrained(Config.model_name)


class ToxicityDataset(Dataset):
    """
    Dataset for supervised toxicity classification.
    Returns input_ids, attention_mask, and labels (if available).
    """

    def __init__(self, texts, tokenizer, labels=None, max_len=Config.max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (MLM).
    Returns input_ids and attention_mask.
    Masking is typically handled by the DataCollator in the training loop.
    """

    def __init__(self, texts, tokenizer, max_len=Config.max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def _load_and_merge_data(load_cached_data=True):
    """
    Loads metadata, merges with raw text, and caches the result.
    Returns full_train_df and test_df.
    """
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "full_train_combined.parquet")
    test_cache_path = os.path.join(cache_dir, "test_full.parquet")

    # 1. Try loading from cache
    if (
        load_cached_data
        and os.path.exists(train_cache_path)
        and os.path.exists(test_cache_path)
    ):
        print(f"Loading cached data from {cache_dir}...")
        full_train_df = pd.read_parquet(train_cache_path)
        test_df = pd.read_parquet(test_cache_path)
        return full_train_df, test_df

    print("Loading data from source files and processing...")

    # 2. Load Metadata
    try:
        train_meta = pd.read_csv(Config.train_metadata_path)
        val_meta = pd.read_csv(Config.val_metadata_path)
        test_meta = pd.read_csv(Config.test_metadata_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Metadata files missing. Ensure metadata generation ran successfully. {e}"
        )

    # 3. Load Raw Data
    # Optimization: Read raw csvs once
    raw_train = pd.read_csv(Config.raw_train_path)
    raw_test = pd.read_csv(Config.raw_test_path)

    # 4. Merge
    # Combine train and val metadata to get full training set for CV
    full_train_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Merge with raw text
    # raw_train has 'id' and 'comment_text'
    full_train_df = pd.merge(
        full_train_meta, raw_train[["id", "comment_text"]], on="id", how="left"
    )

    # Merge test
    test_df = pd.merge(test_meta, raw_test[["id", "comment_text"]], on="id", how="left")

    # Fill NA
    full_train_df["comment_text"] = full_train_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    # 5. Cache
    print(f"Saving processed data to {cache_dir}...")
    full_train_df.to_parquet(train_cache_path, index=False)
    test_df.to_parquet(test_cache_path, index=False)

    return full_train_df, test_df


def prepare_mlm_loaders(debug=False, load_cached_data=True):
    """
    Prepares DataLoader for MLM pre-training using combined train and test text.
    """
    seed_everything(Config.seed)
    tokenizer = get_tokenizer()

    full_train_df, test_df = _load_and_merge_data(load_cached_data=load_cached_data)

    # Combine texts from both train and test for domain adaptation
    all_texts = (
        pd.concat([full_train_df["comment_text"], test_df["comment_text"]], axis=0)
        .astype(str)
        .tolist()
    )

    if debug:
        print(
            f"Debug mode: Reducing MLM dataset from {len(all_texts)} to {Config.debug_subset_size}"
        )
        all_texts = all_texts[: Config.debug_subset_size]

    dataset = MLMDataset(all_texts, tokenizer, max_len=Config.max_len)

    loader = DataLoader(
        dataset,
        batch_size=Config.mlm_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return loader


def prepare_kfold_loaders(fold, debug=False, load_cached_data=True):
    """
    Prepares DataLoaders for a specific fold in 5-Fold CV.
    """
    seed_everything(Config.seed)
    tokenizer = get_tokenizer()

    full_train_df, _ = _load_and_merge_data(load_cached_data=load_cached_data)

    if debug:
        print(
            f"Debug mode: Reducing Train dataset from {len(full_train_df)} to {Config.debug_subset_size}"
        )
        full_train_df = full_train_df.iloc[: Config.debug_subset_size].reset_index(
            drop=True
        )

    # Stratification Logic
    # Create a group key for stratification based on label combinations
    label_cols = Config.target_cols
    full_train_df["stratify_group"] = (
        full_train_df[label_cols].astype(str).agg("".join, axis=1)
    )

    # Handle rare groups for StratifiedKFold (groups with fewer samples than n_folds)
    group_counts = full_train_df["stratify_group"].value_counts()
    rare_groups = group_counts[group_counts < Config.n_folds].index
    full_train_df.loc[
        full_train_df["stratify_group"].isin(rare_groups), "stratify_group"
    ] = "rare"

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # Get indices for the requested fold
    # X is dummy, y is stratify_group
    splits = list(
        skf.split(X=np.zeros(len(full_train_df)), y=full_train_df["stratify_group"])
    )
    train_idx, val_idx = splits[fold]

    train_data = full_train_df.iloc[train_idx].reset_index(drop=True)
    val_data = full_train_df.iloc[val_idx].reset_index(drop=True)

    print(
        f"Fold {fold}: Train samples: {len(train_data)}, Val samples: {len(val_data)}"
    )

    # Create Datasets
    train_dataset = ToxicityDataset(
        texts=train_data["comment_text"].values,
        tokenizer=tokenizer,
        labels=train_data[label_cols].values,
        max_len=Config.max_len,
    )

    val_dataset = ToxicityDataset(
        texts=val_data["comment_text"].values,
        tokenizer=tokenizer,
        labels=val_data[label_cols].values,
        max_len=Config.max_len,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_test_loader(debug=False, load_cached_data=True):
    """
    Prepares DataLoader for the test set.
    Returns loader and array of IDs.
    """
    tokenizer = get_tokenizer()
    _, test_df = _load_and_merge_data(load_cached_data=load_cached_data)

    if debug:
        print(
            f"Debug mode: Reducing Test dataset from {len(test_df)} to {Config.debug_subset_size}"
        )
        test_df = test_df.iloc[: Config.debug_subset_size].reset_index(drop=True)

    dataset = ToxicityDataset(
        texts=test_df["comment_text"].values,
        tokenizer=tokenizer,
        labels=None,  # No labels for test
        max_len=Config.max_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return loader, test_df["id"].values
