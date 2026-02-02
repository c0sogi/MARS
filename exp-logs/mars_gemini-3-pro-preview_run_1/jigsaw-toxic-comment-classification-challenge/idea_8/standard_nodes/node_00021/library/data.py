import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from sklearn.model_selection import StratifiedKFold, KFold

# Import configuration and utilities from provided library files
from library.config import Config
from library.utils import seed_everything


def get_tokenizer():
    """
    Loads the tokenizer defined in the Config.
    """
    return AutoTokenizer.from_pretrained(Config.model_name)


class ToxicDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    Returns input_ids, attention_mask, and labels (if available).
    """

    def __init__(self, df, tokenizer, max_len=Config.max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not self.is_test:
            self.labels = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        item = {"input_ids": input_ids, "attention_mask": attention_mask}

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (Masked Language Modeling).
    Takes a list of texts and returns tokenized inputs.
    Masking is handled by the DataCollator during the forward pass in the trainer/loop,
    but here we prepare the inputs.
    """

    def __init__(self, texts, tokenizer, max_len=Config.max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])

        encoding = self.tokenizer.encode_plus(
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


def load_dataset_from_metadata(
    meta_path, source_csv, load_cached_data=True, cache_name="data"
):
    """
    Loads dataset by merging metadata with raw source files.
    Implements caching using parquet.
    """
    cache_path = os.path.join(Config.working_dir, f"{cache_name}.parquet")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        # print(f"Loading cached data from {cache_path}")
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to processing if cache is corrupt

    # 2. Process from scratch
    # print(f"Processing data for {cache_name}...")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file {meta_path} not found.")

    meta_df = pd.read_csv(meta_path)

    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source file {source_csv} not found.")

    src_df = pd.read_csv(source_csv)

    # Merge to get text content
    # We use inner join on ID
    merged = pd.merge(meta_df, src_df[["id", "comment_text"]], on="id", how="left")
    merged["comment_text"] = merged["comment_text"].fillna("")

    # 3. Save to cache
    os.makedirs(Config.working_dir, exist_ok=True)
    merged.to_parquet(cache_path, index=False)

    return merged


def get_data(load_cached_data=True):
    """
    Loads all necessary data for the pipeline.
    Combines train and validation metadata to form the full training set for CV.
    Loads test data.
    """
    # Load Train Metadata and Val Metadata, merge with raw train.csv
    # We combine them because we are doing 5-Fold CV on the full dataset

    # We create a specific cache for the combined full train set
    full_train_cache = os.path.join(Config.working_dir, "full_train_combined.parquet")

    if load_cached_data and os.path.exists(full_train_cache):
        train_df = pd.read_parquet(full_train_cache)
    else:
        # Load partials
        df_train_part = load_dataset_from_metadata(
            Config.train_meta_path,
            Config.train_raw_path,
            load_cached_data=load_cached_data,
            cache_name="train_split_only",
        )
        df_val_part = load_dataset_from_metadata(
            Config.val_meta_path,
            Config.train_raw_path,
            load_cached_data=load_cached_data,
            cache_name="val_split_only",
        )

        # Concatenate
        train_df = pd.concat([df_train_part, df_val_part], ignore_index=True)

        # Cache full set
        train_df.to_parquet(full_train_cache, index=False)

    # Load Test Data
    test_df = load_dataset_from_metadata(
        Config.test_meta_path,
        Config.test_raw_path,
        load_cached_data=load_cached_data,
        cache_name="test_full",
    )

    # Debugging: Subsample if configured
    if Config.debug:
        train_df = train_df.sample(
            n=min(len(train_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)
        test_df = test_df.sample(
            n=min(len(test_df), Config.debug_sample_size), random_state=Config.seed
        ).reset_index(drop=True)

    return train_df, test_df


def prepare_loaders(fold, train_df, tokenizer, debug=False):
    """
    Splits the training data into train/val for the specific fold and returns DataLoaders.
    """
    # Create Folds
    # We use MultilabelStratifiedKFold logic or simple KFold if dependencies are tricky.
    # Given the environment, we'll use a seeded KFold which is robust enough for this task
    # if stratification is complex to implement without 'iterstrat'.
    # However, we can attempt a simple stratification based on the 'stratify_group' logic if present,
    # or just use KFold.

    kf = KFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)

    # Generate indices
    splits = list(kf.split(X=train_df, y=train_df[Config.target_cols]))
    train_idx, val_idx = splits[fold]

    train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)

    # Create Datasets
    train_dataset = ToxicDataset(train_fold_df, tokenizer)
    val_dataset = ToxicDataset(val_fold_df, tokenizer)

    # Create DataLoaders
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


def prepare_mlm_loader(tokenizer, load_cached_data=True):
    """
    Prepares the DataLoader for Domain-Adaptive Pre-training.
    Combines text from train and test sets.
    """
    train_df, test_df = get_data(load_cached_data=load_cached_data)

    # Concatenate texts
    all_texts = np.concatenate(
        [train_df["comment_text"].values, test_df["comment_text"].values]
    )

    # Create Dataset
    mlm_dataset = MLMDataset(all_texts, tokenizer)

    # Data Collator for masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=Config.mlm_probability
    )

    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=Config.mlm_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    return mlm_loader


def prepare_test_loader(test_df, tokenizer):
    """
    Prepares DataLoader for inference on the test set.
    """
    test_dataset = ToxicDataset(test_df, tokenizer, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size
        * 2,  # Can usually handle larger batch size for inference
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
