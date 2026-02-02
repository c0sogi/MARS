import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from library.config import Config


class ToxicDataset(Dataset):
    """
    Dataset for Supervised Training and Inference.
    Handles tokenization and target processing (binary or soft labels).
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not self.is_test:
            # Select target columns
            self.labels = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
        )

        ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)

        if self.is_test:
            return {"input_ids": ids, "attention_mask": mask}
        else:
            # Convert to float to handle both binary (0/1) and soft labels (probs)
            targets = torch.tensor(self.labels[index], dtype=torch.float)
            return {"input_ids": ids, "attention_mask": mask, "labels": targets}


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pre-training (Masked Language Modeling).
    """

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = df["comment_text"].values.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
        }


def _load_raw_data(meta_file, input_dir):
    """
    Helper to load data by merging metadata with raw source files.
    """
    meta_path = os.path.join(Config.metadata_dir, meta_file)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Identify unique source files referenced in metadata
    sources = meta_df["source_file"].unique()
    dfs = []

    for src in sources:
        src_path = os.path.join(input_dir, src)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        raw_df = pd.read_csv(src_path)

        # Filter metadata for this source
        subset_meta = meta_df[meta_df["source_file"] == src]

        # Merge to get text content. Inner join on ID ensures we only get rows in metadata.
        # raw_df contains 'id' and 'comment_text'
        merged = pd.merge(
            subset_meta, raw_df[["id", "comment_text"]], on="id", how="left"
        )
        dfs.append(merged)

    final_df = pd.concat(dfs, ignore_index=True)

    # Handle missing text
    final_df["comment_text"] = final_df["comment_text"].fillna("missing")

    return final_df


def get_data(load_cached_data=True):
    """
    Loads training, validation, and test data.
    Uses caching (parquet) to speed up subsequent runs.
    """
    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # --- Train Data ---
    if load_cached_data and os.path.exists(Config.train_cache_path):
        train_df = pd.read_parquet(Config.train_cache_path)
    else:
        train_df = _load_raw_data("train.csv", Config.input_dir)
        train_df.to_parquet(Config.train_cache_path)

    # --- Val Data ---
    if load_cached_data and os.path.exists(Config.val_cache_path):
        val_df = pd.read_parquet(Config.val_cache_path)
    else:
        val_df = _load_raw_data("val.csv", Config.input_dir)
        val_df.to_parquet(Config.val_cache_path)

    # --- Test Data ---
    if load_cached_data and os.path.exists(Config.test_cache_path):
        test_df = pd.read_parquet(Config.test_cache_path)
    else:
        test_df = _load_raw_data("test.csv", Config.input_dir)
        test_df.to_parquet(Config.test_cache_path)

    return train_df, val_df, test_df


def prepare_loaders(
    stage="supervised", custom_df=None, load_cached_data=True, debug=Config.debug
):
    """
    Prepares DataLoaders for different stages of the pipeline.

    Args:
        stage (str): 'dapt', 'supervised', 'distillation', or 'test'.
        custom_df (pd.DataFrame): Optional dataframe for distillation (soft labels).
        load_cached_data (bool): Whether to use cached parquet files.
        debug (bool): If True, subsamples data for faster debugging.

    Returns:
        DataLoader(s): Depending on the stage.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Load standard splits
    train_df, val_df, test_df = get_data(load_cached_data=load_cached_data)

    # Debugging subsample
    if debug:
        train_df = train_df.head(200)
        val_df = val_df.head(100)
        test_df = test_df.head(100)
        if custom_df is not None:
            custom_df = custom_df.head(200)

    if stage == "dapt":
        # Domain-Adaptive Pre-training: Combine all available text
        # We only need the text column
        dapt_texts = pd.concat(
            [
                train_df[["comment_text"]],
                val_df[["comment_text"]],
                test_df[["comment_text"]],
            ],
            axis=0,
        ).reset_index(drop=True)

        dataset = MLMDataset(dapt_texts, tokenizer, Config.max_len)

        # Use HuggingFace's DataCollator for dynamic masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        loader = DataLoader(
            dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            collate_fn=data_collator,
            pin_memory=True,
        )
        return loader

    elif stage == "supervised":
        # Standard Supervised Learning (Teacher)
        train_dataset = ToxicDataset(train_df, tokenizer, Config.max_len)
        val_dataset = ToxicDataset(val_df, tokenizer, Config.max_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    elif stage == "distillation":
        # Student Self-Distillation
        # Requires custom_df containing combined train + soft-labeled test data
        if custom_df is None:
            raise ValueError("custom_df must be provided for distillation stage")

        train_dataset = ToxicDataset(custom_df, tokenizer, Config.max_len)
        # Validate on the original validation set to track real performance
        val_dataset = ToxicDataset(val_df, tokenizer, Config.max_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.train_batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    elif stage == "test":
        # Inference on Test Set
        test_dataset = ToxicDataset(test_df, tokenizer, Config.max_len, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.valid_batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
        )
        return test_loader

    else:
        raise ValueError(f"Unknown stage: {stage}")
