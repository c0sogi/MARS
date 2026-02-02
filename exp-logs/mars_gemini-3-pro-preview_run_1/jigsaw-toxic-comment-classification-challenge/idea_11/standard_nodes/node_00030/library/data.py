import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
from library.config import Config

# -----------------------------------------------------------------------------
# Data Loading & Caching Logic
# -----------------------------------------------------------------------------


def load_dataset_from_metadata(
    config: Config, split_name: str, load_cached_data: bool = True
):
    """
    Loads the dataset by merging metadata with raw text files.
    Implements caching using Parquet to speed up subsequent runs.

    Args:
        config: Config object containing paths.
        split_name: 'train', 'val', or 'test'.
        load_cached_data: Whether to try loading from cache.

    Returns:
        pd.DataFrame: The merged dataframe containing 'id', 'comment_text', and labels.
    """
    cache_path = os.path.join(config.cache_dir, f"{split_name}_cache.parquet")

    # 1. Try Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to re-processing if cache is corrupt

    # 2. Process from Scratch
    # Determine metadata file path
    if split_name == "train":
        meta_path = config.train_meta_file
    elif split_name == "val":
        meta_path = config.val_meta_file
    elif split_name == "test":
        meta_path = config.test_meta_file
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    meta_df = pd.read_csv(meta_path)

    # Identify unique source files in metadata to minimize file I/O
    sources = meta_df["source_file"].unique()
    loaded_data = []

    for src in sources:
        src_path = os.path.join(config.input_dir, src)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file {src} missing.")

        # Read raw source file
        src_df = pd.read_csv(src_path)

        # Filter metadata for this source
        subset_meta = meta_df[meta_df["source_file"] == src]

        # Merge to get text content: Inner join on ID
        merged = pd.merge(
            subset_meta, src_df[["id", "comment_text"]], on="id", how="inner"
        )
        loaded_data.append(merged)

    final_df = pd.concat(loaded_data, ignore_index=True)

    # Handle missing text values
    final_df["comment_text"] = final_df["comment_text"].fillna("")

    # 3. Save to Cache
    try:
        os.makedirs(config.cache_dir, exist_ok=True)
        final_df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to cache data: {e}")

    return final_df


# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------


class ToxicityDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len, labels=None):
        """
        Dataset for Supervised (Teacher), Semi-Supervised (Student), and Inference tasks.

        Args:
            texts (list or np.array): List of text strings.
            tokenizer: HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
            labels (list or np.array, optional): Targets (binary or soft probabilities).
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

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

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        result = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.labels is not None:
            # Labels can be binary (long) or soft (float)
            # We convert to float for BCEWithLogitsLoss which handles both
            label_vec = torch.tensor(self.labels[item], dtype=torch.float)
            result["labels"] = label_vec

        return result


class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        """
        Dataset for Masked Language Modeling (DAPT).

        Args:
            texts (list or np.array): List of text strings.
            tokenizer: HuggingFace tokenizer.
            max_len (int): Maximum sequence length.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])

        # Tokenize for MLM
        # We return special_tokens_mask so the collator knows not to mask [CLS], [SEP], etc.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "special_tokens_mask": encoding["special_tokens_mask"].flatten(),
        }


# -----------------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------------


def get_dataloaders(
    config: Config, tokenizer, stage="teacher", fold=0, pseudo_labels_df=None
):
    """
    Creates DataLoaders for the specified training stage.

    Args:
        config: Config object.
        tokenizer: HuggingFace tokenizer.
        stage: 'dapt', 'teacher', 'student', or 'inference'.
        fold: Fold index (reserved for future CV implementation, currently uses fixed metadata split).
        pseudo_labels_df: DataFrame containing pseudo-labels (probabilities) for student training.

    Returns:
        tuple or DataLoader:
            - 'dapt': Returns a single DataLoader.
            - 'teacher'/'student': Returns (train_loader, val_loader).
            - 'inference': Returns (loader, ids).
    """

    # Load Metadata-based DataFrames (using cache if available)
    train_df = load_dataset_from_metadata(config, "train", load_cached_data=True)
    val_df = load_dataset_from_metadata(config, "val", load_cached_data=True)
    test_df = load_dataset_from_metadata(config, "test", load_cached_data=True)

    # Debugging: Subsample datasets if debug mode is active
    if config.debug:
        train_df = train_df.iloc[: config.debug_sample_size]
        val_df = val_df.iloc[: config.debug_sample_size]
        test_df = test_df.iloc[: config.debug_sample_size]
        if pseudo_labels_df is not None:
            pseudo_labels_df = pseudo_labels_df.iloc[: config.debug_sample_size]

    # -------------------------------------------------------------------------
    # Stage 1: DAPT (Domain Adaptive Pre-Training)
    # -------------------------------------------------------------------------
    if stage == "dapt":
        # Combine all available text (Train + Val + Test) for unsupervised learning
        all_texts = pd.concat(
            [train_df["comment_text"], val_df["comment_text"], test_df["comment_text"]]
        ).tolist()

        ds = MLMDataset(all_texts, tokenizer, config.max_len)

        # Use HuggingFace's collator for dynamic masking
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
        )

        loader = DataLoader(
            ds,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
        )
        return loader

    # -------------------------------------------------------------------------
    # Stage 2: Teacher (Supervised Training)
    # -------------------------------------------------------------------------
    elif stage == "teacher":
        # Train on the fixed training split, validate on fixed validation split
        train_texts = train_df["comment_text"].values
        train_labels = train_df[config.target_cols].values

        val_texts = val_df["comment_text"].values
        val_labels = val_df[config.target_cols].values

        train_ds = ToxicityDataset(train_texts, tokenizer, config.max_len, train_labels)
        val_ds = ToxicityDataset(val_texts, tokenizer, config.max_len, val_labels)

        train_loader = DataLoader(
            train_ds,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    # -------------------------------------------------------------------------
    # Stage 3: Student (Semi-Supervised / Self-Training)
    # -------------------------------------------------------------------------
    elif stage == "student":
        if pseudo_labels_df is None:
            raise ValueError("pseudo_labels_df must be provided for student stage.")

        # 1. Original Labeled Data (Hard Labels)
        train_texts_orig = train_df["comment_text"].values
        train_labels_orig = train_df[config.target_cols].values

        # 2. Pseudo-Labeled Test Data (Soft Labels)
        # Merge pseudo labels with test text on 'id' to ensure correct alignment
        test_with_labels = pd.merge(
            test_df[["id", "comment_text"]], pseudo_labels_df, on="id", how="inner"
        )

        test_texts_pseudo = test_with_labels["comment_text"].values
        test_labels_pseudo = test_with_labels[config.target_cols].values

        # Combine Original and Pseudo-Labeled data
        combined_texts = np.concatenate([train_texts_orig, test_texts_pseudo])
        combined_labels = np.concatenate([train_labels_orig, test_labels_pseudo])

        # Create Training Dataset with combined data
        train_ds = ToxicityDataset(
            combined_texts, tokenizer, config.max_len, combined_labels
        )

        # Validation remains the same (clean validation set)
        val_texts = val_df["comment_text"].values
        val_labels = val_df[config.target_cols].values
        val_ds = ToxicityDataset(val_texts, tokenizer, config.max_len, val_labels)

        train_loader = DataLoader(
            train_ds,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    # -------------------------------------------------------------------------
    # Inference (Test Set)
    # -------------------------------------------------------------------------
    elif stage == "inference":
        # Load test data
        texts = test_df["comment_text"].values
        ids = test_df["id"].values

        # No labels for inference
        ds = ToxicityDataset(texts, tokenizer, config.max_len, labels=None)

        loader = DataLoader(
            ds,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        return loader, ids

    else:
        raise ValueError(f"Unknown stage: {stage}")
