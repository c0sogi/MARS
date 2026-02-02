import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from library.config import CFG


def preprocess_data(load_cached_data=True):
    """
    Loads metadata, computes necessary weights and targets, and caches the result.

    Args:
        load_cached_data (bool): If True, attempts to load from cache first.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    cache_train_path = os.path.join(CFG.cache_dir, "train_processed.parquet")
    cache_val_path = os.path.join(CFG.cache_dir, "val_processed.parquet")
    cache_test_path = os.path.join(CFG.cache_dir, "test_processed.parquet")

    # 1. Try Loading from Cache
    if load_cached_data:
        if (
            os.path.exists(cache_train_path)
            and os.path.exists(cache_val_path)
            and os.path.exists(cache_test_path)
        ):
            print("Loading processed data from cache...")
            train_df = pd.read_parquet(cache_train_path)
            val_df = pd.read_parquet(cache_val_path)
            test_df = pd.read_parquet(cache_test_path)
            return train_df, val_df, test_df

    # 2. Process from Scratch
    print("Processing data from metadata...")

    # Load Metadata
    train_df = pd.read_csv(CFG.train_path)
    val_df = pd.read_csv(CFG.val_path)
    test_df = pd.read_csv(CFG.test_path)

    # --- Feature Engineering for Training ---

    # Ensure binary target
    if CFG.binary_target_col not in train_df.columns:
        train_df[CFG.binary_target_col] = (train_df[CFG.target_col] >= 0.5).astype(int)

    # Identify if any identity is mentioned (threshold 0.5)
    # Fill NaNs with 0.0 for calculation
    identity_sub = train_df[CFG.identity_cols].fillna(0.0)
    train_df["identity_present"] = (identity_sub.max(axis=1) >= 0.5).astype(int)

    # Define Bias Traps for Weighting
    # Trap 1: Non-Toxic + Identity (False Positive Risk)
    # Trap 2: Toxic + Identity (Differentiation Hardness)
    # We want to upweight these in the loss and the sampler

    is_toxic = train_df[CFG.binary_target_col] == 1
    has_identity = train_df["identity_present"] == 1

    # Logic:
    # Bias Trap = (Non-Toxic AND Identity) OR (Toxic AND Identity)
    # Note: Toxic+NoIdentity and NonToxic+NoIdentity are "Background"
    bias_trap_mask = has_identity  # Covers both Toxic and Non-Toxic with identity

    # Assign Loss Weights (used in JigsawLoss)
    train_df["loss_weight"] = 1.0
    train_df.loc[bias_trap_mask, "loss_weight"] = CFG.bias_sample_weight

    # Assign Sampler Weights (used in WeightedRandomSampler for Stage 3)
    # We want to aggressively sample bias traps.
    # Base weight 1.0, Trap weight 5.0 (or higher if needed, using config)
    train_df["sampler_weight"] = 1.0
    train_df.loc[bias_trap_mask, "sampler_weight"] = CFG.bias_sample_weight

    # --- Validation Processing ---
    # Validation needs identity columns for metrics, but doesn't need weights
    if CFG.binary_target_col not in val_df.columns:
        val_df[CFG.binary_target_col] = (val_df[CFG.target_col] >= 0.5).astype(int)

    # --- Cache Results ---
    print(f"Saving processed data to {CFG.cache_dir}...")
    train_df.to_parquet(cache_train_path)
    val_df.to_parquet(cache_val_path)
    test_df.to_parquet(cache_test_path)

    return train_df, val_df, test_df


class JigsawDataset(Dataset):
    """
    Dataset for Toxicity Classification (Stage 2 & 3).
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].fillna("").values.astype(str)

        if not self.is_test:
            self.targets = df[CFG.target_col].values
            self.identities = df[CFG.identity_cols].fillna(0.0).values
            self.attacks = df[CFG.aux_attack_col].fillna(0.0).values
            # Default to 1.0 if loss_weight not computed (e.g. validation)
            self.weights = df.get("loss_weight", pd.Series(np.ones(len(df)))).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0)

        if self.is_test:
            return {
                "input_ids": ids,
                "attention_mask": mask,
            }

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "target": torch.tensor(self.targets[idx], dtype=torch.float),
            "identities": torch.tensor(self.identities[idx], dtype=torch.float),
            "attack": torch.tensor(self.attacks[idx], dtype=torch.float),
            "sample_weights": torch.tensor(self.weights[idx], dtype=torch.float),
        }


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pretraining (Stage 1).
    Concatenates Train and Test text.
    """

    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            # Labels are handled by DataCollatorForLanguageModeling
        }


def get_loaders(stage, tokenizer, load_cached_data=True):
    """
    Factory function to get DataLoaders based on the training stage.

    Args:
        stage (str): 'dapt', 'general', or 'robust'.
        tokenizer: Transformers tokenizer.
        load_cached_data (bool): Whether to use cached dataframes.

    Returns:
        tuple: (train_loader, val_loader) or (train_loader, None) for DAPT.
    """
    # Load Dataframes
    train_df, val_df, test_df = preprocess_data(load_cached_data=load_cached_data)

    # --- Stage 1: Domain-Adaptive Pretraining (MLM) ---
    if stage == "dapt":
        print("Preparing DAPT DataLoader (Train + Test text)...")
        # Combine texts
        all_texts = np.concatenate(
            [
                train_df["comment_text"].fillna("").values,
                test_df["comment_text"].fillna("").values,
            ]
        )

        train_dataset = MLMDataset(all_texts, tokenizer, CFG.max_len)

        # MLM Collator handles masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=CFG.mlm_probability
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            collate_fn=data_collator,
        )

        return train_loader, None

    # --- Stage 2 & 3: Classification ---

    # Validation Loader (Common for both stages)
    val_dataset = JigsawDataset(val_df, tokenizer, CFG.max_len)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    train_dataset = JigsawDataset(train_df, tokenizer, CFG.max_len)

    if stage == "general":
        print("Preparing General Fine-Tuning DataLoader (Uniform Stratified)...")
        # Uniform Stratified: We rely on the natural distribution (shuffle=True)
        # The loss function handles the weighting.
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    elif stage == "robust":
        print("Preparing Robust Optimization DataLoader (Bias-Weighted Sampling)...")
        # Robust: We use WeightedRandomSampler to force bias traps into batches
        sampler_weights = train_df["sampler_weight"].values
        sampler = WeightedRandomSampler(
            weights=sampler_weights, num_samples=len(train_df), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            sampler=sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return train_loader, val_loader


def get_test_loader(tokenizer, load_cached_data=True):
    """
    Returns DataLoader for the test set.
    """
    _, _, test_df = preprocess_data(load_cached_data=load_cached_data)

    test_dataset = JigsawDataset(test_df, tokenizer, CFG.max_len, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    return test_loader, test_df["id"].values
