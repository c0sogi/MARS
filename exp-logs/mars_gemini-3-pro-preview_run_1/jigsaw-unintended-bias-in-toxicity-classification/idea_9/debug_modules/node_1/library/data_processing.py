import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from library.config import CFG
from library.utils import get_logger

# Initialize logger
logger = get_logger()


def get_tokenizer():
    """
    Loads the tokenizer defined in CFG.
    """
    return AutoTokenizer.from_pretrained(CFG.model_name)


def _preprocess_df(df, is_test=False):
    """
    Internal function to process dataframe: handle NaNs, generate weights, and aux labels.
    """
    # Fill missing text
    df["comment_text"] = df["comment_text"].fillna("")

    if is_test:
        return df

    # Fill identity columns and identity_attack with 0.0
    cols_to_fill = CFG.identity_cols + ["identity_attack"]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            # Should not happen given metadata, but safe fallback
            df[col] = 0.0

    # --- Generate Bias-Centric Sample Weights ---
    # Define conditions
    # Toxic: target >= 0.5
    # Identity Mention: value >= 0.5 (standard threshold for binary mention)

    is_toxic = df["target"] >= 0.5

    # Check if any identity is mentioned
    identity_values = df[CFG.identity_cols].values
    has_identity = (identity_values >= 0.5).any(axis=1)

    # Initialize weights
    weights = np.full(len(df), CFG.normal_weight, dtype=np.float32)

    # Condition 1: Toxic + Identity Mention (Subgroup Positive)
    # These are toxic comments mentioning an identity.
    # Bias: Model might predict LOW toxicity if it over-corrects (BNSP).
    mask_pos_bias = is_toxic & has_identity
    weights[mask_pos_bias] = CFG.bias_pos_weight

    # Condition 2: Non-Toxic + Identity Mention (Background Positive / Subgroup Negative)
    # These are neutral comments mentioning an identity.
    # Bias: Model might predict HIGH toxicity due to identity association (BPSN).
    mask_neg_bias = (~is_toxic) & has_identity
    weights[mask_neg_bias] = CFG.bias_neg_weight

    df["loss_weight"] = weights

    return df


def get_data(load_cached_data=True, debug=None, debug_size=None):
    """
    Loads train, val, and test data. Implements caching to parquet.
    """
    if debug is None:
        debug = CFG.debug
    if debug_size is None:
        debug_size = CFG.debug_sample_size

    # Define cache paths
    cache_train = os.path.join(CFG.cache_dir, "train_processed.parquet")
    cache_val = os.path.join(CFG.cache_dir, "val_processed.parquet")
    cache_test = os.path.join(CFG.cache_dir, "test_processed.parquet")

    # Attempt to load cache
    if load_cached_data:
        if (
            os.path.exists(cache_train)
            and os.path.exists(cache_val)
            and os.path.exists(cache_test)
        ):
            logger.info("Loading processed data from cache...")
            train_df = pd.read_parquet(cache_train)
            val_df = pd.read_parquet(cache_val)
            test_df = pd.read_parquet(cache_test)

            if debug:
                logger.info(f"Debug mode: Sampling {debug_size} rows.")
                train_df = train_df.iloc[:debug_size]
                val_df = val_df.iloc[:debug_size]
                test_df = test_df.iloc[:debug_size]

            return train_df, val_df, test_df
        else:
            logger.info("Cache not found. Processing from scratch...")
    else:
        logger.info("Ignoring cache. Processing from scratch...")

    # Load raw metadata
    logger.info(f"Loading raw data from {CFG.metadata_dir}...")
    train_df = pd.read_csv(CFG.train_path)
    val_df = pd.read_csv(CFG.val_path)
    test_df = pd.read_csv(CFG.test_path)

    # Process
    logger.info("Preprocessing DataFrames...")
    train_df = _preprocess_df(train_df, is_test=False)
    val_df = _preprocess_df(val_df, is_test=False)
    test_df = _preprocess_df(test_df, is_test=True)

    # Save to cache
    logger.info(f"Saving processed data to {CFG.cache_dir}...")
    train_df.to_parquet(cache_train, index=False)
    val_df.to_parquet(cache_val, index=False)
    test_df.to_parquet(cache_test, index=False)

    # Debug sampling (applied after saving cache to preserve full cache)
    if debug:
        logger.info(f"Debug mode: Sampling {debug_size} rows.")
        train_df = train_df.iloc[:debug_size]
        val_df = val_df.iloc[:debug_size]
        test_df = test_df.iloc[:debug_size]

    return train_df, val_df, test_df


class JigsawDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    Returns: input_ids, attention_mask, target, aux_labels, loss_weight
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test
        self.texts = df["comment_text"].values

        if not is_test:
            self.targets = df["target"].values
            self.weights = df["loss_weight"].values

            # Prepare Aux Labels: [Identity Cols (9) + Identity Attack (1)]
            # We treat them as binary or continuous. Task says attributes are fractional.
            # We will pass the fractional values directly for BCE (soft labels).
            self.identity_vals = df[CFG.identity_cols].values
            self.identity_attack_vals = df["identity_attack"].values.reshape(-1, 1)
            self.aux_labels = np.hstack([self.identity_vals, self.identity_attack_vals])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

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

        target = torch.tensor(self.targets[idx], dtype=torch.float)
        aux_labels = torch.tensor(self.aux_labels[idx], dtype=torch.float)
        weight = torch.tensor(self.weights[idx], dtype=torch.float)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "target": target,
            "aux_labels": aux_labels,
            "loss_weight": weight,
        }


class MLMDataset(Dataset):
    """
    Dataset for Domain-Adaptive Pretraining (Masked Language Modeling).
    Uses combined text from Train and Test.
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
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=False,
        )

        return {
            "input_ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            # Labels for MLM are typically handled by DataCollatorForLanguageModeling
            # which masks input_ids dynamically. We return the clean inputs here.
        }


def get_loaders(train_df, val_df, tokenizer):
    """
    Creates DataLoaders for training and validation.
    Uses WeightedRandomSampler for training to oversample bias traps.
    """
    train_dataset = JigsawDataset(train_df, tokenizer, CFG.max_len, is_test=False)
    val_dataset = JigsawDataset(val_df, tokenizer, CFG.max_len, is_test=False)

    # --- Weighted Sampler for Training ---
    # We use the calculated loss weights as sampling weights.
    # This ensures batches contain a high density of "Bias Trap" examples.
    samples_weights = torch.tensor(train_df["loss_weight"].values, dtype=torch.double)
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        sampler=sampler,  # Using sampler, so shuffle must be False
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_mlm_loader(train_df, test_df, tokenizer):
    """
    Creates a DataLoader for Domain Adaptation (MLM) using combined train and test text.
    """
    # Combine texts
    all_texts = np.concatenate(
        [
            train_df["comment_text"].fillna("").values,
            test_df["comment_text"].fillna("").values,
        ]
    )

    mlm_dataset = MLMDataset(all_texts, tokenizer, CFG.max_len)

    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=CFG.mlm_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return mlm_loader


def get_test_loader(test_df, tokenizer):
    """
    Creates DataLoader for inference on test set.
    """
    test_dataset = JigsawDataset(test_df, tokenizer, CFG.max_len, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
