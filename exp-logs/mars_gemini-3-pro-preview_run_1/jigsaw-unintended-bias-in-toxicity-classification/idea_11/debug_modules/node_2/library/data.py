import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from library.config import Config
from library.utils import set_seed


class ToxicityDataset(Dataset):
    """
    PyTorch Dataset for Toxicity Classification.
    Handles tokenization and retrieval of targets (toxicity, identities, subtypes) and sample weights.
    """

    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        # Pre-extract columns to numpy for faster access
        self.texts = self.df[Config.text_col].astype(str).values

        if not self.is_test:
            self.targets = self.df[Config.target_col].values
            self.binary_targets = self.df[Config.binary_target_col].values

            # Identity targets (multi-label)
            # Fill NaNs with 0.0 (assuming NaN means identity not mentioned)
            self.identity_targets = (
                self.df[Config.identity_cols].fillna(0.0).values.astype(np.float32)
            )

            # Auxiliary targets (subtypes)
            # Fill NaNs with 0.0
            self.aux_targets = (
                self.df[Config.aux_cols].fillna(0.0).values.astype(np.float32)
            )

            # Sample weights (if pre-calculated)
            if "weight" in self.df.columns:
                self.weights = self.df["weight"].values.astype(np.float32)
            else:
                self.weights = np.ones(len(self.df), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.texts[idx]

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

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
            item["binary_target"] = torch.tensor(
                self.binary_targets[idx], dtype=torch.float
            )
            item["identity_target"] = torch.tensor(
                self.identity_targets[idx], dtype=torch.float
            )
            item["aux_target"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)
            item["weight"] = torch.tensor(self.weights[idx], dtype=torch.float)

        return item


class MLMDataset(Dataset):
    """
    Dataset for Masked Language Modeling (Domain-Adaptive Pretraining).
    Returns input_ids and attention_mask. Masking is typically handled by DataCollator.
    """

    def __init__(self, texts, tokenizer, max_len):
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


class DataMiner:
    """
    Utility to mine 'Hard Negative' examples from unlabeled data (Test set)
    using predictions from a 'Scout' model.
    """

    def __init__(self):
        self.identity_cols = Config.identity_cols
        self.mining_threshold_identity = Config.mining_threshold_identity
        self.mining_threshold_toxicity = Config.mining_threshold_toxicity

    def augment_training_data(
        self, original_train_df, scout_preds_df, scout_identity_preds
    ):
        """
        Identifies bias traps in scout predictions and merges them into the training set.

        Args:
            original_train_df (pd.DataFrame): The original labeled training data.
            scout_preds_df (pd.DataFrame): Dataframe containing 'id' and 'comment_text' of the mining pool.
                                           Must be aligned with predictions.
            scout_identity_preds (np.ndarray): Predictions for identity columns (N_samples, N_identities).

        Returns:
            pd.DataFrame: The augmented training dataframe.
        """
        # scout_preds_df is expected to have 'toxicity_pred' column or similar,
        # but usually we pass the dataframe and separate arrays.
        # Here we assume scout_preds_df has 'prediction' (toxicity) and metadata.

        print("Mining hard negatives (Bias Traps)...")

        # Ensure alignment
        if len(scout_preds_df) != len(scout_identity_preds):
            raise ValueError("Scout predictions length mismatch.")

        # 1. Identify Hard Negatives: High Identity Prob AND Low Toxicity Prob
        # We look for ANY identity > threshold
        max_identity_prob = scout_identity_preds.max(axis=1)
        toxicity_prob = scout_preds_df["prediction"].values

        mask_hard_negative = (max_identity_prob > self.mining_threshold_identity) & (
            toxicity_prob < self.mining_threshold_toxicity
        )

        mined_df = scout_preds_df[mask_hard_negative].copy()
        mined_identities = scout_identity_preds[mask_hard_negative]

        print(
            f"Found {len(mined_df)} hard negative candidates out of {len(scout_preds_df)}."
        )

        if len(mined_df) == 0:
            return original_train_df

        # 2. Create Pseudo-labels
        # Target = 0 (Non-toxic)
        mined_df[Config.target_col] = 0.0
        mined_df[Config.binary_target_col] = 0

        # Identity Targets = 1 where prob > threshold (Pseudo-labeling identities)
        # We populate the specific identity columns
        for i, col in enumerate(self.identity_cols):
            # If the model was confident it's this identity, set to 1.0
            mined_df[col] = (
                mined_identities[:, i] > self.mining_threshold_identity
            ).astype(float)

        # Aux Targets = 0 (Assume no specific subtypes for non-toxic)
        for col in Config.aux_cols:
            mined_df[col] = 0.0

        # 3. Assign High Weights to these mined samples
        # They are explicitly hard negatives.
        mined_df["weight"] = 5.0

        # 4. Merge
        # Ensure columns match
        common_cols = [c for c in original_train_df.columns if c in mined_df.columns]
        augmented_df = pd.concat(
            [original_train_df, mined_df[common_cols]], axis=0, ignore_index=True
        )

        # Fill any missing columns in mined data (that existed in train) with defaults
        augmented_df = augmented_df.fillna(0)

        print(f"Augmented Train Shape: {augmented_df.shape}")
        return augmented_df


def calculate_weights(df):
    """
    Calculates sample weights for the loss function based on bias subgroups.
    """
    weights = np.ones(len(df))

    # Ensure binary target
    if Config.binary_target_col not in df.columns:
        binary_target = (df[Config.target_col] >= 0.5).astype(int)
    else:
        binary_target = df[Config.binary_target_col].astype(int)

    # Check for presence of any identity
    # We use 0.5 as threshold for "mentioned"
    identity_cols = df[Config.identity_cols].fillna(0.0)
    has_identity = (identity_cols.max(axis=1) >= 0.5).astype(int)

    # 1. Upweight Toxic samples (Class Imbalance)
    weights += (binary_target == 1) * 3.0

    # 2. Upweight Identity Mentions (General Bias)
    weights += (has_identity == 1) * 3.0

    # 3. Specific Bias Subgroups (Intersection)
    # Toxic + Identity (BNSP risk)
    weights += ((binary_target == 1) & (has_identity == 1)) * 10.0

    # Non-Toxic + Identity (BPSN risk) - These are the most critical for False Positives
    weights += ((binary_target == 0) & (has_identity == 1)) * 10.0

    return weights


def load_data(mode="train", load_cached_data=True):
    """
    Loads and preprocesses data. Handles caching.

    Args:
        mode (str): 'train', 'val', 'test', or 'mlm'.
        load_cached_data (bool): Whether to attempt loading from parquet cache.

    Returns:
        pd.DataFrame or list: Processed dataframe or list of texts (for MLM).
    """
    os.makedirs(Config.working_dir, exist_ok=True)

    # Define cache paths
    cache_map = {
        "train": Config.cached_train_path,
        "val": Config.cached_val_path,
        "test": Config.cached_test_path,
        "mlm": None,  # MLM combines train/test, usually dynamic
    }

    cache_path = cache_map.get(mode)

    # Try loading cache
    if load_cached_data and cache_path and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}...")
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-processing.")

    print(f"Processing {mode} data from metadata...")

    if mode == "train":
        df = pd.read_csv(Config.train_path)
        # Calculate weights
        df["weight"] = calculate_weights(df)

        # Cache
        if cache_path:
            df.to_parquet(cache_path, index=False)

    elif mode == "val":
        df = pd.read_csv(Config.val_path)
        # Weights for validation are not strictly necessary for metric,
        # but good if we validate loss.
        df["weight"] = calculate_weights(df)

        if cache_path:
            df.to_parquet(cache_path, index=False)

    elif mode == "test":
        df = pd.read_csv(Config.test_path)
        # Fill missing text
        df[Config.text_col] = df[Config.text_col].fillna("")

        if cache_path:
            df.to_parquet(cache_path, index=False)

    elif mode == "mlm":
        # Combine Train and Test text
        df_train = pd.read_csv(Config.train_path)
        df_test = pd.read_csv(Config.test_path)

        texts = pd.concat(
            [df_train[Config.text_col].fillna(""), df_test[Config.text_col].fillna("")]
        ).unique()

        return texts.tolist()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df


def get_weighted_loader(df, tokenizer, batch_size, is_test=False, num_workers=None):
    """
    Creates a DataLoader. For training, uses Stratified Weighted Sampling
    to oversample minority classes and identity subgroups.
    """
    if num_workers is None:
        num_workers = Config.num_workers

    dataset = ToxicityDataset(df, tokenizer, Config.max_len, is_test=is_test)

    if is_test:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # For training, create weighted sampler
    # We use the 'weight' column we calculated earlier as the sampling weight
    # This ensures high-weight samples (bias subgroups) are seen more frequently.
    samples_weights = torch.tensor(df["weight"].values, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=samples_weights, num_samples=len(samples_weights), replacement=True
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
