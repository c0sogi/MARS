import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from library.config import Config
from library.utils import seed_everything


class ToxicityDataset(Dataset):
    """
    Dataset class for Toxicity Classification.
    Handles tokenization and returns inputs, targets, and sample weights.
    """

    def __init__(self, df, tokenizer, config, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.is_test = is_test

        # Pre-extract data to numpy arrays for faster access in __getitem__
        self.texts = df["comment_text"].fillna("").values.astype(str)
        self.ids = df["id"].values

        if not self.is_test:
            self.targets = df[config.target_col].values.astype(np.float32)
            # Auxiliary identity targets
            self.aux_targets = (
                df[config.aux_identity_cols].fillna(0.0).values.astype(np.float32)
            )
            # Auxiliary attack target (if present, else 0)
            if config.aux_attack_col in df.columns:
                self.attack_targets = (
                    df[config.aux_attack_col].fillna(0.0).values.astype(np.float32)
                )
            else:
                self.attack_targets = np.zeros(len(df), dtype=np.float32)

            # Sample weights for the loss function
            if "loss_weight" in df.columns:
                self.sample_weights = df["loss_weight"].values.astype(np.float32)
            else:
                self.sample_weights = np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        # We do not pad here; we pad in collate_fn to save compute
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.config.max_len,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoding["attention_mask"], dtype=torch.long
            ),
            "id": self.ids[idx],
        }

        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float)
            item["aux_targets"] = torch.tensor(self.aux_targets[idx], dtype=torch.float)
            item["attack_target"] = torch.tensor(
                self.attack_targets[idx], dtype=torch.float
            )
            item["sample_weight"] = torch.tensor(
                self.sample_weights[idx], dtype=torch.float
            )

        return item


def collate_fn(batch):
    """
    Dynamic padding for batches.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    ids = [item["id"] for item in batch]

    # Pad sequences
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )

    batch_out = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "id": torch.tensor(ids, dtype=torch.long),
    }

    if "target" in batch[0]:
        targets = torch.stack([item["target"] for item in batch])
        aux_targets = torch.stack([item["aux_targets"] for item in batch])
        attack_targets = torch.stack([item["attack_target"] for item in batch])
        sample_weights = torch.stack([item["sample_weight"] for item in batch])

        batch_out["target"] = targets
        batch_out["aux_targets"] = aux_targets
        batch_out["attack_target"] = attack_targets
        batch_out["sample_weight"] = sample_weights

    return batch_out


def calculate_weights(df, identity_cols):
    """
    Calculates sample weights for Loss and sampler weights for WeightedRandomSampler.
    Focuses on 'Bias Trap' examples.
    """
    # Ensure binary target
    if "binary_target" not in df.columns:
        df["binary_target"] = (df["target"] >= 0.5).astype(int)

    # Check if any identity is mentioned (threshold 0.5)
    # We use fillna(0) to treat NaNs as no identity mentioned
    df["identity_mentioned"] = (
        (df[identity_cols].fillna(0.0) >= 0.5).any(axis=1).astype(int)
    )

    # Define subgroups
    # 1. Toxic + Identity Mention (BNSP trap)
    toxic_ident = (df["binary_target"] == 1) & (df["identity_mentioned"] == 1)
    # 2. Non-Toxic + Identity Mention (BPSN trap)
    nontoxic_ident = (df["binary_target"] == 0) & (df["identity_mentioned"] == 1)
    # 3. Toxic + No Identity
    toxic_no_ident = (df["binary_target"] == 1) & (df["identity_mentioned"] == 0)
    # 4. Non-Toxic + No Identity (Background)
    nontoxic_no_ident = (df["binary_target"] == 0) & (df["identity_mentioned"] == 0)

    # --- Loss Weights (sample_weights) ---
    # We want the model to pay more attention to the traps during backprop
    weights = np.ones(len(df))
    weights[toxic_ident] = 5.0
    weights[nontoxic_ident] = 5.0
    weights[toxic_no_ident] = (
        1.0  # Keep standard toxic weight normal, or slightly elevated if recall is low
    )
    weights[nontoxic_no_ident] = 1.0

    df["loss_weight"] = weights

    # --- Sampler Weights (for WeightedRandomSampler) ---
    # We want to oversample the traps in the batch construction
    # Count frequencies
    count_ti = toxic_ident.sum()
    count_nti = nontoxic_ident.sum()
    count_tni = toxic_no_ident.sum()
    count_ntni = nontoxic_no_ident.sum()

    total = len(df)

    # Desired distribution (heuristic)
    # We want to see more identity examples than their natural prevalence
    # Let's assign weights inversely proportional to frequency, smoothed
    sampler_w = np.zeros(len(df))

    if count_ti > 0:
        sampler_w[toxic_ident] = 1.0 / count_ti
    if count_nti > 0:
        sampler_w[nontoxic_ident] = 1.0 / count_nti
    if count_tni > 0:
        sampler_w[toxic_no_ident] = (
            0.5 / count_tni
        )  # Downweight slightly relative to traps
    if count_ntni > 0:
        sampler_w[nontoxic_no_ident] = (
            0.1 / count_ntni
        )  # Significantly downweight background negatives

    # Normalize (optional, but good for debugging)
    sampler_w = sampler_w / sampler_w.sum()

    df["sampler_weight"] = sampler_w

    return df


def load_and_process_data(config, mode="train", load_cached_data=True):
    """
    Loads data from metadata, calculates weights, and caches the result.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    if mode == "train":
        filename = "train_processed.parquet"
        input_path = config.train_path
    elif mode == "val":
        filename = "val_processed.parquet"
        input_path = config.val_path
    elif mode == "test":
        filename = "test_processed.parquet"
        input_path = config.test_path
    else:
        raise ValueError("Invalid mode")

    cache_path = os.path.join(config.output_dir, filename)

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached {mode} data from {cache_path}")
        try:
            df = pd.read_parquet(cache_path)
            # If debug mode, slice it
            if config.debug and mode == "train":
                df = df.iloc[: config.train_subset_size]
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    # 2. Compute from scratch
    print(f"Processing {mode} data from {input_path}")
    df = pd.read_csv(input_path)

    # Handle missing text
    df["comment_text"] = df["comment_text"].fillna("")

    # Calculate weights only for training data
    if mode == "train":
        df = calculate_weights(df, config.aux_identity_cols)

    # Save to cache
    print(f"Saving processed {mode} data to {cache_path}")
    df.to_parquet(cache_path, index=False)

    # Debug slice
    if config.debug and mode == "train":
        df = df.iloc[: config.train_subset_size]

    return df


def get_dataloaders(config, load_cached_data=True):
    """
    Creates DataLoaders for train, validation, and test sets.
    """
    seed_everything(config.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Load Data
    train_df = load_and_process_data(
        config, mode="train", load_cached_data=load_cached_data
    )
    val_df = load_and_process_data(
        config, mode="val", load_cached_data=load_cached_data
    )
    test_df = load_and_process_data(
        config, mode="test", load_cached_data=load_cached_data
    )

    # Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, config, is_test=False)
    val_dataset = ToxicityDataset(val_df, tokenizer, config, is_test=False)
    test_dataset = ToxicityDataset(test_df, tokenizer, config, is_test=True)

    # Sampler for Training
    # We use WeightedRandomSampler to enforce the bias-centric distribution
    if "sampler_weight" in train_df.columns:
        samples_weights = torch.from_numpy(train_df["sampler_weight"].values)
        sampler = WeightedRandomSampler(
            weights=samples_weights, num_samples=len(samples_weights), replacement=True
        )
        shuffle = False  # mutually exclusive with sampler
    else:
        sampler = None
        shuffle = True

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
