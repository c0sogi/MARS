import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from library.config import (
    WORKING_DIR,
    CACHE_DIR,
    SEMIOTIC_REGEX,
    TRAIN_META_PATH,
    VAL_META_PATH,
    ModelConfig,
)
from library.text_utils import TextDataset, format_context_window


class HeterogeneousDataset(TextDataset):
    """
    Dataset wrapper for Tier 2 Transformer.
    Inherits from library.text_utils.TextDataset.

    Ensures that decoder targets are padded to a fixed length for batch processing.
    """

    def __getitem__(self, idx):
        # Get item from parent class (handles char encoding and basic BPE encoding)
        item = super().__getitem__(idx)

        # Pad decoder_target to max_dec_len if present
        if "decoder_target" in item:
            target = item["decoder_target"]
            current_len = len(target)

            if current_len < self.max_dec_len:
                pad_size = self.max_dec_len - current_len
                # 0 is PAD ID for BPE as per text_utils.py configuration
                padding = torch.zeros(pad_size, dtype=torch.long)
                item["decoder_target"] = torch.cat([target, padding])
            elif current_len > self.max_dec_len:
                # Truncation should have happened in TextDataset, but safety check
                item["decoder_target"] = target[: self.max_dec_len]
                # Ensure EOS is at the end if we truncated
                item["decoder_target"][-1] = 3

        return item


def create_density_maximized_dataset(df, config, split="train", load_cached_data=True):
    """
    Constructs the Density-Maximized Class-Balanced dataset.

    1. Generates context windows for ALL tokens.
    2. Filters for Semiotic tokens (digits/latin).
    3. (Train Only) Applies aggressive class balancing (upsampling rare, downsampling dominant).
    """
    # Define cache path
    cache_path = os.path.join(CACHE_DIR, f"density_maximized_{split}.parquet")

    # 1. Caching Mechanism
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached density maximized dataset for {split}: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Creating density maximized dataset for {split}...")

    # 2. Context Formatting
    # We must apply this to the FULL dataframe to preserve neighbors before filtering
    print("  - Formatting context windows...")
    input_texts = format_context_window(df, context_window=config.context_window)

    # Create a working copy to avoid SettingWithCopy warnings on original df
    df_proc = df.copy()

    # Ensure 'id' column exists for TextDataset
    if "id" not in df_proc.columns:
        df_proc["id"] = (
            df_proc["sentence_id"].astype(str) + "_" + df_proc["token_id"].astype(str)
        )

    df_proc["input_text"] = input_texts

    # Ensure text columns are strings
    df_proc["before"] = df_proc["before"].fillna("").astype(str)
    if "after" in df_proc.columns:
        df_proc["after"] = df_proc["after"].fillna("").astype(str)
        df_proc["target_text"] = df_proc["after"]

    # 3. Semiotic Filtering
    # Filter for tokens containing digits or latin characters
    print("  - Filtering for semiotic tokens...")
    semiotic_mask = df_proc["before"].str.contains(SEMIOTIC_REGEX, regex=True, na=False)
    df_semiotic = df_proc[semiotic_mask].copy()

    print(f"    Raw tokens: {len(df_proc)} -> Semiotic tokens: {len(df_semiotic)}")

    # 4. Class Balancing (Train Only)
    if split == "train":
        print("  - Applying aggressive class balancing...")

        # Get class distribution
        class_counts = df_semiotic["class"].value_counts()

        # Determine target count based on 'DATE' class (or max if DATE missing)
        # We treat DATE as the anchor for "standard" semiotic frequency
        if "DATE" in class_counts:
            target_count = class_counts["DATE"]
        else:
            target_count = class_counts.max()

        print(f"    Target balancing count (Reference: DATE/Max): {target_count}")

        balanced_dfs = []
        for cls_name, count in class_counts.items():
            cls_df = df_semiotic[df_semiotic["class"] == cls_name]

            if count == 0:
                continue

            if count < target_count:
                # Upsample rare classes with replacement
                resampled = cls_df.sample(
                    n=target_count, replace=True, random_state=config.seed
                )
                balanced_dfs.append(resampled)
            elif count > target_count:
                # Downsample dominant classes
                resampled = cls_df.sample(
                    n=target_count, replace=False, random_state=config.seed
                )
                balanced_dfs.append(resampled)
            else:
                balanced_dfs.append(cls_df)

        # Concatenate and shuffle
        df_final = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=config.seed)
            .reset_index(drop=True)
        )
        print(f"    Balanced dataset size: {len(df_final)}")

    else:
        # For validation, we keep the natural distribution of semiotic tokens
        # to evaluate real-world performance on the target subset.
        df_final = df_semiotic.reset_index(drop=True)
        print(f"    Validation dataset size: {len(df_final)}")

    # 5. Save to Cache
    print(f"Saving processed dataset to {cache_path}")
    df_final.to_parquet(cache_path, index=False)

    return df_final


def get_dataloaders(config, char_tokenizer, bpe_tokenizer, load_cached_data=True):
    """
    Prepares DataLoaders for the Hybrid Cascade Tier 2 model.
    """
    print("Initializing DataLoaders...")

    # 1. Load Raw Metadata
    df_train_raw = pd.read_csv(TRAIN_META_PATH)
    df_val_raw = pd.read_csv(VAL_META_PATH)

    # 2. Handle Debug/Subset
    # We must subset by SENTENCE to preserve context integrity
    if config.debug and config.subset_size:
        print(f"Debug Mode: Subsetting to {config.subset_size} sentences.")

        # Train Subset
        train_sents = df_train_raw["sentence_id"].unique()[: config.subset_size]
        df_train_raw = df_train_raw[df_train_raw["sentence_id"].isin(train_sents)]

        # Val Subset (smaller)
        val_subset_size = max(1, int(config.subset_size * 0.2))
        val_sents = df_val_raw["sentence_id"].unique()[:val_subset_size]
        df_val_raw = df_val_raw[df_val_raw["sentence_id"].isin(val_sents)]

    # 3. Create Processed Datasets
    df_train = create_density_maximized_dataset(
        df_train_raw, config, split="train", load_cached_data=load_cached_data
    )
    df_val = create_density_maximized_dataset(
        df_val_raw, config, split="val", load_cached_data=load_cached_data
    )

    # 4. Instantiate PyTorch Datasets
    train_ds = HeterogeneousDataset(
        df_train,
        char_tokenizer,
        bpe_tokenizer,
        max_enc_len=config.max_enc_len,
        max_dec_len=config.max_dec_len,
        mode="train",
    )

    val_ds = HeterogeneousDataset(
        df_val,
        char_tokenizer,
        bpe_tokenizer,
        max_enc_len=config.max_enc_len,
        max_dec_len=config.max_dec_len,
        mode="val",
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
