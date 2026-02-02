import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm
from library.config import Config
from library.utils import set_seed

# Ensure parallel tokenizers don't cause deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SyntheticCorruptor:
    """
    Handles the logic for synthetically corrupting sentences by removing a single word
    to create training examples for the Locator and Infiller models.
    """

    @staticmethod
    def corrupt_batch(sentences, seed=42):
        """
        Corrupts a list of sentences.
        Returns a DataFrame with columns:
        ['original_sentence', 'corrupted_sentence', 'missing_word', 'gap_char_index']
        """
        random.seed(seed)
        data = []

        for sent in sentences:
            words = sent.split()
            # Constraint: Never remove first or last word
            if len(words) < 3:
                continue

            # Pick random index to remove (1 to len-2)
            gap_idx = random.randint(1, len(words) - 2)
            missing_word = words[gap_idx]

            # Reconstruct sentence
            pre_gap = words[:gap_idx]
            post_gap = words[gap_idx + 1 :]

            # Join with spaces
            pre_gap_str = " ".join(pre_gap)
            post_gap_str = " ".join(post_gap)

            corrupted_sentence = f"{pre_gap_str} {post_gap_str}"

            # The gap is located immediately after pre_gap_str.
            # In the corrupted sentence, this corresponds to the length of pre_gap_str.
            # Example: "A B C". Remove B. -> "A C". pre="A". len=1.
            # Gap is at index 1 (the space). The token ending at 1 is 'A'.
            gap_char_index = len(pre_gap_str)

            data.append(
                {
                    "original_sentence": sent,
                    "corrupted_sentence": corrupted_sentence,
                    "missing_word": missing_word,
                    "gap_char_index": gap_char_index,
                }
            )

        return pd.DataFrame(data)


class LocatorDataset(Dataset):
    """
    Dataset for Stage 1: Locator (DeBERTa-v3).
    Target: Binary sequence label (1 for the token immediately preceding the gap).
    """

    def __init__(self, df, tokenizer, max_len=256):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["corrupted_sentence"]
        gap_char_idx = row["gap_char_index"]

        # Tokenize with offset mapping to align character positions to tokens
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)

        # Create binary labels
        labels = torch.zeros_like(input_ids, dtype=torch.float)

        # Find the token that ends exactly at the gap_char_index
        # The gap is between words, so we look for the word ending just before the space
        found = False
        for i, (start, end) in enumerate(offsets):
            if end == gap_char_idx:
                labels[i] = 1.0
                found = True
                break
            # Fallback for complex tokenization edge cases:
            # if token contains the split point (rare with space splitting)
            if start < gap_char_idx and end > gap_char_idx:
                labels[i] = 1.0
                found = True
                break

        # If truncation cut off the gap, no label is set (all zeros).
        # This is acceptable as noise or can be filtered, but rare with MAX_LEN=256.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InfillerDataset(Dataset):
    """
    Dataset for Stage 2: In-Filler (RoBERTa-Large).
    Input: Sentence with <mask> inserted at the gap.
    Target: The token ID of the missing word.
    """

    def __init__(self, df, tokenizer, max_len=256):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        corrupted_text = row["corrupted_sentence"]
        gap_char_idx = row["gap_char_index"]
        missing_word = row["missing_word"]

        # Construct masked sentence
        # corrupted_text[:gap] is the pre-gap part.
        # We insert " <mask>" (space + mask) to ensure correct tokenization context
        part1 = corrupted_text[:gap_char_idx]
        part2 = corrupted_text[gap_char_idx:]  # Includes the space and following words

        # RoBERTa mask token
        mask_token = self.tokenizer.mask_token
        masked_text = f"{part1} {mask_token}{part2}"

        # Tokenize input
        encoding = self.tokenizer(
            masked_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Prepare labels (MLM objective)
        labels = torch.full(input_ids.shape, -100, dtype=torch.long)

        # Find the mask token index
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) > 0:
            mask_idx = mask_indices[
                0
            ]  # Take the first mask if multiple (should be one)

            # Encode the target word.
            # We add a leading space because it was in the middle of a sentence.
            target_encoding = self.tokenizer(
                " " + missing_word, add_special_tokens=False
            )
            target_ids = target_encoding["input_ids"]

            if len(target_ids) > 0:
                # We predict the first token of the missing word.
                # While some words split into multiple tokens, predicting the first
                # is the primary objective for the single-mask heuristic.
                labels[mask_idx] = target_ids[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TestDataset(Dataset):
    """
    Dataset for Inference.
    Returns input_ids and the raw sentence ID.
    """

    def __init__(self, df, tokenizer, max_len=256):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["sentence"]
        sample_id = row["id"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "offset_mapping": encoding["offset_mapping"].squeeze(0),
            "id": sample_id,
            "raw_text": text,
        }


def process_and_cache_data(
    source_path, cache_path, sample_size, seed, desc="Processing"
):
    """
    Loads raw data, samples it, applies synthetic corruption, and caches the result.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached {desc} data from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Verify schema matches expected processed output
        if "corrupted_sentence" in df.columns:
            return df
        print(
            f"Cached data at {cache_path} is invalid (missing 'corrupted_sentence'). Regenerating..."
        )

    print(f"Generating {desc} data from scratch...")
    # Load raw sentences
    df = pd.read_parquet(source_path)

    # Sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Corrupt
    sentences = df["sentence"].tolist()
    processed_df = SyntheticCorruptor.corrupt_batch(sentences, seed=seed)

    # Cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    processed_df.to_parquet(cache_path, index=False)
    print(f"Saved {len(processed_df)} processed samples to {cache_path}")

    return processed_df


def get_dataloaders(debug=False):
    """
    Main entry point to get DataLoaders for the pipeline.
    Handles caching, tokenization, and splitting.

    Args:
        debug (bool): If True, uses a small subset of data for quick validation.

    Returns:
        dict: Contains 'train_locator', 'val_locator', 'train_infiller', 'val_infiller', 'test' dataloaders.
    """
    set_seed(Config.SEED)

    # Determine sizes
    train_size = Config.DEBUG_SIZE if debug else Config.TRAIN_SIZE
    val_size = Config.DEBUG_SIZE if debug else Config.VAL_SIZE

    # ---------------------------------------------------------
    # 1. Prepare Dataframes (with Caching)
    # ---------------------------------------------------------
    # We use separate caches for Locator and Infiller to match Config structure,
    # though logically they could share the same corruption source.
    # Here we generate them independently to allow different random corruptions if needed.

    # Locator Data
    df_train_loc = process_and_cache_data(
        Config.TRAIN_META_PATH,
        Config.LOCATOR_TRAIN_CACHE,
        train_size,
        Config.SEED,
        "Locator Train",
    )
    df_val_loc = process_and_cache_data(
        Config.VAL_META_PATH,
        Config.LOCATOR_VAL_CACHE,
        val_size,
        Config.SEED,
        "Locator Val",
    )

    # Infiller Data (Can reuse the same logic/seed or different)
    # Using seed+1 to ensure diversity if we wanted, but keeping same seed for consistency is fine.
    df_train_inf = process_and_cache_data(
        Config.TRAIN_META_PATH,
        Config.INFILLER_TRAIN_CACHE,
        train_size,
        Config.SEED,
        "Infiller Train",
    )
    df_val_inf = process_and_cache_data(
        Config.VAL_META_PATH,
        Config.INFILLER_VAL_CACHE,
        val_size,
        Config.SEED,
        "Infiller Val",
    )

    # Test Data
    if os.path.exists(Config.TEST_CACHE):
        df_test = pd.read_parquet(Config.TEST_CACHE)
    else:
        df_test = pd.read_parquet(Config.TEST_META_PATH)
        if debug:
            df_test = df_test.head(Config.DEBUG_SIZE)
        df_test.to_parquet(Config.TEST_CACHE, index=False)

    # ---------------------------------------------------------
    # 2. Initialize Tokenizers
    # ---------------------------------------------------------
    print("Initializing tokenizers...")
    tokenizer_loc = AutoTokenizer.from_pretrained(
        Config.LOCATOR_MODEL_NAME, use_fast=True
    )
    tokenizer_inf = AutoTokenizer.from_pretrained(
        Config.INFILLER_MODEL_NAME, use_fast=True
    )

    # ---------------------------------------------------------
    # 3. Create Datasets
    # ---------------------------------------------------------
    train_ds_loc = LocatorDataset(df_train_loc, tokenizer_loc, Config.MAX_LEN)
    val_ds_loc = LocatorDataset(df_val_loc, tokenizer_loc, Config.MAX_LEN)

    train_ds_inf = InfillerDataset(df_train_inf, tokenizer_inf, Config.MAX_LEN)
    val_ds_inf = InfillerDataset(df_val_inf, tokenizer_inf, Config.MAX_LEN)

    # Test dataset uses Locator tokenizer (Stage 1 input)
    test_ds = TestDataset(df_test, tokenizer_loc, Config.MAX_LEN)

    # ---------------------------------------------------------
    # 4. Create DataLoaders
    # ---------------------------------------------------------
    # Locator uses default collation (stacking tensors) because we pad in Dataset
    train_args = {
        "batch_size": Config.TRAIN_BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
    }
    val_args = {
        "batch_size": Config.INF_BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
    }

    train_loader_loc = DataLoader(train_ds_loc, shuffle=True, **train_args)
    val_loader_loc = DataLoader(val_ds_loc, shuffle=False, **val_args)

    train_loader_inf = DataLoader(train_ds_inf, shuffle=True, **train_args)
    val_loader_inf = DataLoader(val_ds_inf, shuffle=False, **val_args)

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        batch_size=Config.INF_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    print(
        f"DataLoaders ready. Train Size: {len(df_train_loc)}, Val Size: {len(df_val_loc)}"
    )

    return {
        "train_locator": train_loader_loc,
        "val_locator": val_loader_loc,
        "train_infiller": train_loader_inf,
        "val_infiller": val_loader_inf,
        "test": test_loader,
        "tokenizer_locator": tokenizer_loc,
        "tokenizer_infiller": tokenizer_inf,
    }
