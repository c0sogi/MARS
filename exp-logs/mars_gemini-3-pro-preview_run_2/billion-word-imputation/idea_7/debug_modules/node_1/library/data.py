import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config

# -----------------------------------------------------------------------------
# Dataset Classes
# -----------------------------------------------------------------------------


class LocatorDataset(Dataset):
    """
    Dataset for Stage 1: Locator.
    Input: Sentence with one word removed.
    Label: 1 at the token position immediately PRECEDING the gap, 0 otherwise.
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # words is a list of strings (the modified sentence)
        # gap_index is the index in 'words' after which the gap exists.
        words = (
            row["words"].tolist()
            if isinstance(row["words"], np.ndarray)
            else row["words"]
        )
        gap_index = row["gap_index"]

        # Reconstruct sentence for tokenizer
        text = " ".join(words)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: 0 for non-gap, 1 for gap position
        labels = torch.zeros_like(input_ids, dtype=torch.long)

        # Map word index to token index
        word_ids = encoding.word_ids(batch_index=0)

        target_token_idx = -1
        found = False

        # We look for the last token that belongs to the word at 'gap_index'
        # This marks the position immediately preceding the missing word.
        for i, word_id in enumerate(word_ids):
            if word_id == gap_index:
                target_token_idx = i
                found = True
            elif word_id is not None and word_id > gap_index:
                break

        if found:
            labels[target_token_idx] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class InfillerDataset(Dataset):
    """
    Dataset for Stage 2: In-Filler.
    Input: Sentence with <mask> token.
    Label: The original word ID at the mask position.
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["masked_text"]
        target_word = row["target_word"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: -100 everywhere except mask
        labels = torch.full_like(input_ids, -100)

        # Find mask token index
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_indices) > 0:
            # Encode target word (without special tokens)
            target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
            if len(target_ids) > 0:
                # If target word splits into multiple tokens, we predict the first one
                # (Simplification for single-word prediction task)
                labels[mask_indices[0]] = target_ids[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class VerifierDataset(Dataset):
    """
    Dataset for Stage 3: Verifier.
    Input: Complete sentence.
    Label: 1 (Real/Correct) or 0 (Fake/Incorrect).
    """

    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        label = row["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TestDataset(Dataset):
    """
    Dataset for Inference (Test Set).
    """

    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["sentence"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        return {
            "id": row["id"],
            "sentence": text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "offset_mapping": encoding["offset_mapping"].squeeze(0),
        }


# -----------------------------------------------------------------------------
# Data Processing & Caching
# -----------------------------------------------------------------------------


def process_and_cache_data(load_cached_data=True, debug_size=None):
    """
    Generates the specific datasets for Locator, Infiller, and Verifier.
    Caches them as Parquet files.
    """

    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    locator_train_path = os.path.join(cache_dir, "locator_train.parquet")
    locator_val_path = os.path.join(cache_dir, "locator_val.parquet")
    infiller_train_path = os.path.join(cache_dir, "infiller_train.parquet")
    infiller_val_path = os.path.join(cache_dir, "infiller_val.parquet")
    verifier_train_path = os.path.join(cache_dir, "verifier_train.parquet")
    verifier_val_path = os.path.join(cache_dir, "verifier_val.parquet")

    paths = [
        locator_train_path,
        locator_val_path,
        infiller_train_path,
        infiller_val_path,
        verifier_train_path,
        verifier_val_path,
    ]

    # Check if all exist
    if load_cached_data and all(os.path.exists(p) for p in paths):
        print("Loading cached datasets from parquet...")
        return (
            pd.read_parquet(locator_train_path),
            pd.read_parquet(locator_val_path),
            pd.read_parquet(infiller_train_path),
            pd.read_parquet(infiller_val_path),
            pd.read_parquet(verifier_train_path),
            pd.read_parquet(verifier_val_path),
        )

    print("Generating datasets from scratch...")

    # Load raw metadata
    df_train_full = pd.read_parquet(Config.TRAIN_META_PATH)
    df_val_full = pd.read_parquet(Config.VAL_META_PATH)

    # Sample
    train_size = Config.get_train_size()
    val_size = Config.get_val_size()

    if debug_size:
        train_size = min(train_size, debug_size)
        val_size = min(val_size, debug_size)

    df_train = df_train_full.sample(n=train_size, random_state=Config.SEED).reset_index(
        drop=True
    )
    df_val = df_val_full.sample(n=val_size, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Helper to generate data
    def generate_task_data(df, is_train=True):
        locator_rows = []
        infiller_rows = []
        verifier_rows = []

        # Collect all removed words to use as "Wrong Word" negatives for Verifier
        removed_words_pool = []

        # First pass: Generate Locator and Infiller data, and collect words
        temp_data = []

        for idx, row in df.iterrows():
            sentence = row["sentence"]
            words = sentence.split()

            if len(words) < 3:
                continue  # Cannot remove internal word

            # Pick random word to remove (not first, not last)
            remove_idx = random.randint(1, len(words) - 2)
            target_word = words[remove_idx]
            removed_words_pool.append(target_word)

            # --- Locator Data ---
            # Modified sentence (list of words)
            mod_words = words[:remove_idx] + words[remove_idx + 1 :]
            # Gap is after words[remove_idx-1], which is now at index remove_idx-1 in mod_words
            gap_index = remove_idx - 1

            locator_rows.append(
                {
                    "id": row["id"],
                    "words": mod_words,  # Store as list
                    "gap_index": gap_index,
                }
            )

            # --- Infiller Data ---
            # Masked sentence string
            masked_words = words.copy()
            masked_words[remove_idx] = "<mask>"
            masked_text = " ".join(masked_words)

            infiller_rows.append(
                {
                    "id": row["id"],
                    "masked_text": masked_text,
                    "target_word": target_word,
                }
            )

            # Store for Verifier generation
            temp_data.append(
                {
                    "original_words": words,
                    "remove_idx": remove_idx,
                    "target_word": target_word,
                }
            )

        # Second pass: Generate Verifier data (using pool)
        for item in temp_data:
            words = item["original_words"]
            remove_idx = item["remove_idx"]
            target_word = item["target_word"]

            # 1. Positive Sample
            verifier_rows.append({"text": " ".join(words), "label": 1})

            # 2. Negative Position (Hard Negative 1)
            # Insert target_word at wrong index
            mod_words = words[:remove_idx] + words[remove_idx + 1 :]
            possible_indices = [j for j in range(len(mod_words) + 1) if j != remove_idx]

            if possible_indices:
                wrong_idx = random.choice(possible_indices)
                neg_pos_words = (
                    mod_words[:wrong_idx] + [target_word] + mod_words[wrong_idx:]
                )
                verifier_rows.append({"text": " ".join(neg_pos_words), "label": 0})

            # 3. Negative Word (Hard Negative 2)
            # Insert wrong word at correct index
            if removed_words_pool:
                wrong_word = random.choice(removed_words_pool)
                # Ensure it's not the same (unlikely but possible)
                attempts = 0
                while (
                    wrong_word == target_word
                    and len(removed_words_pool) > 1
                    and attempts < 5
                ):
                    wrong_word = random.choice(removed_words_pool)
                    attempts += 1

                neg_word_words = (
                    mod_words[:remove_idx] + [wrong_word] + mod_words[remove_idx:]
                )
                verifier_rows.append({"text": " ".join(neg_word_words), "label": 0})

        return (
            pd.DataFrame(locator_rows),
            pd.DataFrame(infiller_rows),
            pd.DataFrame(verifier_rows),
        )

    print("Processing training data...")
    loc_train, inf_train, ver_train = generate_task_data(df_train, is_train=True)
    print("Processing validation data...")
    loc_val, inf_val, ver_val = generate_task_data(df_val, is_train=False)

    print("Saving to cache...")
    loc_train.to_parquet(locator_train_path)
    loc_val.to_parquet(locator_val_path)
    inf_train.to_parquet(infiller_train_path)
    inf_val.to_parquet(infiller_val_path)
    ver_train.to_parquet(verifier_train_path)
    ver_val.to_parquet(verifier_val_path)

    return loc_train, loc_val, inf_train, inf_val, ver_train, ver_val


# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------


def get_dataloaders(load_cached_data=True, debug_size=None):
    """
    Returns DataLoaders for all three stages and the loaded tokenizers.
    """
    # 1. Prepare Data
    loc_train_df, loc_val_df, inf_train_df, inf_val_df, ver_train_df, ver_val_df = (
        process_and_cache_data(load_cached_data, debug_size)
    )

    # 2. Tokenizers
    print("Loading tokenizers...")
    tokenizer_deberta = AutoTokenizer.from_pretrained(
        Config.LOCATOR_MODEL_NAME, use_fast=True
    )
    tokenizer_roberta = AutoTokenizer.from_pretrained(
        Config.INFILLER_MODEL_NAME, use_fast=True
    )

    # 3. Datasets
    print("Creating Datasets...")
    train_ds_loc = LocatorDataset(loc_train_df, tokenizer_deberta, Config.MAX_LENGTH)
    val_ds_loc = LocatorDataset(loc_val_df, tokenizer_deberta, Config.MAX_LENGTH)

    train_ds_inf = InfillerDataset(inf_train_df, tokenizer_roberta, Config.MAX_LENGTH)
    val_ds_inf = InfillerDataset(inf_val_df, tokenizer_roberta, Config.MAX_LENGTH)

    train_ds_ver = VerifierDataset(ver_train_df, tokenizer_deberta, Config.MAX_LENGTH)
    val_ds_ver = VerifierDataset(ver_val_df, tokenizer_deberta, Config.MAX_LENGTH)

    # 4. DataLoaders
    print("Creating DataLoaders...")
    train_loader_loc = DataLoader(
        train_ds_loc,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_loc = DataLoader(
        val_ds_loc,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_loader_inf = DataLoader(
        train_ds_inf,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_inf = DataLoader(
        val_ds_inf,
        batch_size=Config.INFILLER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    train_loader_ver = DataLoader(
        train_ds_ver,
        batch_size=Config.VERIFIER_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader_ver = DataLoader(
        val_ds_ver,
        batch_size=Config.VERIFIER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return {
        "locator": (train_loader_loc, val_loader_loc),
        "infiller": (train_loader_inf, val_loader_inf),
        "verifier": (train_loader_ver, val_loader_ver),
        "tokenizers": {"deberta": tokenizer_deberta, "roberta": tokenizer_roberta},
    }


def get_test_dataloader():
    """
    Creates a DataLoader for the test set (Locator stage input).
    """
    df_test = pd.read_parquet(Config.TEST_META_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.LOCATOR_MODEL_NAME, use_fast=True)

    ds = TestDataset(df_test, tokenizer, Config.MAX_LENGTH)

    return DataLoader(
        ds,
        batch_size=Config.LOCATOR_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
