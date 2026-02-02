import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from library.config import Config


class DynamicMaskingDataset(Dataset):
    def __init__(self, df, tokenizer, mode="train", max_len=Config.MAX_LEN):
        self.data = df
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.mask_token = tokenizer.mask_token
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["sentence"]

        # -------------------------------------------------------
        # TEST MODE
        # -------------------------------------------------------
        if self.mode == "test":
            # In test mode, the word is already removed.
            # We return inputs for the Locator and offset mapping for reconstruction.
            inputs = self.tokenizer(
                sentence,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "offset_mapping": inputs["offset_mapping"].squeeze(0),
                "id": row["id"],
                "sentence": sentence,
            }

        # -------------------------------------------------------
        # TRAIN / VAL MODE
        # -------------------------------------------------------
        words = sentence.split()

        # Handle short sentences gracefully
        if len(words) < 3:
            words = ["This", "is", "dummy", "."]

        # Select gap index
        if self.mode == "val":
            # Deterministic selection for validation stability
            rng = np.random.RandomState(idx)
            valid_indices = list(range(1, len(words) - 1))
            if not valid_indices:
                valid_indices = [1]
            gap_idx = rng.choice(valid_indices)
        else:
            # Training: Prioritize single-token words for cleaner Filler training
            valid_indices = list(range(1, len(words) - 1))
            if not valid_indices:
                valid_indices = [1]

            gap_idx = valid_indices[0]  # Default
            random.shuffle(valid_indices)

            # Try to find a word that tokenizes to a single token
            for try_idx in valid_indices[:5]:
                word_to_check = words[try_idx]
                # Prepend space because it's a middle-of-sentence word
                tokens = self.tokenizer.encode(
                    " " + word_to_check, add_special_tokens=False
                )
                if len(tokens) == 1:
                    gap_idx = try_idx
                    break

        removed_word = words[gap_idx]
        pre_gap_words = words[:gap_idx]
        post_gap_words = words[gap_idx + 1 :]

        # Construct Inputs
        locator_text = " ".join(pre_gap_words + post_gap_words)
        filler_text = " ".join(pre_gap_words + [self.mask_token] + post_gap_words)

        # Tokenize
        locator_inputs = self.tokenizer(
            locator_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        filler_inputs = self.tokenizer(
            filler_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # -------------------------------------------------------
        # Compute Locator Label (Index of token preceding gap)
        # -------------------------------------------------------
        # The insertion point is at the end of the pre_gap string.
        char_insertion_point = len(" ".join(pre_gap_words))

        offsets = locator_inputs["offset_mapping"].squeeze(0)
        locator_label = 0  # Default to CLS

        seq_len = locator_inputs["attention_mask"].sum().item()

        found = False
        for i in range(seq_len):
            start, end = offsets[i]
            # Check if token ends exactly at the insertion point
            if end == char_insertion_point:
                locator_label = i
                found = True
                break
            # Fallback for tokenizer edge cases (e.g., next token starts at point)
            if start == char_insertion_point:
                locator_label = i - 1
                found = True
                break

        # -------------------------------------------------------
        # Compute Filler Label (Token ID of removed word)
        # -------------------------------------------------------
        word_tokens = self.tokenizer.encode(
            " " + removed_word, add_special_tokens=False
        )
        target_token_id = word_tokens[0] if word_tokens else self.tokenizer.unk_token_id

        filler_input_ids = filler_inputs["input_ids"].squeeze(0)
        filler_labels = torch.full(filler_input_ids.shape, -100, dtype=torch.long)

        mask_indices = (filler_input_ids == self.mask_token_id).nonzero(as_tuple=True)[
            0
        ]
        if len(mask_indices) > 0:
            filler_labels[mask_indices[0]] = target_token_id

        return {
            "locator_input_ids": locator_inputs["input_ids"].squeeze(0),
            "locator_attention_mask": locator_inputs["attention_mask"].squeeze(0),
            "locator_labels": torch.tensor(locator_label, dtype=torch.long),
            "filler_input_ids": filler_input_ids,
            "filler_attention_mask": filler_inputs["attention_mask"].squeeze(0),
            "filler_labels": filler_labels,
        }


def get_dataloaders(load_cached_data=True, debug=False):
    """
    Prepares DataLoaders with caching for sampled subsets.
    """
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    cached_train_path = os.path.join(Config.CACHE_DIR, "train_subset.parquet")
    cached_val_path = os.path.join(Config.CACHE_DIR, "val_subset.parquet")

    # -------------------------------------------------------
    # Load Training Data
    # -------------------------------------------------------
    df_train = None
    if load_cached_data and os.path.exists(cached_train_path):
        print(f"Loading cached training data from {cached_train_path}")
        df_train = pd.read_parquet(cached_train_path)
    else:
        print(f"Loading raw training data from {Config.TRAIN_PATH}")
        df_full = pd.read_parquet(Config.TRAIN_PATH)

        sample_size = Config.MAX_TRAIN_SAMPLES
        if debug:
            sample_size = 1000

        if len(df_full) > sample_size:
            df_train = df_full.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)
        else:
            df_train = df_full

        print(f"Caching training subset to {cached_train_path}")
        df_train.to_parquet(cached_train_path)
        del df_full

    # -------------------------------------------------------
    # Load Validation Data
    # -------------------------------------------------------
    df_val = None
    if load_cached_data and os.path.exists(cached_val_path):
        print(f"Loading cached validation data from {cached_val_path}")
        df_val = pd.read_parquet(cached_val_path)
    else:
        print(f"Loading raw validation data from {Config.VAL_PATH}")
        df_full_val = pd.read_parquet(Config.VAL_PATH)

        sample_size = Config.VAL_SAMPLES
        if debug:
            sample_size = 100

        if len(df_full_val) > sample_size:
            df_val = df_full_val.sample(
                n=sample_size, random_state=Config.SEED
            ).reset_index(drop=True)
        else:
            df_val = df_full_val

        print(f"Caching validation subset to {cached_val_path}")
        df_val.to_parquet(cached_val_path)
        del df_full_val

    # -------------------------------------------------------
    # Load Test Data
    # -------------------------------------------------------
    print(f"Loading test data from {Config.TEST_PATH}")
    df_test = pd.read_parquet(Config.TEST_PATH)
    if debug:
        df_test = df_test.iloc[:100]

    # -------------------------------------------------------
    # Create Datasets & Loaders
    # -------------------------------------------------------
    train_dataset = DynamicMaskingDataset(df_train, tokenizer, mode="train")
    val_dataset = DynamicMaskingDataset(df_val, tokenizer, mode="val")
    test_dataset = DynamicMaskingDataset(df_test, tokenizer, mode="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
