import os
import torch
import pandas as pd
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
)
from library.config import Config
from library.utils import set_seed

# Initialize logger
logger = logging.getLogger("project_logger")


class BaseTextDataset(Dataset):
    """
    Base class for loading text data from Parquet files.
    Handles loading, sampling for debug, and basic setup.
    """

    def __init__(self, data_path, tokenizer, max_len, debug=False, sample_size=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self._load_data(data_path, debug, sample_size)

    def _load_data(self, path, debug, sample_size):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at {path}")

        try:
            df = pd.read_parquet(path, engine="pyarrow")

            # Determine sample size
            limit = None
            if debug:
                limit = Config.DEBUG_SAMPLE_SIZE
            if sample_size is not None:
                limit = sample_size

            if limit is not None and len(df) > limit:
                df = df.sample(n=limit, random_state=Config.SEED).reset_index(drop=True)

            return df
        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError


class LocatorDataset(BaseTextDataset):
    """
    Dataset for the Locator task.
    Randomly removes a word and labels the token preceding the gap.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=Config.MAX_LEN,
        debug=False,
        sample_size=None,
    ):
        super().__init__(data_path, tokenizer, max_len, debug, sample_size)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["sentence"]

        # Simple whitespace tokenization for word selection
        words = sentence.split()

        # Safety check for short sentences
        if len(words) < 3:
            # Return a dummy valid sample if sentence is too short to remove middle word
            # This avoids crashing but won't contribute meaningfully to loss
            encoded = self.tokenizer(
                sentence,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(-100, dtype=torch.long),
            }

        # Randomly select a word to remove (excluding first and last)
        # Target index is relative to the list of words
        remove_idx = np.random.randint(1, len(words) - 1)

        # Create the "gap" sentence
        # The word at remove_idx is removed.
        # The gap is effectively after the word at remove_idx - 1.
        pre_gap_words = words[:remove_idx]
        post_gap_words = words[remove_idx + 1 :]

        # Reconstruct sentence
        pre_gap_str = " ".join(pre_gap_words)
        text_with_gap = pre_gap_str + " " + " ".join(post_gap_words)

        # Tokenize with offsets to find the token boundary
        encoding = self.tokenizer(
            text_with_gap,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Cite solution_lesson_node_00001: Use index labels for Pointer Network.
        # Label is the index of the token preceding the gap.
        label_idx = -100  # Default to ignore_index

        # The gap is after 'pre_gap_str'.
        # We need to find the token that ends exactly at len(pre_gap_str).
        target_char_idx = len(pre_gap_str)

        for i, (start, end) in enumerate(offset_mapping):
            if attention_mask[i] == 0:
                break
            # Check if this token ends at the gap position
            if end == target_char_idx:
                label_idx = i
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label_idx, dtype=torch.long),
        }


class FillerDataset(BaseTextDataset):
    """
    Dataset for the Filler task.
    Randomly masks a word and sets the label to the original word ID.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=Config.MAX_LEN,
        debug=False,
        sample_size=None,
    ):
        super().__init__(data_path, tokenizer, max_len, debug, sample_size)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["sentence"]
        words = sentence.split()

        if len(words) < 3:
            # Fallback for too short sentences
            encoded = self.tokenizer(
                sentence,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.full_like(encoded["input_ids"].squeeze(0), -100),
            }

        # Randomly select word to mask
        mask_idx = np.random.randint(1, len(words) - 1)
        target_word = words[mask_idx]

        # Replace with mask token
        words[mask_idx] = self.tokenizer.mask_token
        masked_sentence = " ".join(words)

        # Tokenize
        encoding = self.tokenizer(
            masked_sentence,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Prepare labels
        labels = torch.full_like(input_ids, -100)

        # Find the mask token index
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) > 0:
            # Get the ID of the target word
            # If word tokenizes to multiple tokens, we take the first one
            # This is a heuristic for the "one word" constraint
            target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
            if len(target_ids) > 0:
                # We only mask one word, so we fill the first mask occurrence
                # In rare cases where the original sentence had a [MASK] token, this might be noisy,
                # but input data is clean text.
                labels[mask_positions[0]] = target_ids[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SubmissionDataset(BaseTextDataset):
    """
    Dataset for the Test set (Submission).
    Returns ID and tokenized sentence (with missing word).
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        max_len=Config.MAX_LEN,
        debug=False,
        sample_size=None,
    ):
        super().__init__(data_path, tokenizer, max_len, debug, sample_size)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence_id = row["id"]
        sentence = row["sentence"]

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "id": sentence_id,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "raw_sentence": sentence,  # Useful for reconstruction
        }


def get_dataloaders(
    task, tokenizer=None, batch_size=None, debug=False, load_cached_data=False
):
    """
    Factory function to get dataloaders for specific tasks.

    Args:
        task (str): 'locator_train', 'filler_train', or 'submission'.
        tokenizer: Pre-initialized tokenizer. If None, creates one.
        batch_size (int): Batch size. If None, uses Config defaults.
        debug (bool): Whether to use a small subset.
        load_cached_data (bool): Placeholder for caching logic (not used for dynamic datasets).

    Returns:
        tuple: (train_loader, val_loader) or (test_loader)
    """
    set_seed(Config.SEED)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    # Determine Batch Size
    if batch_size is None:
        if "locator" in task:
            batch_size = Config.LOCATOR_PARAMS["batch_size"]
        elif "filler" in task:
            batch_size = Config.FILLER_PARAMS["batch_size"]
        else:
            batch_size = 128

    logger.info(
        f"Creating dataloaders for task: {task} | Debug: {debug} | Batch: {batch_size}"
    )

    if task == "locator_train":
        train_ds = LocatorDataset(Config.TRAIN_DATA_PATH, tokenizer, debug=debug)
        val_ds = LocatorDataset(Config.VAL_DATA_PATH, tokenizer, debug=debug)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        return train_loader, val_loader

    elif task == "filler_train":
        train_ds = FillerDataset(Config.TRAIN_DATA_PATH, tokenizer, debug=debug)
        val_ds = FillerDataset(Config.VAL_DATA_PATH, tokenizer, debug=debug)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        return train_loader, val_loader

    elif task == "submission":
        test_ds = SubmissionDataset(Config.TEST_DATA_PATH, tokenizer, debug=debug)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )
        return test_loader

    else:
        raise ValueError(f"Unknown task: {task}")
