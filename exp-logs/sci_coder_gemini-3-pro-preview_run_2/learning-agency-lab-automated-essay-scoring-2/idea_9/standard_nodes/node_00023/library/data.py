import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config


def preprocess_text(text):
    """
    Performs minimal text cleaning by normalizing whitespace.
    This preserves the original character of the essay while fixing formatting artifacts.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned text.
    """
    if pd.isna(text) or text == "":
        return ""

    # Replace all whitespace sequences (newlines, tabs, etc.) with a single space
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


class EssayDataset(Dataset):
    """
    PyTorch Dataset for Essay Scoring.
    Handles loading, preprocessing, and tokenization of essay text.
    """

    def __init__(self, df, tokenizer, max_length=1024, is_test=False):
        """
        Args:
            df (pd.DataFrame): Dataframe containing 'full_text' and optionally 'score'/'essay_id'.
            tokenizer: Transformers tokenizer instance.
            max_length (int): Maximum sequence length for truncation.
            is_test (bool): Whether this is the test set (indicates no labels).
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # Extract columns to numpy arrays for faster access during iteration
        self.texts = self.df["full_text"].values
        self.essay_ids = (
            self.df["essay_id"].values if "essay_id" in self.df.columns else None
        )

        # Handle labels for training/validation sets
        if not self.is_test and "score" in self.df.columns:
            self.labels = self.df["score"].values.astype(float)
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = preprocess_text(self.texts[idx])

        # Tokenize without padding (padding is handled in collate_fn for efficiency)
        # We return python lists (return_tensors=None) to allow the collator to pad dynamically
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_tensors=None,
        )

        sample = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        # Add label if available
        if self.labels is not None:
            sample["labels"] = self.labels[idx]

        # Add essay_id if available (crucial for submission mapping)
        if self.essay_ids is not None:
            sample["essay_id"] = self.essay_ids[idx]

        return sample


def get_collate_fn(tokenizer):
    """
    Returns a collate function that dynamically pads the batch to the longest sequence.

    Args:
        tokenizer: Transformers tokenizer with a .pad method.

    Returns:
        function: A collate function compatible with torch.utils.data.DataLoader.
    """

    def collate_fn(batch):
        # batch is a list of dicts from __getitem__
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]

        # Dynamic padding using the tokenizer
        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        batch_out = {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }

        # Stack labels into a tensor if present
        if "labels" in batch[0]:
            batch_out["labels"] = torch.tensor(
                [x["labels"] for x in batch], dtype=torch.float
            )

        # Collect essay_ids as a list if present
        if "essay_id" in batch[0]:
            batch_out["essay_ids"] = [x["essay_id"] for x in batch]

        return batch_out

    return collate_fn


def load_data(tokenizer, split="train", debug=False):
    """
    Loads the dataset from the metadata CSVs defined in Config.

    Args:
        tokenizer: Transformers tokenizer.
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, loads a small subset defined in Config.debug_sample_size.

    Returns:
        EssayDataset: The initialized dataset ready for the DataLoader.
    """
    # Determine file path and mode based on split
    if split == "train":
        path = Config.train_path
        is_test = False
    elif split == "val":
        path = Config.val_path
        is_test = False
    elif split == "test":
        path = Config.test_path
        is_test = True
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")

    # Load Data
    df = pd.read_csv(path)

    # Debugging: subset the data if requested
    if debug:
        df = df.head(Config.debug_sample_size)

    # Initialize Dataset
    dataset = EssayDataset(
        df=df, tokenizer=tokenizer, max_length=Config.max_length, is_test=is_test
    )

    return dataset
